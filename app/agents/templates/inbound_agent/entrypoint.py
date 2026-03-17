import asyncio
import uuid
import logging
import json

from dataclasses import dataclass

from livekit import api
from livekit.plugins import noise_cancellation
from livekit.agents import (metrics,
                            function_tool,
                            JobContext,
                            AutoSubscribe,
                            AgentServer,
                            AgentSession,
                            MetricsCollectedEvent,
                            RoomInputOptions)

from app.agents import agent_metrics, UserData
from app.agents.factory.agent import AgentFactory
from app.models import call_models
from app.shared import schemas
from app.shared import settings
from app.utils.requests import _request

inbound_server = AgentServer()

logger = logging.getLogger(__name__)

settings = settings.get_settings()
usage_collector = metrics.UsageCollector()

#TODO Fetch from db
ud = UserData(
    user_id="6992f9020296c31229cfacf0",
    name="Gurtej Singh",
    email="gurtej@gmail.com",
    phone="+917460015555"
)

class BGTasks:
    def __init__(self):
        self.tasks = []
    
    def add(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.append(task)
    
    async def wait_all(self):
        await asyncio.gather(*self.tasks)

@inbound_server.rtc_session(agent_name="inbound-agent")
async def inbound_entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Example: resolve from headers / room metadata / API
    metadata = json.loads(ctx.job.metadata)
    agent_id = metadata["agent_id"]
    ud.agent_id = agent_id
    ud.user_timezone = ud.user_current_time = None
    print("*"*10,"\n")
    print(f"Received Agent Metadata:\n {metadata}\n")
    print("*"*10,"\n")
    print(f"Starting session with agent_id: {agent_id}")
    remote_participant = await ctx.wait_for_participant()
    if remote_participant.attributes.get("sip.phoneNumber", None):
        ud.user_timezone, ud.user_current_time = await AgentFactory.get_time_from_phone(remote_participant.attributes["sip.phoneNumber"])
    user_data = await AgentFactory.get_user_data(agent_id)
    agent_config = await AgentFactory.load_agent_config(user_data,agent_id)
    user_data.outbound_trunk_id = agent_config.outbound_trunk_id
    user_data.human_escalation_phone = agent_config.human_phone_number
    agent_config.call_type = metadata["call_type"]
    agent_config.ctx = ctx

    # Start Egress Service if recording allowed
    recording_url = None
    if agent_config.allow_recording is True:
        recording_uid = uuid.uuid4()
        filepath = f"recordings/{recording_uid}.mp3"
        recording_url= f"{settings.AWS_BUCKET_ENDPOINT_RECORDING}/{filepath}"
        try:
            # Use explicit file_type to ensure the protobuf is well-formed for the server
            file_output = api.EncodedFileOutput(
                file_type=api.EncodedFileType.MP3,
                filepath=f"recordings/inbound_{ctx.room.name}.mp3",
                s3=api.S3Upload(
                    access_key=settings.AWS_ACCESS_KEY,
                    secret=settings.AWS_SECRET_KEY,
                    bucket=settings.AWS_BUCKET_NAME_RECORDING,
                    endpoint=settings.AWS_BUCKET_ENDPOINT_RECORDING,
                    region=settings.AWS_REGION
                )
            )

            # 2. Pass that list to the RoomCompositeEgressRequest
            egress_info = await ctx.api.egress.start_room_composite_egress(
                api.RoomCompositeEgressRequest(
                    room_name=ctx.room.name,
                    audio_only=True,
                    file_outputs=[file_output] # This must be a list
                )
            )
            logger.info(f"Started egress for room {ctx.room.name}")
        except Exception as e:
            logger.exception("Failed to start egress: %s", e)


    session = AgentSession(
        preemptive_generation=True, 
        userdata=user_data,
        user_away_timeout=10)
    
    agent = AgentFactory.from_config(agent_config)

    inactivity_task: asyncio.Task | None = None

    async def user_presence_task():
        # try to ping the user 3 times, if we get no answer, close the session
        logger.info("User presence task started due to inactivity.")
        for _ in range(3):
            await session.generate_reply(
                instructions=(
                    "The user has been inactive. Politely check if the user is still present."
                )
            )
            await asyncio.sleep(10)
        logger.info("Session closed due to user inactivity.")
        session.shutdown()

    # @session.on("user_state_changed")
    # def _user_state_changed(ev: UserStateChangedEvent):
    #     nonlocal inactivity_task
    #     if ev.new_state == "away":
    #         inactivity_task = asyncio.create_task(user_presence_task())
    #         return inactivity_task

    #     # ev.new_state: listening, speaking, ..
    #     if ev.new_state=="speaking" and inactivity_task is not None:
    #         inactivity_task.cancel()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)
        
        metrics_handlers = {
            "stt_metrics": agent_metrics.print_stt_metrics,
            "llm_metrics": agent_metrics.print_llm_metrics,
            "tts_metrics": agent_metrics.print_tts_metrics,
            "vad_metrics": agent_metrics.print_vad_metrics,
            "eou_metrics": agent_metrics.print_eou_metrics,
        }
        
        handler = metrics_handlers.get(ev.metrics.type)
        if handler:
            handler(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        await call_models.save_usage_summary(session.userdata.call_id, summary)

    async def post_call_tasks():
        await log_usage()
        await wait_for_egress()
        await call_models.on_call_ended(agent_config, session, recording_url)



        await post_call_analysis(session)

    ctx.add_shutdown_callback(post_call_tasks)


    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )
    await ctx.wait_for_participant()
    
    # If a graph workflow is active, the first AgentTask node (e.g. 'Greetings')
    # will naturally handle generating the system greeting via LLM prompt rules.
    if not getattr(agent_config, 'workflow_graph_json', None):
        await session.say(agent_config.greeting, allow_interruptions=False)
        # await session.generate_reply(instructions="Confirm the user is connected and greet them warmly.")

    async def wait_for_egress():
        for _ in range(10): # Try for 30 seconds
            req = api.ListEgressRequest(egress_id=egress_info.egress_id)
            status_info = await ctx.api.egress.list_egress(req)
            current = status_info.items[0]           
            if current.status == api.EgressStatus.EGRESS_COMPLETE:
                return
            await asyncio.sleep(3)
    if agent_config.max_duration and agent_config.max_duration>0:
        asyncio.create_task(session_timeout_monitor(ctx, agent_config.max_duration))
async def post_call_analysis(session: AgentSession):
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": f"{settings.P1_ISC_API_KEY}"
    }
    transcript = call_models.build_transcript_string(session.history.items)

    res = await _request(
        "POST",
        f"{settings.P1_ISC_URL}/post-call-analysis",
        headers=headers,
        json={
            "transcript": transcript
        })
    analysis = schemas.PostCallAnalysis(**res["data"])
    logger.info(f"Post-call analysis: {analysis}")
    await call_models.save_analysis(session.userdata.call_id, analysis)


async def session_timeout_monitor(ctx: JobContext, timeout: int):
    await asyncio.sleep(timeout)
    print(f"Hard timeout reached ({timeout}s). Shutting down.")
    
    # You can play a "Goodbye" TTS here if you have a reference to the session
    # then kill the job.
    ctx.shutdown()