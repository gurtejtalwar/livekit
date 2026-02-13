import asyncio
import logging

from dataclasses import dataclass

from livekit.plugins import noise_cancellation
from livekit.agents import (metrics,
                            JobContext,
                            AutoSubscribe,
                            AgentServer,
                            AgentSession,
                            MetricsCollectedEvent,
                            RoomInputOptions)

from app.agent import agent_metrics, UserData
from app.agent.factory.agent import AgentFactory
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
    id="693a6b84dc31118495e34e27",
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
    agent_id = ctx.job.metadata
    agent_config = await AgentFactory.load_agent_config(ud,agent_id) #TODO HAZARD pass uer&agent ID
    agent_config.ctx = ctx
    session = AgentSession(preemptive_generation=True, 
                           userdata=ud,
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
        # if handler:
        #     handler(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        await call_models.save_usage_summary(session.userdata.call_id, summary)

    async def post_call_tasks():
        await log_usage()
        await post_call_analysis(session)
        await call_models.on_call_ended(agent_config, session)

    ctx.add_shutdown_callback(post_call_tasks)


    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )

    # await session.say(agent_config.greeting)
    await session.generate_reply(instructions="Confirm the user is connected and greet them warmly.")

    
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

