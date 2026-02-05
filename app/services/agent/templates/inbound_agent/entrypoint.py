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

from app.services.agent import agent_metrics
from app.services.agent.factory import AgentFactory, load_agent_config
from app.models.call_models import save_usage_summary

inbound_server = AgentServer()

logger = logging.getLogger(__name__)

usage_collector = metrics.UsageCollector()

#TODO Fetch from db
@dataclass
class UserData:
    id: str
    name: str
    email: str
    phone: str

ud = UserData(
    id="693a6b84dc31118495e34e27",
    name="Gurtej Singh",
    email="gurtej@gmail.com",
    phone="+917460015555"
)

@inbound_server.rtc_session(agent_name="inbound-agent")
async def inbound_entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Example: resolve from headers / room metadata / API
    # customer_id = ctx.job.metadata.get("customer_id")
    # agent_id = ctx.job.metadata.get("agent_id")
    agent_config = await load_agent_config(ud,"eminence") #TODO HAZARD pass uer&agent ID
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
            "stt_metrics": agent_metrics.handle_stt_metrics,
            "llm_metrics": agent_metrics.handle_llm_metrics,
            "tts_metrics": agent_metrics.handle_tts_metrics,
            "vad_metrics": agent_metrics.handle_vad_metrics,
            "eou_metrics": agent_metrics.handle_eou_metrics,
        }
        
        handler = metrics_handlers.get(ev.metrics.type)
        if handler:
            handler(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        await save_usage_summary(session.call_id, summary)

    ctx.add_shutdown_callback(log_usage)

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
