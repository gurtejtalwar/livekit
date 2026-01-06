import logging
import asyncio

from livekit.agents import (
    AgentSession,
    AgentServer,
    AutoSubscribe,
    JobContext,
    cli,
    RoomInputOptions,
    metrics,
    MetricsCollectedEvent,
    UserStateChangedEvent

)
from livekit.plugins import noise_cancellation


logger = logging.getLogger("inbound-agent")
for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)



from dotenv import load_dotenv

logger = logging.getLogger("inbound-agent")

from dataclasses import dataclass
from app.services.agent.factory import AgentFactory, load_agent_config

load_dotenv(override=True)

usage_collector = metrics.UsageCollector()
agent_server = AgentServer()

@dataclass
class UserData:
    name: str
    email: str
    phone: str

ud = UserData(
    name="Gurtej Singh",
    email="gurtej@gmail.com",
    phone="+917460015555"
)

@agent_server.rtc_session(agent_name="inbound-agent")
async def inbound_entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Example: resolve from headers / room metadata / API
    # customer_id = ctx.job.metadata.get("customer_id")
    # agent_id = ctx.job.metadata.get("agent_id")

    agent_config = await load_agent_config(ud,"some-agent-id") #TODO HAZARD pass uer&agent ID
    agent = AgentFactory.create_agent(agent_config)
    session = AgentSession(preemptive_generation=True, 
                           userdata=ud,
                           user_away_timeout=10)


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

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

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


if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    cli.run_app(agent_server)
