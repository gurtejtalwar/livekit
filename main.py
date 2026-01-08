import logging
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich import box

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

from app.services.agent.factory import AgentFactory, load_agent_config


logger = logging.getLogger("inbound-agent")
for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


logger = logging.getLogger("inbound-agent")
console = Console()

load_dotenv(override=True)

usage_collector = metrics.UsageCollector()
agent_server = AgentServer()


def format_timestamp(timestamp: int) -> str:
    """Return raw timestamp."""
    return str(timestamp)


def create_table(title: str) -> Table:
    """Create a formatted table with standard styling."""
    return Table(
        title=title,
        box=box.ROUNDED,
        highlight=True,
        show_header=True,
        header_style="bold cyan"
    )


def display_metrics_table(table: Table):
    """Display metrics table with spacing."""
    console.print("\n")
    console.print(table)
    console.print("\n")


def handle_stt_metrics(metrics_data: metrics.STTMetrics):
    """Handle STT metrics display."""
    table = create_table("[bold blue]STT Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Label", str(metrics_data.label))
    table.add_row("Request ID", str(metrics_data.request_id))
    table.add_row("Timestamp", timestamp)
    table.add_row("Audio Duration", f"[white]{metrics_data.audio_duration / 1000:.4f}[/white]s")
    table.add_row("Streamed", "✓" if metrics_data.streamed else "✗")

    display_metrics_table(table)


def handle_llm_metrics(metrics_data: metrics.LLMMetrics):
    """Handle LLM metrics display."""
    table = create_table("[bold blue]LLM Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Label", str(metrics_data.label))
    table.add_row("Request ID", str(metrics_data.request_id))
    table.add_row("Timestamp", timestamp)
    table.add_row("Duration", f"[white]{metrics_data.duration / 1000:.4f}[/white]s")
    table.add_row("Time to First Token", f"[white]{metrics_data.ttft / 1000:.4f}[/white]s")
    table.add_row("Cancelled", "✓" if metrics_data.cancelled else "✗")
    table.add_row("Completion Tokens", str(metrics_data.completion_tokens))
    table.add_row("Input Tokens", str(metrics_data.prompt_tokens))
    table.add_row("Cached Tokens", str(metrics_data.prompt_cached_tokens))
    table.add_row("Total Tokens", str(metrics_data.total_tokens))
    table.add_row("Tokens/Second", f"{metrics_data.tokens_per_second:.2f}")

    display_metrics_table(table)


def handle_tts_metrics(metrics_data: metrics.TTSMetrics):
    """Handle TTS metrics display."""
    table = create_table("[bold blue]TTS Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Label", str(metrics_data.label))
    table.add_row("Request ID", str(metrics_data.request_id))
    table.add_row("Timestamp", timestamp)
    table.add_row("Duration", f"[white]{metrics_data.duration / 1000:.4f}[/white]s")
    table.add_row("Audio Duration", f"[white]{metrics_data.audio_duration / 1000:.4f}[/white]s")
    table.add_row("Time to First Byte", f"[white]{metrics_data.ttfb / 1000:.4f}[/white]s")
    table.add_row("Characters Count", str(metrics_data.characters_count))
    table.add_row("Cancelled", "✓" if metrics_data.cancelled else "✗")
    table.add_row("Streamed", "✓" if metrics_data.streamed else "✗")

    display_metrics_table(table)


def handle_vad_metrics(metrics_data: metrics.VADMetrics):
    """Handle VAD metrics display."""
    table = create_table("[bold blue]VAD Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Label", str(metrics_data.label))
    table.add_row("Timestamp", timestamp)
    table.add_row("Idle Time", f"[white]{metrics_data.idle_time / 1000:.4f}[/white]s")
    table.add_row("Inference Count", str(metrics_data.inference_count))
    table.add_row("Inference Duration Total", f"[white]{metrics_data.inference_duration_total / 1000:.4f}[/white]s")

    display_metrics_table(table)


def handle_eou_metrics(metrics_data: metrics.EOUMetrics):
    """Handle EOU metrics display."""
    table = create_table("[bold blue]EOU Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Timestamp", timestamp)
    table.add_row("Speech Id", metrics_data.speech_id or "N/A")
    table.add_row("End of Utterance Delay", f"[white]{metrics_data.end_of_utterance_delay / 1000:.4f}[/white]s")
    table.add_row("Transcription Delay", f"[white]{metrics_data.transcription_delay / 1000:.4f}[/white]s")
    table.add_row("On User Turn Completed Delay", f"[white]{metrics_data.on_user_turn_completed_delay / 1000:.4f}[/white]s")

    display_metrics_table(table)

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
    session = AgentSession(preemptive_generation=True, 
                           userdata=ud,
                           user_away_timeout=10)

    agent = AgentFactory.create_agent(agent_config)

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
            # "stt_metrics": handle_stt_metrics,
            # "llm_metrics": handle_llm_metrics,
            "tts_metrics": handle_tts_metrics,
            "vad_metrics": handle_vad_metrics,
            "eou_metrics": handle_eou_metrics,
        }
        
        handler = metrics_handlers.get(ev.metrics.type)
        if handler:
            handler(ev.metrics)
    
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
