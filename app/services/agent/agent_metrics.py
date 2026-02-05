from rich.console import Console
from rich.table import Table
from rich import box

from livekit.agents import metrics

console = Console()

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
