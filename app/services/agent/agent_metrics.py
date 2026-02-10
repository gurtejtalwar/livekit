from rich.console import Console
from rich.table import Table
from rich import box

from livekit.agents import metrics
from dataclasses import dataclass
from typing import Optional

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


def print_stt_metrics(metrics_data: metrics.STTMetrics):
    """Handle STT metrics display."""
    table = create_table("[bold blue]STT Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Label", str(metrics_data.label))
    table.add_row("Request ID", str(metrics_data.request_id))
    table.add_row("Timestamp", timestamp)
    table.add_row("Audio Duration", f"[white]{metrics_data.audio_duration}[/white]s")
    table.add_row("Streamed", "✓" if metrics_data.streamed else "✗")

    display_metrics_table(table)


def print_llm_metrics(metrics_data: metrics.LLMMetrics):
    """Handle LLM metrics display."""
    table = create_table("[bold blue]LLM Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Label", str(metrics_data.label))
    table.add_row("Request ID", str(metrics_data.request_id))
    table.add_row("Timestamp", timestamp)
    table.add_row("Duration", f"[white]{metrics_data.duration}[/white]s")
    table.add_row("Time to First Token", f"[white]{metrics_data.ttft}[/white]s")
    table.add_row("Cancelled", "✓" if metrics_data.cancelled else "✗")
    table.add_row("Completion Tokens", str(metrics_data.completion_tokens))
    table.add_row("Input Tokens", str(metrics_data.prompt_tokens))
    table.add_row("Cached Tokens", str(metrics_data.prompt_cached_tokens))
    table.add_row("Total Tokens", str(metrics_data.total_tokens))
    table.add_row("Tokens/Second", f"{metrics_data.tokens_per_second}")

    display_metrics_table(table)


def print_tts_metrics(metrics_data: metrics.TTSMetrics):
    """Handle TTS metrics display."""
    table = create_table("[bold blue]TTS Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Label", str(metrics_data.label))
    table.add_row("Request ID", str(metrics_data.request_id))
    table.add_row("Timestamp", timestamp)
    table.add_row("Duration", f"[white]{metrics_data.duration}[/white]s")
    table.add_row("Audio Duration", f"[white]{metrics_data.audio_duration}[/white]s")
    table.add_row("Time to First Byte", f"[white]{metrics_data.ttfb}[/white]s")
    table.add_row("Characters Count", str(metrics_data.characters_count))
    table.add_row("Cancelled", "✓" if metrics_data.cancelled else "✗")
    table.add_row("Streamed", "✓" if metrics_data.streamed else "✗")

    display_metrics_table(table)


def print_vad_metrics(metrics_data: metrics.VADMetrics):
    """Handle VAD metrics display."""
    table = create_table("[bold blue]VAD Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Label", str(metrics_data.label))
    table.add_row("Timestamp", timestamp)
    table.add_row("Idle Time", f"[white]{metrics_data.idle_time}[/white]s")
    table.add_row("Inference Count", str(metrics_data.inference_count))
    table.add_row("Inference Duration Total", f"[white]{metrics_data.inference_duration_total}[/white]s")

    display_metrics_table(table)


def print_eou_metrics(metrics_data: metrics.EOUMetrics):
    """Handle EOU metrics display."""
    table = create_table("[bold blue]EOU Metrics Report[/bold blue]")
    table.add_column("Metric", style="bold green")
    table.add_column("Value", style="yellow")

    timestamp = format_timestamp(metrics_data.timestamp)

    table.add_row("Type", str(metrics_data.type))
    table.add_row("Timestamp", timestamp)
    table.add_row("Speech Id", metrics_data.speech_id or "N/A")
    table.add_row("End of Utterance Delay", f"[white]{metrics_data.end_of_utterance_delay}[/white]s")
    table.add_row("Transcription Delay", f"[white]{metrics_data.transcription_delay}[/white]s")
    table.add_row("On User Turn Completed Delay", f"[white]{metrics_data.on_user_turn_completed_delay}[/white]s")

    display_metrics_table(table)


@dataclass
class AvgAccumulator:
    total: float = 0.0
    count: int = 0

    def add(self, value: Optional[float]):
        if value is None:
            return
        self.total += value
        self.count += 1

    def avg(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.total / self.count

class CallMetricsAggregator:
    def __init__(self):
        # STT
        self.stt_audio_duration = AvgAccumulator()

        # LLM
        self.llm_duration = AvgAccumulator()
        self.llm_ttft = AvgAccumulator()
        self.llm_tokens_per_second = AvgAccumulator()
        self.llm_total_tokens = AvgAccumulator()

        # TTS
        self.tts_duration = AvgAccumulator()
        self.tts_audio_duration = AvgAccumulator()
        self.tts_ttfb = AvgAccumulator()

        # VAD
        self.vad_idle_time = AvgAccumulator()
        self.vad_inference_duration_total = AvgAccumulator()

        # EOU
        self.eou_delay = AvgAccumulator()
        self.eou_transcription_delay = AvgAccumulator()
        self.eou_turn_completed_delay = AvgAccumulator()

def handle_stt_metrics(metrics_data: metrics.STTMetrics, agg: CallMetricsAggregator):
    agg.stt_audio_duration.add(metrics_data.audio_duration)

def handle_llm_metrics(metrics_data: metrics.LLMMetrics, agg: CallMetricsAggregator):
    agg.llm_duration.add(metrics_data.duration)
    agg.llm_ttft.add(metrics_data.ttft)
    agg.llm_tokens_per_second.add(metrics_data.tokens_per_second)
    agg.llm_total_tokens.add(metrics_data.total_tokens)

def handle_tts_metrics(metrics_data: metrics.TTSMetrics, agg: CallMetricsAggregator):
    agg.tts_duration.add(metrics_data.duration)
    agg.tts_audio_duration.add(metrics_data.audio_duration)
    agg.tts_ttfb.add(metrics_data.ttfb)

def handle_vad_metrics(metrics_data: metrics.VADMetrics, agg: CallMetricsAggregator):
    agg.vad_idle_time.add(metrics_data.idle_time)
    agg.vad_inference_duration_total.add(
        metrics_data.inference_duration_total
    )
