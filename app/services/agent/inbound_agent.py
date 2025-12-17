import asyncio
import time
import os
import logging
import json
import tempfile
import pytz
from datetime import datetime
from typing import Tuple, AsyncIterable
from contextlib import contextmanager
from dataclasses import asdict

import faiss
import pickle
from groq import AsyncGroq
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    llm,
    RoomInputOptions,
    RoomOutputOptions,
    metrics, 
    MetricsCollectedEvent,
    RunContext,
    ChatContext, 
    ChatMessage

)
from livekit.agents.llm import function_tool
from livekit.plugins import (
    deepgram, 
    openai, 
    cartesia, 
    silero, 
    noise_cancellation, 
    elevenlabs, 
    assemblyai, 
    groq,
    langchain
)

from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.turn_detector.english import EnglishModel

from app.services.agent.cache import semantic_context_cache

logger = logging.getLogger("inbound-agent")
for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from collections import OrderedDict
from typing import Optional

@function_tool
async def get_current_time(input: str) -> str:
    """Get the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%I:%M %p')}" 

###### Pinecone Vector DB Loader ######
from pathlib import Path
from dotenv import load_dotenv
import os
from functools import lru_cache

load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ---------------------- TIMER UTILITY ----------------------
class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dur = time.perf_counter() - self.start
        print(f"\nTIMER: {self.name} took {dur:.4f} seconds")

# ---------------------- GLOBAL SETUP ----------------------

with Timer("Load Index, Tokenizer and Embedding Model"):
    index = faiss.read_index("dev_scripts/faiss.index")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        export=True
    )

def embed(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

with open("dev_scripts/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
    
def get_text_from_indices(indices):
    """Return the text chunks for each FAISS result index."""
    result = []
    for idx in indices:
        if 0 <= idx < len(chunks):
            result.append(chunks[idx])
        else:
            result.append("[INVALID INDEX]")
    return result

# ---------------------- MAIN PIPELINE ----------------------
@llm.function_tool
async def ask_knowledge_base(question: str):
    """Ultra-fast retrieval with streaming context"""
    with Timer("KB Tool Total:"):
        with Timer("Embed Query"):
            q_emb = embed(question)
        with Timer("FAISS Search"):
            dist, idx = index.search(q_emb, k=3)    # top 3 matches
        indices = idx[0]                        # array of indices
        matched_text = get_text_from_indices(indices)
        context = "\n".join(matched_text)
        return context

###### Inbound RAG Agent ######
class InboundAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a Eminence Technology customer service AI assistant. "
                "For ANY Eminence Technology-related or factual question, you MUST use the 'ask_knowledge_base' tool FIRST. "
                "Do not rely on your internal memory. "
                "After receiving the tool's output, use it to construct a conversational, human-like answer. "
                "If the tool returns no relevant data, politely say you don't have enough information. "
                "Keep responses concise and optimized for spoken delivery. PLEASE MAKE SURE THAT THE RESPONSES ARE SHORT SO THAT IT MIMICKS A PHONE CONVERSATION BETWEEN HUMANS. "
                "Do not respond with asterick, bullet points,etc  please respond how you would in a normal conversation with a human. "
                "PLEASE keep your tone friendly and enthusiastic. Always Respond politely to the customer. You are allowed to do small talks with the customer BUT DO NOT STRAY AWAY FROM THE BUSINESS AND OBJECTIVE OF THE CONVERSATION"
                "Format numbers naturally (e.g., 'five hundred and twelve gigabytes')." \
                # "Please return the text with formatted emotion type before sentence to indicate the TTS model on which emotion to synthesie the speed with, for eg, [enthusiastically] Hello, how are you."
            ),
            stt=assemblyai.STT(),
            # stt=assemblyai.STT(model="universal-streaming-multilingual"),
            # llm=openai.LLM(model="gpt-4o-mini", tool_choice="auto", max_completion_tokens=50),
            llm=groq.LLM(model="qwen/qwen3-32b", tool_choice="auto", max_completion_tokens=os.environ.get("MAX_TOKENS", 100)),
            # tts=elevenlabs.TTS(),#model="eleven_v3",voice_id="EkK5I93UQWFDigLMpZcX"),
            tts=cartesia.TTS
            (
                model="sonic-3",
                voice="6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
                emotion="Happy",
                speed=1.0,
                volume=2
            ),
            tools=[get_current_time, ask_knowledge_base],
            min_endpointing_delay=0.1,  # Minimum wait after silence
            max_endpointing_delay=0.5,  # Maximum wait before forcing turn end
            allow_interruptions=True,
            use_tts_aligned_transcript=False
        )
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """
        Custom LLM node using Groq streaming (Qwen 2.5 32B)
        """

        client = AsyncGroq(api_key=GROQ_API_KEY)

        # --- 1. Convert ChatContext → OpenAI-style messages ---
        messages = [
            {
                "role": msg.role,
                "content": content_to_string(msg.content),
            }
            for msg in chat_ctx.items
        ]

        # --- 2. Streaming completion ---
        stream = await client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
            temperature=0.3,
            max_completion_tokens=100,
            stream=True,
            reasoning_effort="none",
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue
            if delta.content.startswith("<think>"):
                continue
            yield llm.ChatChunk(
                id="assistant-stream",
                delta=llm.ChoiceDelta(role=delta.role,
                                      content=delta.content,
                                      tool_calls=[delta.tool_calls] if delta.tool_calls else []),
            )
async def inbound_entrypoint(ctx: JobContext):
    # Prewarm in parallel with connection
    # prewarm_task = asyncio.create_task(prewarm())
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    # await prewarm_task  # Ensure prewarm completes
    
    agent = InboundAgent()
    session = AgentSession()

    usage_collector = metrics.UsageCollector()
    metrics_history = []
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        metrics_history.append(ev.metrics.model_dump())

    @session.on("session_end")
    def _on_session_end():
        report = ctx.make_session_report()
        report_dict = report.to_dict()
        ist = pytz.timezone("Asia/Kolkata")
        current_date = datetime.now(ist).strftime("%Y%m%d_%H%M%S")

        tmp_dir = tempfile.gettempdir()
        
        # Write session report
        session_filename = os.path.join(
            tmp_dir,
            f"session_report_{ctx.room.name}_{current_date}.json"
        )
        with open(session_filename, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        print(f"Session report for {ctx.room.name} saved to {session_filename}")
        
        # Write metrics report
        metrics_filename = write_metrics_to_json(usage_collector, ctx.room.name, current_date, metrics_history)
        
        # Log usage summary
        summary = usage_collector.get_summary()
        logger.info(
            f"Session metrics - Tokens: {summary.llm_prompt_tokens + summary.llm_completion_tokens}, "
            f"TTS chars: {summary.tts_characters_count}, "
            f"STT duration: {summary.stt_audio_duration:.2f}s"
        )

    ctx.add_shutdown_callback(_on_session_end)
    await session.say("Thanks for calling Eminence Technology customer support. My name is Lala, let me know how I can assist you")

def content_to_string(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    return ""

# ---------------------- METRICS UTILITIES ----------------------
def write_metrics_to_json(usage_collector: metrics.UsageCollector, room_name: str, timestamp: str, metrics_history: list) -> str:
    """
    Write aggregated metrics and raw metrics history to a JSON file.
    
    Args:
        usage_collector: UsageCollector instance with aggregated metrics
        room_name: Name of the room for filename
        timestamp: Formatted timestamp for filename
        metrics_history: List of raw metric events
    
    Returns:
        Path to the written metrics file
    """
    tmp_dir = tempfile.gettempdir()
    
    # Get aggregated usage summary
    usage_summary = usage_collector.get_summary()
    
    # Convert UsageSummary dataclass to dict
    usage_dict = asdict(usage_summary)
    
    # Extract all metrics by type for comprehensive analysis
    llm_metrics_list = []
    tts_metrics_list = []
    stt_metrics_list = []
    eou_metrics_list = []
    vad_metrics_list = []
    realtime_model_metrics_list = []
    
    for metric in metrics_history:
        metric_type = metric.get("type")
        
        if metric_type == "llm_metrics":
            llm_metrics_list.append(metric)
        elif metric_type == "tts_metrics":
            tts_metrics_list.append(metric)
        elif metric_type == "stt_metrics":
            stt_metrics_list.append(metric)
        elif metric_type == "eou_metrics":
            eou_metrics_list.append(metric)
        elif metric_type == "vad_metrics":
            vad_metrics_list.append(metric)
        elif metric_type == "realtime_model_metrics":
            realtime_model_metrics_list.append(metric)
    
    # Calculate statistics helper
    def calc_stats(values, key=None):
        """Calculate min, max, avg for a list of values or dict key"""
        if key is not None:
            values = [v.get(key) for v in values if v.get(key) is not None]
        values = [v for v in values if v is not None and isinstance(v, (int, float)) and v >= 0]
        
        if not values:
            return None
        return {
            "count": len(values),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "avg": round(sum(values) / len(values), 4),
        }
    
    # Build comprehensive metrics report
    metrics_report = {
        "timestamp": timestamp,
        "room_name": room_name,
        "summary": {
            "total_metrics_events": len(metrics_history),
            "llm_calls": len(llm_metrics_list),
            "tts_calls": len(tts_metrics_list),
            "stt_calls": len(stt_metrics_list),
            "eou_detections": len(eou_metrics_list),
            "vad_events": len(vad_metrics_list),
            "realtime_model_calls": len(realtime_model_metrics_list),
        },
        "usage_summary": {
            "tokens": {
                "llm_prompt_tokens": usage_dict["llm_prompt_tokens"],
                "llm_prompt_cached_tokens": usage_dict["llm_prompt_cached_tokens"],
                "llm_completion_tokens": usage_dict["llm_completion_tokens"],
                "llm_total_tokens": usage_dict["llm_prompt_tokens"] + usage_dict["llm_completion_tokens"],
            },
            "llm_realtime_model": {
                "input_text_tokens": usage_dict["llm_input_text_tokens"],
                "input_audio_tokens": usage_dict["llm_input_audio_tokens"],
                "input_image_tokens": usage_dict["llm_input_image_tokens"],
                "input_cached_text_tokens": usage_dict["llm_input_cached_text_tokens"],
                "input_cached_audio_tokens": usage_dict["llm_input_cached_audio_tokens"],
                "input_cached_image_tokens": usage_dict["llm_input_cached_image_tokens"],
                "output_text_tokens": usage_dict["llm_output_text_tokens"],
                "output_audio_tokens": usage_dict["llm_output_audio_tokens"],
                "output_image_tokens": usage_dict["llm_output_image_tokens"],
            },
            "tts": {
                "characters_count": usage_dict["tts_characters_count"],
                "audio_duration_seconds": usage_dict["tts_audio_duration"],
            },
            "stt": {
                "audio_duration_seconds": usage_dict["stt_audio_duration"],
            },
        },
        "performance_metrics": {
            "llm": {
                "ttft_seconds": calc_stats(llm_metrics_list, "ttft"),
                "duration_seconds": calc_stats(llm_metrics_list, "duration"),
                "tokens_per_second": calc_stats(llm_metrics_list, "tokens_per_second"),
                "total_requests": len(llm_metrics_list),
                "total_cancelled": sum(1 for m in llm_metrics_list if m.get("cancelled")),
            },
            "tts": {
                "ttfb_seconds": calc_stats(tts_metrics_list, "ttfb"),
                "duration_seconds": calc_stats(tts_metrics_list, "duration"),
                "audio_duration_seconds": calc_stats(tts_metrics_list, "audio_duration"),
                "total_requests": len(tts_metrics_list),
                "total_cancelled": sum(1 for m in tts_metrics_list if m.get("cancelled")),
                "streamed_count": sum(1 for m in tts_metrics_list if m.get("streamed")),
            },
            "stt": {
                "audio_duration_seconds": calc_stats(stt_metrics_list, "audio_duration"),
                "duration_seconds": calc_stats(stt_metrics_list, "duration"),
                "total_requests": len(stt_metrics_list),
                "streamed_count": sum(1 for m in stt_metrics_list if m.get("streamed")),
            },
            "eou": {
                "end_of_utterance_delay_seconds": calc_stats(eou_metrics_list, "end_of_utterance_delay"),
                "transcription_delay_seconds": calc_stats(eou_metrics_list, "transcription_delay"),
                "on_user_turn_completed_delay_seconds": calc_stats(eou_metrics_list, "on_user_turn_completed_delay"),
                "total_detections": len(eou_metrics_list),
            },
            "vad": {
                "idle_time_seconds": calc_stats(vad_metrics_list, "idle_time"),
                "inference_duration_total_seconds": calc_stats(vad_metrics_list, "inference_duration_total"),
                "inference_count": calc_stats(vad_metrics_list, "inference_count"),
                "total_events": len(vad_metrics_list),
            },
            "realtime_model": {
                "ttft_seconds": calc_stats(realtime_model_metrics_list, "ttft"),
                "duration_seconds": calc_stats(realtime_model_metrics_list, "duration"),
                "tokens_per_second": calc_stats(realtime_model_metrics_list, "tokens_per_second"),
                "total_requests": len(realtime_model_metrics_list),
                "total_cancelled": sum(1 for m in realtime_model_metrics_list if m.get("cancelled")),
            },
            "conversation_latency": {
                "estimated_total_latency_seconds": calc_stats(
                    [
                        {
                            "latency": (eou.get("end_of_utterance_delay", 0) + 
                                       next((llm.get("ttft", 0) for llm in llm_metrics_list), 0) +
                                       next((tts.get("ttfb", 0) for tts in tts_metrics_list), 0))
                        }
                        for eou in eou_metrics_list[:len(tts_metrics_list)]
                    ],
                    "latency"
                ),
                "note": "Formula: eou.end_of_utterance_delay + llm.ttft + tts.ttfb",
            },
        },
        "raw_metrics_count": len(metrics_history),
        "raw_metrics": metrics_history,
    }
    
    # Write to JSON file
    filename = os.path.join(
        tmp_dir,
        f"metrics_{room_name}_{timestamp}.json"
    )
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Metrics for {room_name} saved to {filename}")
    return filename