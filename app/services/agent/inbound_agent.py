import asyncio
import time
import os
import logging
from typing import Tuple, AsyncIterable
from contextlib import contextmanager

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
from livekit.plugins import deepgram, openai, cartesia, silero, noise_cancellation, elevenlabs, assemblyai, groq

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
            # vad=silero.VAD.load(min_speech_duration=0.2,
            #                     min_silence_duration=0.3),
            # turn_detection=EnglishModel(),
            # preemptive_generation=True,
            tools=[get_current_time, ask_knowledge_base],
            min_endpointing_delay=0.1,  # Minimum wait after silence
            max_endpointing_delay=0.5,  # Maximum wait before forcing turn end
            allow_interruptions=True,
            use_tts_aligned_transcript=False
        )
    # async def llm_node(
    #     self,
    #     chat_ctx: llm.ChatContext,
    #     tools: list[llm.FunctionTool],
    #     model_settings: ModelSettings,
    # ) -> AsyncIterable[llm.ChatChunk]:
    #     """
    #     Custom LLM node using Groq streaming (Qwen 2.5 32B)
    #     """

    #     client = AsyncGroq(api_key=GROQ_API_KEY)

    #     # --- 1. Convert ChatContext → OpenAI-style messages ---
    #     messages = [
    #         {
    #             "role": msg.role,
    #             "content": content_to_string(msg.content),
    #         }
    #         for msg in chat_ctx.items
    #     ]

    #     # --- 2. Streaming completion ---
    #     stream = await client.chat.completions.create(
    #         model="qwen/qwen3-32b",
    #         messages=messages,
    #         temperature=0.3,
    #         max_completion_tokens=100,
    #         stream=True,
    #     )

    #     async for chunk in stream:
    #         if not chunk.choices:
    #             continue

    #         delta = chunk.choices[0].delta
    #         if not delta or not delta.content:
    #             continue

    #         yield llm.ChatChunk(
    #             id="assistant-stream",
    #             role="assistant",
    #             content=delta.content,
    #         )
async def inbound_entrypoint(ctx: JobContext):
    # Prewarm in parallel with connection
    # prewarm_task = asyncio.create_task(prewarm())
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    # await prewarm_task  # Ensure prewarm completes
    
    agent = InboundAgent()
    session = AgentSession()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)
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