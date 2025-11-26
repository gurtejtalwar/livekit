import asyncio
import time
from contextlib import contextmanager
import os
import logging
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    RoomInputOptions,
    RoomOutputOptions,
    metrics, 
    MetricsCollectedEvent
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, cartesia, silero, noise_cancellation, elevenlabs, assemblyai
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding

from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.turn_detector.english import EnglishModel

from app.services.agent.cache import semantic_context_cache

logger = logging.getLogger("inbound-agent")
for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from collections import OrderedDict
from typing import Optional
from livekit import agents, rtc


class LRUCache:
    """Simple thread-safe LRU for async apps (no awaits needed)."""
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        # mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: str):
        self.cache[key] = value
        self.cache.move_to_end(key)

        # evict least recently used
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

# create global cache instance
mongo_lru_cache = LRUCache(max_size=512)

@function_tool
async def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%I:%M %p')}" 

###### Pinecone Vector DB Loader ######
from pathlib import Path
from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from functools import lru_cache

load_dotenv(override=True)

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
with Timer("Pinecone + Index Initialization"):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pc_index = pc.Index("ai-tutor")
    # pc_index = pc.Index("prod-itsbot")
    vector_store = PineconeVectorStore(
        pinecone_index=pc_index,
        # namespace="f280790d-1517-456a-8954-2c296b38f8e1"
    )
    embed_model = OpenAIEmbedding(
        api_key=os.environ["OPENAI_API_KEY"],
        model="text-embedding-3-small"
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    retriever = index.as_retriever(similarity_top_k=1, use_async=True)

with Timer("Mongo Initialization"):
    mongo_client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = mongo_client["itsbot-db"]
    parents_collection = db["parentdocs"]

# ---------------------- CACHED EMBEDDINGS ----------------------
@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    """Cache embeddings for repeated/similar queries."""
    return embed_model.get_text_embedding(text)

# Run cache check and retriever fetch IN PARALLEL
async def check_cache(question):
    with Timer("Cache Fetch"):
        return semantic_context_cache.get(question)

async def fetch_from_retriever(question):
    with Timer("Retriever Fetch"):
        return await retriever.aretrieve(question)
# ---------------------- MAIN PIPELINE ----------------------
@llm.function_tool
async def ask_knowledge_base(question: str):
    """Ultra-fast retrieval with streaming context"""
    
    # Run cache and retriever in parallel
    cache_task = asyncio.create_task(check_cache(question))
    retriever_task = asyncio.create_task(fetch_from_retriever(question))
    
    # Wait for whichever completes first
    done, pending = await asyncio.wait(
        {cache_task, retriever_task},
        return_when=asyncio.FIRST_COMPLETED,
        timeout=0.5  # Max 500ms wait
    )
    
    # Check cache first
    if cache_task in done:
        cache_result = await cache_task
        if cache_result:
            matched_question, cached_context, similarity = cache_result
            print(f"✓ Cache Hit! Similarity: {similarity:.3f}")
            
            # Cancel pending retriever
            for task in pending:
                task.cancel()
            
            return cached_context
    
    # Use retriever results
    if retriever_task in done:
        results = await retriever_task
    else:
        results = await retriever_task  # Wait if not done yet
    
    # Build context with LIMIT
    context_parts = [node.text for node in results[:3]]  # Limit to top 3 for speed
    context = "\n".join(context_parts)
    
    # Async cache update (fire and forget)
    asyncio.create_task(semantic_context_cache.set_async(question, context))
    
    return context
# ---------------------- PRE-WARM CONNECTIONS ----------------------
async def prewarm():
    """Pre-initialize HTTPS sessions and caches to avoid cold-start delay."""
    print("Prewarming connections...")
    try:
        _ = await retriever.aretrieve("ping")  # Warm Pinecone/OpenAI
    except Exception as e:
        print("Prewarm failed (harmless):", e)
    _ = get_cached_embedding("ping")  # Warm cache
    print("Prewarm complete.")

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
            stt=deepgram.STT(),
            # stt=assemblyai.STT(model="universal-streaming-multilingual"),
            llm=openai.LLM(model="gpt-4o-mini", tool_choice="auto", max_completion_tokens=50),
            # tts=elevenlabs.TTS(),#model="eleven_v3",voice_id="EkK5I93UQWFDigLMpZcX"),
            tts=cartesia.TTS
            (
                model="sonic-turbo",
                voice="228fca29-3a0a-435c-8728-5cb483251068",
                emotion="Happy",
                speed="slow",
                volume=100
            ),
            vad=silero.VAD.load(min_speech_duration=0.2,
                                min_silence_duration=0.3),
            # turn_detection=EnglishModel(),
            # preemptive_generation=True,
            tools=[get_current_time, ask_knowledge_base],
            min_endpointing_delay=0.1,  # Minimum wait after silence
            max_endpointing_delay=1,  # Maximum wait before forcing turn end
            allow_interruptions=True,
            use_tts_aligned_transcript=False
        )
    async def llm_node(
        self, chat_ctx, tools, model_settings=None
    ):
        """Optimized LLM node with minimal overhead"""
        with Timer("LLM Node:"):
            async for chunk in super().llm_node(chat_ctx, tools, model_settings):
                yield chunk 


    async def tts_node(self, text, model_settings):
        return super().tts_node(text, model_settings)

async def inbound_entrypoint(ctx: JobContext):
    # Prewarm in parallel with connection
    prewarm_task = asyncio.create_task(prewarm())
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await prewarm_task  # Ensure prewarm completes
    
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

