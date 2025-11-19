import asyncio
import time
from contextlib import contextmanager
import os
import logging
from collections.abc import AsyncIterable

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    RoomInputOptions,
    RoomOutputOptions,
)
from livekit.agents.llm import function_tool, ChatContext, ChatMessage
from livekit.plugins import deepgram, openai, cartesia, silero, noise_cancellation
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

# ---------------------- MAIN PIPELINE ----------------------
@llm.function_tool
async def ask_knowledge_base(question: str):
    """Retrieve relevant documents from Pinecone + fetch full docs from Mongo."""
    with Timer("Full Query"):

        # Run cache check and retriever fetch IN PARALLEL
        async def check_cache():
            with Timer("Cache Fetch"):
                return semantic_context_cache.get(question)
        
        async def fetch_from_retriever():
            with Timer("Retriever Fetch"):
                return await retriever.aretrieve(question)
        
        # Start both operations simultaneously
        cache_task = asyncio.create_task(check_cache())
        retriever_task = asyncio.create_task(fetch_from_retriever())
        
        # Wait for cache check first (it's usually faster)
        cache_result = await cache_task
        
        
        if cache_result:
            matched_question, cached_context, similarity = cache_result
            print(f"✓ Cache Hit! Similarity: {similarity:.3f}")
            print(f"  Matched: '{matched_question[:60]}...'")
            
            # Cancel the retriever task since we don't need it
            retriever_task.cancel()
            # try:
            #     await retriever_task
            # except asyncio.CancelledError:
            #     pass  # Expected when we cancel
            
            return cached_context + "\n\n(Cached)"
        
        print("✗ Cache Miss - Using retriever results...")
        
        # Cache miss: wait for retriever (which is already running!)
        results = await retriever_task

        # Step 4: Build context
        context = "\nContext:\n" + "\n".join(node.text for node in results)

        # Step 5: Store in semantic cache
        asyncio.create_task(
                semantic_context_cache.set_async(question, context)
            )

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
                "Format numbers naturally (e.g., 'five hundred and twelve gigabytes')."
            ),
            tools=[get_current_time, ask_knowledge_base],
        )

        self._fast_llm = openai.LLM(model="gpt-4o-mini")
        self._fast_llm_prompt = llm.ChatMessage(
            role="system",
            content=[
                "Generate a short instant response to the user's message with 5 to 10 words.",
                "Do not answer the questions directly. Examples:, let me think about that, "
                "wait a moment, that's a good question, etc.",
            ],
        )


    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        # Create a short "silence filler" response to quickly acknowledge the user's input
        fast_llm_ctx = turn_ctx.copy(
            exclude_instructions=True, exclude_function_call=True
        ).truncate(max_items=3)
        fast_llm_ctx.items.insert(0, self._fast_llm_prompt)
        fast_llm_ctx.items.append(new_message)

        # # Intentionally not awaiting SpeechHandle to allow the main response generation to
        # # run concurrently
        # self.session.say(
        #     self._fast_llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable(),
        #     add_to_chat_ctx=False,
        # )

        # Alternatively, if you want the reply to be aware of this "silence filler" response,
        # you can await the fast llm done and add the message to the turn context. But note
        # that not all llm supports completing from an existing assistant message.

        fast_llm_fut = asyncio.Future[str]()

        async def _fast_llm_reply() -> AsyncIterable[str]:
            filler_response: str = ""
            async for chunk in self._fast_llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable():
                filler_response += chunk
                yield chunk
            fast_llm_fut.set_result(filler_response)

        self.session.say(_fast_llm_reply(), add_to_chat_ctx=False)

        filler_response = await fast_llm_fut
        logger.info(f"Fast response: {filler_response}")
        turn_ctx.add_message(role="assistant", content=filler_response, interrupted=False)

async def inbound_entrypoint(ctx: JobContext):
    await prewarm()
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = InboundAgent()
    session = AgentSession(
        stt=deepgram.STT(language="en"),
        llm=openai.LLM(model="gpt-4o-mini", tool_choice="auto"),
        tts=cartesia.TTS
        (
            # model="sonic-turbo",
            voice="228fca29-3a0a-435c-8728-5cb483251068",
            emotion="Happy",
            speed="slow"
        ),
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
        preemptive_generation=True,
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )
    time.sleep(0.2)
    await session.say("Thanks for calling Eminence Technology customer support. My name is Lala, let me know how I can assist you")

