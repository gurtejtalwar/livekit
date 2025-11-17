import asyncio
import time
from contextlib import contextmanager
import os
import logging
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
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, cartesia, silero, noise_cancellation
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding

from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("inbound-agent")
for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# @contextmanager
# def timer(name="Operation"):
#     start = time.perf_counter()
#     yield
#     end = time.perf_counter()
#     print(f"TIMER: {name} took {end - start:.4f} seconds")

# Function tools to enhance your agent's capabilities
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
        print(f"TIMER: {self.name} took {dur:.4f} seconds")

# ---------------------- GLOBAL SETUP ----------------------
with Timer("Pinecone + Index Initialization"):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pc_index = pc.Index("stage-itsbot")
    vector_store = PineconeVectorStore(
        pinecone_index=pc_index,
        namespace="4802b88d-d493-4648-a04d-8dc5585cf271"
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

        # # (1) Get (possibly cached) embedding
        # with Timer("Embedding Creation"):
        #     query_vec = get_cached_embedding(question)

        # (2) Perform Pinecone query (async)
        with Timer("Retriever Query"):
            results = await retriever.aretrieve(question)

        results = [node for node in results if node.score >= 0.6]

        # (3) Extract doc IDs
        ids = [
            r.node.metadata.get("doc_id")
            for r in results
            if r.node.metadata.get("doc_id")
        ]

        if not ids:
            return {"retriever_results": results, "mongo_docs": []}

        # (4) Fetch Mongo docs asynchronously
        async def fetch_mongo(ids):
            cursor = parents_collection.find(
                {"parentDocId": {"$in": ids}},
                {"_id": 0, "data": 1}
            )
            docs = await cursor.to_list(length=None)

            data_values = [doc["data"] for doc in docs if "data" in doc]

            return "\n".join(data_values)

        with Timer("Mongo Fetch"):
            mongo_docs = await fetch_mongo(ids)

        print(f"Mongo Docs: \n{mongo_docs}")
        return mongo_docs

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
                "You are an ITSCNC customer service AI assistant. "
                "For ANY ITSCNC-related or factual question, you MUST use the 'ask_knowledge_base' tool FIRST. "
                "Do not rely on your internal memory. "
                "After receiving the tool's output, use it to construct a conversational, human-like answer. "
                "If the tool returns no relevant data, politely say you don't have enough information. "
                "Keep responses concise and optimized for spoken delivery. "
                "Format numbers naturally (e.g., 'five hundred and twelve gigabytes')."
            ),
            tools=[get_current_time, ask_knowledge_base],
        )


async def entrypoint(ctx: JobContext):
    await prewarm()
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = InboundAgent()
    session = AgentSession(
        stt=deepgram.STT(language="en"),
        llm=openai.LLM(model="gpt-4o-mini", tool_choice="required"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
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
    await session.say("Thanks for calling ItsCNC customer support. My name is Lala, let me know how I can assist you")

if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the agent with the name that matches your dispatch rule
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="inbound-agent",
        initialize_process_timeout=60  # This must match your dispatch rule
    ))
