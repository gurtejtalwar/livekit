import os
from functools import lru_cache

from pinecone import Pinecone
from motor.motor_asyncio import AsyncIOMotorClient

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore

from livekit.agents import llm

from app.utils import core_utils

# ---------------------- GLOBAL SETUP ----------------------
with core_utils.Timer("Pinecone + Index Initialization"):
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

with core_utils.Timer("Mongo Initialization"):
    mongo_client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = mongo_client["itsbot-db"]
    parents_collection = db["parentdocs"]

# ---------------------- CACHED EMBEDDINGS ----------------------
@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    """Cache embeddings for repeated/similar queries."""
    return embed_model.get_text_embedding(text)

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

# ---------------------- MAIN PIPELINE ----------------------
@llm.function_tool
async def ask_knowledge_base(question: str):
    """Retrieve relevant documents from Pinecone + fetch full docs from Mongo."""
    with core_utils.Timer("Full Query"):

        # (1) Get (possibly cached) embedding
        with core_utils.Timer("Embedding Creation"):
            query_vec = get_cached_embedding(question)

        # (2) Perform Pinecone query (async)
        with core_utils.Timer("Retriever Query"):
            results = await retriever.aretrieve(question)

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

        with core_utils.Timer("Mongo Fetch"):
            mongo_docs = await fetch_mongo(ids)

        print(f"Mongo Docs: \n{mongo_docs}")
        return mongo_docs

@llm.function_tool
async def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%I:%M %p')}" 
