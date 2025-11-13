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

@contextmanager
def timer(name="Operation"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"TIMER: {name} took {end - start:.4f} seconds")

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

load_dotenv(override=True)

###### PINECONE SETUP ######
# Initialize Pinecone client
# index = pc.Index("ai-tutor")

# Wrap in a LlamaIndex vector store
with timer("Pinecone Initialization"):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pc_index = pc.Index("stage-itsbot")
    vector_store = PineconeVectorStore(pinecone_index=pc_index, namespace="4802b88d-d493-4648-a04d-8dc5585cf271")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(api_key=os.environ["OPENAI_API_KEY"],model="text-embedding-3-small")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

from pymongo import MongoClient
with timer("Mongo Intialization"):
    mongo_client = MongoClient(os.environ["MONGO_URI"])  # e.g., "mongodb://localhost:27017"
    db = mongo_client["itsbot-db"]  # replace with your DB name
    parents_collection = db["parentdocs"]

@llm.function_tool
async def ask_knowledge_base(question: str) -> str:
    from llama_index.llms.openai import OpenAI
# ---- Retriever Function ----
    """Query the Pinecone index (knowledge base) for relevant information."""
    from llama_index.llms.openai import OpenAI
    with timer("Retriever Initialization"):
        retriever  = index.as_query_engine(similarity_top_k=2, 
                                            use_async=True, 
                                            llm=OpenAI(
                                                api_key=os.environ["OPENAI_API_KEY"],
                                                model="gpt-4o-mini")
                                            )
    with timer("Fetching Retriever"):
        results = retriever.retrieve(question)

    # Filter nodes by similarity >= 0.7
    filtered_nodes = [node for node in results if node.score >= 0.6]

    if not filtered_nodes:
        return "⚠️ No results found with similarity >= 0.7"

    # Fetch documents from MongoDB based on doc_id
    fetched_docs = []
    for node_with_score in filtered_nodes:
        doc_id = node_with_score.node.metadata.get("doc_id")
        if doc_id:
            with timer("Fetching Mongo"):
                doc = parents_collection.find_one({"parentDocId": doc_id})
            if doc:
                fetched_docs.append(doc.get("data", ""))  # assuming the string is in 'content'

    if not fetched_docs:
        return "⚠️ No corresponding documents found in MongoDB"

    # Combine or return first match
    print(f"\nQuery results are:\n{fetched_docs}")

    return "\n\n".join(fetched_docs)

###### PINECONE SETUP ######

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
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = InboundAgent()
    session = AgentSession(
        stt=deepgram.STT(language="multi"),
        llm=openai.LLM(model="gpt-4o-mini", tool_choice="required"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=False,
        ),
        room_output_options=RoomOutputOptions(
            audio_enabled=True,  # Avatar handles audio
        ),
    )
    await session.generate_reply(
        instructions="Greet the user as an ITSCNC customer service representative. Speak english."
    )

if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the agent with the name that matches your dispatch rule
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="inbound-agent"  # This must match your dispatch rule
    ))
