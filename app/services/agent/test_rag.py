from dotenv import load_dotenv
import os
import asyncio
from PIL import Image

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RoomOutputOptions, function_tool
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
    # hedra,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Pinecone Assistant SDK
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone, ServerlessSpec
# from pinecone_plugins.assistant.models.chat import Message

load_dotenv()

# Initialize Pinecone Assistant
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("ai-tutor")

# Wrap in a LlamaIndex vector store
vector_store = PineconeVectorStore(pinecone_index=index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

@function_tool
async def ask_knowledge_base(question: str) -> str:
    """Query the Pinecone index (knowledge base) for relevant information."""
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response.response

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an iPear customer service AI assistant. "
                "Use the 'ask_knowledge_base' tool for any iPear-related questions. "
                "Keep responses conversational and helpful. "
                "Format numbers and units naturally for speech - say 'five hundred and twelve gigabytes' instead of 'five one two GB', "
                "'one hundred and twenty-eight gigabytes' instead of 'one two eight GB', and 'one terabyte' instead of 'one TB'."
            ),
            tools=[ask_knowledge_base],
        )

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        # turn_detection=MultilingualModel(),
    )

    # Avatar setup
    # avatar_image = Image.open("/Users/jimmybradford/Downloads/rep.png")
    # avatar = hedra.AvatarSession(avatar_image=avatar_image)
    # await avatar.start(session, room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        # room_input_options=RoomInputOptions(
        #     noise_cancellation=noise_cancellation.BVC(),
        #     close_on_disconnect=False,
        # ),
    )

    await session.generate_reply(
        instructions="Greet the user as an iPear customer service representative. Speak english."
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="inbound-agent"))