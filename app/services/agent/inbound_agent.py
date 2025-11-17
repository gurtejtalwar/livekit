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

from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from app.utils import core_utils
from app.utils import agent_utils

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

###### Pinecone Vector DB Loader ######
from pathlib import Path
from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv(override=True)


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
            tools=[agent_utils.get_current_time, agent_utils.ask_knowledge_base],
        )


async def entrypoint(ctx: JobContext):
    await agent_utils.prewarm()
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
