import asyncio
import time
import os
import logging
from typing import Tuple, AsyncIterable
from contextlib import contextmanager

import faiss
import pickle
from groq import AsyncGroq


from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
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

from app.utils.timer import Timer

logger = logging.getLogger("inbound-agent")
for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from collections import OrderedDict
from typing import Optional


from pathlib import Path
from dotenv import load_dotenv
import os
from functools import lru_cache

load_dotenv(override=True)

agent_server = AgentServer()

# ---------------------- MAIN PIPELINE ----------------------

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
            # stt=assemblyai.STT(),
            # stt=assemblyai.STT(model="universal-streaming-multilingual"),
            # llm=openai.LLM(model="gpt-4o-mini", tool_choice="auto", max_completion_tokens=50),
            llm=groq.LLM(model="qwen/qwen3-32b", tool_choice="auto", max_completion_tokens=100),
            # tts=elevenlabs.TTS(),#model="eleven_v3",voice_id="EkK5I93UQWFDigLMpZcX"),
            tts=cartesia.TTS
            (
                model="sonic-turbo",
                voice="6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
                emotion="Happy",
                speed=1.0,
                volume=2
            ),
            turn_detection=EnglishModel(),
            tools=[get_current_time, ask_knowledge_base],
            min_endpointing_delay=0.05,  # Minimum wait after silence
            max_endpointing_delay=0.3,  # Maximum wait before forcing turn end
            allow_interruptions=True,
            use_tts_aligned_transcript=False
        )

from app.services.agent.factory import AgentFactory, load_agent_config

@agent_server.rtc_session(agent_name="inbound-agent")
async def inbound_entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Example: resolve from headers / room metadata / API
    # customer_id = ctx.job.metadata.get("customer_id")
    # agent_id = ctx.job.metadata.get("agent_id")

    agent_config = await load_agent_config("some-customer-id","some-agent-id")
    agent = AgentFactory.create_agent(agent_config)

    session = AgentSession(preemptive_generation=True)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )

    await session.say(agent_config.greeting)

def content_to_string(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    return ""