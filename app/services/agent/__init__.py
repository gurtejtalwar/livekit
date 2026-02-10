from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, ConfigDict

from livekit.agents import JobContext

class STTProvider(str, Enum):
    DEEPGRAM = "deepgram"
    ASSEMBLYAI = "assemblyai"

class TTSProvider(str, Enum):
    CARTESIA = "cartesia"
    ELEVENLABS = "elevenlabs"

class LLMProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"


class CallDetails(BaseModel):
    livekit_call_id: str
    call_to: str
    call_from: str
    dispatch_rule: str
    trunk_id: str
    twilio_account_sid: Optional[str] = None
    twilio_call_sid: Optional[str] = None
    hostname: Optional[str] = None


class AgentConfig(BaseModel):
    user_id: str
    agent_id: str
    agent_name: str
    knowledge_base_id: str

    call_details: Optional[CallDetails] = None

    # LLM
    system_prompt: str
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = "qwen/qwen3-32b"
    max_tokens: int = 100

    # Voice
    tts_provider: TTSProvider = TTSProvider.CARTESIA
    voice_id: str = None
    emotion: Optional[str] = "Happy"
    speed: float = 1.0
    volume: float = 1.0

    # STT
    stt_provider: STTProvider = STTProvider.DEEPGRAM

    # Tools
    tools: List[str] = field(default_factory=list) # tool names, not functions

    # Behavior
    allow_interruptions: bool = True

    greeting: str = "Hello! How can I assist you today?"
    # Livekit JobContext
    ctx: JobContext = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )