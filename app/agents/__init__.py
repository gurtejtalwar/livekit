from typing import List, Optional, Literal
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

@dataclass
class UserData:
    user_id: str
    name: str
    email: str
    phone: str
    agent_id: str = ""
    user_timezone: str = ""
    user_current_time: str = ""
    call_id: str = None

class CallDetails(BaseModel):
    livekit_call_id: str
    call_to: str
    call_from: str
    dispatch_rule: Optional[str] = None
    trunk_id: str
    twilio_account_sid: Optional[str] = None
    twilio_call_sid: Optional[str] = None
    hostname: Optional[str] = None

class CallerDetails(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class ModelBase(BaseModel):
    model: str
    provider: str

class STTConfig(ModelBase):
    min_endpoiniting: float = None
    max_endpointing: float = None
class LLMConfig(ModelBase):
    max_tokens: int
class TTSConfig(ModelBase):
    voice_id: str
    emotion: str
    speed: str
    volume: str
    emotion: str

class AgentConfig(BaseModel):
    user_id: str
    agent_id: str
    agent_name: str
    knowledge_base_id: Optional[str] = None

    call_type: Literal["inbound", "outbound", "test-inbound", "test-outbound"] = None
    call_details: Optional[CallDetails] = None

    system_prompt: str

    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig

    language: Optional[str] = "English"
    additional_languages: List[str] = field(default_factory=list)

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