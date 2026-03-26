from typing import List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator

from livekit.agents import JobContext, stt, llm, tts

CARTESIA_SPEED_MAP = {
    "slowest": 0.5,
    "slow": 0.75,
    "normal": 1.0,
    "fast": 1.25,
    "fastest": 1.5,
}

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
    outbound_trunk_id: str = "" #TODO WRAP IN OBJECT
    human_escalation_phone: str = "" #TODO WRAP IN OBJECT

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
    language: str = "en"
class LLMConfig(ModelBase):
    max_tokens: int
class TTSConfig(ModelBase):
    voice_id: str
    emotion: str
    speed: str|float
    volume: str|float
    emotion: str
    language: str = "en"

    # Ensure volume is always float
    # @field_validator("volume", mode="before")
    # @classmethod
    # def cast_volume(cls, v):
    #     if v is None:
    #         return None
    #     return float(v)


    # Normalize speed depending on model
    @field_validator("speed", mode="after")
    @classmethod
    def normalize_speed(cls, v, info):
        model = info.data.get("model")

        # Try converting numeric strings → float
        if isinstance(v, str):
            try:
                v = float(v)
            except:
                pass  # keep as string if not numeric

        # sonic-3 → must be float OR mapped enum
        if model == "sonic-3":
            if isinstance(v, (int, float)):
                return float(v)

            if v in CARTESIA_SPEED_MAP:
                return CARTESIA_SPEED_MAP[v]

            raise ValueError(f"Invalid speed: {v}")

        # sonic-2 / turbo → allow enum or raw string
        return v
    
class SIPConfig(BaseModel):
    outbound_trunk_id: str

class LivekitPlugins(BaseModel):
    lk_stt: stt.STT = None
    lk_llm: llm.LLM = None
    lk_tts: tts.TTS = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
class ModelConfig(BaseModel):
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig

class AgentConfig(BaseModel):
    models: ModelConfig
    lk_plugins: LivekitPlugins = LivekitPlugins()
    
    user_id: str
    agent_id: str
    agent_name: str
    knowledge_base_id: Optional[str] = None
    workflow_graph_json: Optional[dict] = None

    call_type: Literal["inbound", "outbound", "test-inbound", "test-outbound"] = None
    call_details: Optional[CallDetails] = None

    system_prompt: str


    language: Optional[str] = "English"
    additional_languages: List[str] = field(default_factory=list)

    # Tools
    tools: List[str] = field(default_factory=list) # tool names, not functions

    # Behavior
    allow_interruptions: bool = True
    allow_recording: bool

    greeting: str = "Hello! How can I assist you today?"
    # Livekit JobContext
    ctx: JobContext = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )
    max_duration: Optional[int] = None #TODO WRAP IN OBJECT
    outbound_trunk_id: str
    human_phone_number: Optional[str] = None