from typing import List, Optional
from dataclasses import dataclass, field

class STTProvider:
    DEEPGRAM = "deepgram"
    ASSEMBLYAI = "assemblyai"

class TTSProvider:
    CARTESIA = "cartesia"
    ELEVENLABS = "elevenlabs"

class LLMProvider:
    GROQ = "groq"
    OPENAI = "openai"

@dataclass
class AgentConfig:
    agent_id: str
    knowledge_base_id: str
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
