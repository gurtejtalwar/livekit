from dataclasses import dataclass, field
from typing import List, Optional

from livekit.agents import Agent
from livekit.plugins import deepgram, cartesia, groq
from livekit.plugins.turn_detector.english import EnglishModel

from app.services.agent.tools import TOOL_REGISTRY
from app.database.db import db

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

async def load_agent_config(customer_id: str, agent_id: str) -> AgentConfig:
    # TODO
    #  return hardcoded config
    return AgentConfig(
        agent_id=agent_id,
        system_prompt="You are a helpful customer support agent.",
        llm_provider="groq",
        llm_model="openai/gpt-oss-20b",
        max_tokens=100,
        tts_provider="cartesia",
        voice_id="6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
        emotion="Happy",
        speed=1.0,
        volume=2.0,
        stt_provider="deepgram",
        tools=["ask_knowledge_base", "get_current_time"],
        allow_interruptions=True,
        greeting="Hello! How can I assist you today?"
    )

class AgentFactory:
    @staticmethod
    def create_agent(config: AgentConfig) -> Agent:
        # ----- STT -----
        if config.stt_provider == "deepgram":
            stt = deepgram.STT()
        else:
            raise ValueError("Unsupported STT")

        # ----- LLM -----
        if config.llm_provider == "groq":
            llm = groq.LLM(
                model=config.llm_model,
                tool_choice="auto",
                max_completion_tokens=config.max_tokens,
            )
        else:
            raise ValueError("Unsupported LLM")

        # ----- TTS -----
        if config.tts_provider == "cartesia":
            tts = cartesia.TTS(
                model="sonic-turbo",
                voice=config.voice_id,
                emotion=config.emotion,
                speed=config.speed,
                volume=config.volume,
            )
        else:
            raise ValueError("Unsupported TTS")

        # ----- Tools -----
        tools = [TOOL_REGISTRY[name] for name in config.tools]

        return Agent(
            instructions=config.system_prompt,
            stt=stt,
            llm=llm,
            tts=tts,
            tools=tools,
            allow_interruptions=config.allow_interruptions,
            turn_detection=EnglishModel(),
            min_endpointing_delay=0.05,
            max_endpointing_delay=0.3,
        )

from bson import ObjectId
from fastapi import HTTPException
from typing import Dict, Any


async def get_agent_configuration(
    user_id: str,
    agent_id: str,
) -> Dict[str, Any]:
    agent_object_id = ObjectId(agent_id)

    pipeline = [
        {
            "$match": {
                "_id": agent_object_id
            }
        },
        {
            "$lookup": {
                "from": "voice-agent-config",
                "let": {"agentId": "$_id"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$eq": ["$agentId", "$$agentId"]
                            }
                        }
                    }
                ],
                "as": "agentConfigData"
            }
        },
        {
            "$lookup": {
                "from": "voice-voice-config",
                "let": {"agentId": "$_id"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$eq": ["$agentId", "$$agentId"]
                            }
                        }
                    }
                ],
                "as": "voiceConfigData"
            }
        },
        {
            "$project": {
                "_id": 1,
                "agentName": 1,
                "knowledgeBaseId": 1,
                "resourceCentreName": 1,
                "assignedPhoneNumberId": 1,
                "config": 1,
                "agentConfig": {
                    "$arrayElemAt": ["$agentConfigData", 0]
                },
                "voiceConfig": {
                    "$arrayElemAt": ["$voiceConfigData", 0]
                },
            }
        },
    ]

    result = await db["agent_collection"].aggregate(pipeline).to_list(length=1)

    if not result:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = result[0]
    agent_config = agent.get("agentConfig", {})
    voice_config = agent.get("voiceConfig", {})
    agent_base_config = agent.get("config", {})

    return {
        "_id": str(agent["_id"]),
        "name": agent.get("agentName"),
        "resourceCentreId": agent.get("knowledgeBaseId")
        or agent.get("resourceCentreName"),
        "phoneNumberId": agent.get("assignedPhoneNumberId"),
        "tools": agent_config.get("tools", []),
        "greeting": agent_config.get("welcomeMessage"),
        "llm": agent_config.get("llmModel")
        or agent_base_config.get("model"),
        "gptCustomization": agent_config.get(
            "gptCustomizationEnabled"
        ),
        "systemPrompt": agent_config.get("systemPrompt")
        or agent_base_config.get("systemPrompt"),
        "voice_id": voice_config.get("voiceType")
        or agent_base_config.get("voice"),
        "language": voice_config.get("language")
        or agent_base_config.get("language"),
        "speed": voice_config.get("speakingSpeed"),
        "tone": voice_config.get("tone"),
        "autoSwitchLanguage": voice_config.get("autoSwitchLanguage"),
        "emotionAwareResponse": voice_config.get("emotionAwareResponse"),
        "callerMemory": voice_config.get("callerMemory"),
    }
