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

prompt ="""
                "You are a Eminence Technology customer service AI assistant. "
                "For ANY Eminence Technology-related or factual question, you MUST use the 'ask_knowledge_base' tool FIRST. "
                "Do not rely on your internal memory. "
                "For ANY appointment booking related information/actions you have access to the following tools: 
                book_appointment: Use this tool to book new appointments for customers., 
                cancel_appointment: Use this tool to cancel existing appointments for customers., 
                get_available_slots: Use this tool to check available appointment slots., 
                reschedule_appointment: Use this tool to reschedule existing appointments for customers."

                "After receiving the tool's output, use it to construct a conversational, human-like answer. "
                "If the tool returns no relevant data, politely say you don't have enough information. "
                "Keep responses concise and optimized for spoken delivery. PLEASE MAKE SURE THAT THE RESPONSES ARE SHORT SO THAT IT MIMICKS A PHONE CONVERSATION BETWEEN HUMANS. "
                "Do not respond with asterick, bullet points,etc  please respond how you would in a normal conversation with a human. "
                "PLEASE keep your tone friendly and enthusiastic. Always Respond politely to the customer. You are allowed to do small talks with the customer BUT DO NOT STRAY AWAY FROM THE BUSINESS AND OBJECTIVE OF THE CONVERSATION"
                "Format numbers naturally (e.g., 'five hundred and twelve gigabytes')." 
"""

async def load_agent_config(customer_id: str, agent_id: str) -> AgentConfig:
    # TODO
    #  return hardcoded config
    return AgentConfig(
        agent_id=agent_id,
        system_prompt=prompt,
        llm_provider="groq",
        llm_model="openai/gpt-oss-20b",
        max_tokens=100,
        tts_provider="cartesia",
        voice_id="6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
        emotion="Happy",
        speed=1.0,
        volume=2.0,
        stt_provider="deepgram",
        tools=["ask_knowledge_base", "get_current_time", 
               "book_appointment", "cancel_appointment", 
               "get_available_slots", "reschedule_appointment"],
        allow_interruptions=True,
        greeting="Hello! How can I assist you today?"
    )

class AgentFactory:
    @staticmethod
    def create_agent(config: AgentConfig) -> Agent:
        # ----- STT -----
        #TODO need class methods
        if config.stt_provider == "deepgram":
            stt = deepgram.STT()
        else:
            raise ValueError("Unsupported STT")

        # ----- LLM -----
        #TODO need class methods
        if config.llm_provider == "groq":
            llm = groq.LLM(
                model=config.llm_model,
                tool_choice="auto",
                max_completion_tokens=config.max_tokens,
            )
        else:
            raise ValueError("Unsupported LLM")

        # ----- TTS -----
        #TODO need class methods
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
