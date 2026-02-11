import logging
import asyncio
import time
from dataclasses import dataclass, field

from livekit.agents import Agent
from livekit.plugins import deepgram, cartesia, groq, openai, elevenlabs
from livekit.plugins.turn_detector.english import EnglishModel

from app.services.agent import AgentConfig, CallDetails
from app.services.agent.prompt import inbound as inbound_prompt
from app.services.agent.tools import resolve_tools
from app.models import call_models
from app.shared import schemas

logger = logging.getLogger("factory")


async def load_agent_config(user_data, agent_id: str) -> AgentConfig:
    # TODO
    #  return hardcoded config
    return AgentConfig(
        user_id=str(user_data.id),
        agent_name="TestAgent",
        agent_id=str(agent_id),
        knowledge_base_id= "perceptyne" if agent_id == "perceptyne" else "eminence", #TODO
        system_prompt=inbound_prompt.f_prompt+f"\nUser Data: Name: {user_data.name}, Email: {user_data.email}, Phone: {user_data.phone}\n",
        llm_provider="groq",
        llm_model="qwen/qwen3-32b",
        max_tokens=1000,
        tts_provider="elevenlabs",
        voice_id="FGY2WhTYpPnrIDTdsKH5",#"820a3788-2b37-4d21-847a-b65d8a68c99a",
        # emotion="Determined",
        speed=0.75,
        volume=2.0,
        stt_provider="deepgram",
        tools=["end_call", "ask_knowledge_base", #TODO HAZARD
               "get_current_time", #"transfer_to_human",
               "book_appointment", "cancel_appointment", 
               "get_available_slots", "reschedule_appointment"],
        allow_interruptions=True,
        greeting="Hello! How can I assist you today?"
    )

class InboundAgent(Agent):
    def __init__(self, config: AgentConfig):
        tools = resolve_tools(config)
        super().__init__(
            instructions=config.system_prompt,
            stt=deepgram.STT(),
            llm=openai.LLM(
                model="gpt-5.1",
                max_completion_tokens=config.max_tokens,
            ),
            # groq.LLM(
            #     model=config.llm_model,
            #     tool_choice="auto",
            #     max_completion_tokens=config.max_tokens,
            # ),
            # tts=elevenlabs.TTS(voice_id="FGY2WhTYpPnrIDTdsKH5"),
            # tts=deepgram.TTS(),
            tts=cartesia.TTS(
                model="sonic-turbo",
                # voice=config.voice_id,
                # emotion=config.emotion,
                # speed=config.speed,
                # volume=config.volume,
            ),
            turn_detection=EnglishModel(),
            tools=tools,
            allow_interruptions=config.allow_interruptions,
            min_endpointing_delay=0.05,
            max_endpointing_delay=0.6,
        )

        self.config = config

    async def on_enter(self):
        logger.info("Node: on_enter called")
        await update_config_with_caller_context(self.config)
        await call_models.on_call_arrived(self.config, self.session)
    # def sync_wrapper(metrics: LLMMetrics):
        #     asyncio.create_task(self.on_metrics_collected(metrics))

        # self.session.llm.on("metrics_collected", sync_wrapper)
        # self.session.generate_reply()
    
    async def stt_node(self,
                 audio,
                 model_settings):
        logger.info("Node: stt_node called")
        return self.default.stt_node(self, audio, model_settings)


    async def llm_node(self, chat_ctx, tools, model_settings):
        logger.info("Node: llm_node called")
        return self.default.llm_node(self, chat_ctx, tools, model_settings)
    
    async def trancription_node(self, text, model_settings):
        logger.info("Node: transcription_node called")
        return self.default.transcription_node(self, text, model_settings)
          
    async def tts_node(self,
                 text,
                 model_settings):
        logger.info("Node: tts_node called")
        return self.default.tts_node(self, text, model_settings)
    
    async def on_exit(self):
        logger.info("Node: on_exit called")

class AgentFactory:
    @staticmethod
        #TODO move to InboundAgent
    def from_config(cfg: AgentConfig) -> Agent: 
        #TODO move to InboundAgent
        # ----- STT -----
        #TODO need class methods
        if cfg.stt_provider == "deepgram":
            stt = deepgram.STT()
        else:
            raise ValueError("Unsupported STT")

        # ----- LLM -----
        #TODO need class methods
        if cfg.llm_provider == "groq":
            llm = groq.LLM(
                model=cfg.llm_model,
                tool_choice="auto",
                max_completion_tokens=cfg.max_tokens,
            )
        else:
            raise ValueError("Unsupported LLM")

        # ----- TTS -----
        #TODO need class methods
        if cfg.tts_provider == "cartesia":
            tts = cartesia.TTS(
                model="sonic-turbo",
                voice=cfg.voice_id,
                emotion=cfg.emotion,
                speed=cfg.speed,
                volume=cfg.volume,
            )
        if cfg.tts_provider == "elevenlabs":
            tts = elevenlabs.TTS(
                model="eleven_turbo_v2_5",
                voice_id=cfg.voice_id,
                # emotion=config.emotion,
                # speed=config.speed,
                # volume=config.volume,
            )
        else:
            raise ValueError("Unsupported TTS")

        # tools = resolve_tools(cfg)
        #TODO move to InboundAgent

        return InboundAgent(cfg)
        # return Agent(
        #     instructions=inbound_prompt.f_prompt,
        #     stt=stt,
        #     llm=llm,
        #     tts=tts,
        #     tools=tools,
        #     allow_interruptions=config.allow_interruptions,
        #     turn_detection=EnglishModel(),
        #     min_endpointing_delay=0.05,
        #     max_endpointing_delay=0.3,
        # )

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

#TODO Data redundancy, can be fetched directly from context
async def update_config_with_caller_context(config: AgentConfig) -> AgentConfig:
    # 1. Identify the SIP Participant
    # Usually, in a telephony call, there is only one remote participant
    caller = None
    for p in config.ctx.room.remote_participants.values():
        if p.identity.startswith("sip_"):
            caller = p
            break

    # 2. Extract the IDs from attributes
    trunk_id = None
    dispatch_rule_id = None
    
    if caller:
        # LiveKit populates these specific keys for SIP calls
        livekit_call_id = caller.attributes.get("sip.callID")
        trunk_id = caller.attributes.get("sip.trunkID")
        dispatch_rule_id = caller.attributes.get("sip.ruleID")
        call_from = caller.attributes.get("sip.phoneNumber")
        call_to = caller.attributes.get("sip.trunkPhoneNumber")
        hostname = caller.attributes.get("sip.hostname")
        twilio_account_sid = caller.attributes.get("sip.twilio.accountSid")
        twilio_call_sid = caller.attributes.get("sip.twilio.callSid")
        
        logger.info(f"Inbound Call: Trunk={trunk_id}, Dispatch={dispatch_rule_id}")
   
    
        # Example: resolve SIP details from LiveKit JobContext
        config.call_details = CallDetails(
            livekit_call_id=livekit_call_id,
            trunk_id=trunk_id,
            dispatch_rule=dispatch_rule_id,
            call_to=call_to,
            call_from=call_from,
            twilio_call_sid=twilio_call_sid,
            twilio_account_sid=twilio_account_sid,
            hostname=hostname,
        )
        
    return config
