import logging
from dataclasses import dataclass, field
from phonenumbers import timezone
from datetime import datetime
import pytz
import phonenumbers

from livekit.agents import Agent, llm
from livekit.plugins import deepgram, cartesia, groq, openai, elevenlabs, assemblyai, silero
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from app.agent import AgentConfig, CallDetails
from app.agent.prompt import inbound as inbound_prompt
from app.agent.tools import resolve_tools
from app.models import call_models
from app.shared import schemas
from app.agent import helper
from app.agent import factory

logger = logging.getLogger("factory")


language_names = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
}

deepgram_language_codes = {
    "en": "en",
    "es": "es",
    "fr": "fr-CA",
    "de": "de",
    "it": "it",
}
            
cartesia_language_codes = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
}

vad = silero.VAD.load(
    activation_threshold=0.6,
    prefix_padding_duration=0.5,
    min_silence_duration=1,
    sample_rate=8000,
)

class InboundAgent(Agent):
    def __init__(self, config: AgentConfig):
        tools = resolve_tools(config)
        super().__init__(
            instructions=config.system_prompt,
            stt=assemblyai.STT(),#language="multi"), #TODO can be set dynamically based on agent config
            # stt=assemblyai.STT(model="universal-streaming-multilingual"),
            llm=groq.LLM(
                model="openai/gpt-oss-20b",
                tool_choice="auto",
                max_completion_tokens=config.max_tokens,
                ),
            # openai.LLM(
            #     model="gpt-5.1",
            #     max_completion_tokens=config.max_tokens,
            # ),

            # tts=elevenlabs.TTS(voice_id="FGY2WhTYpPnrIDTdsKH5"),
            # tts=deepgram.TTS(),
            tts=cartesia.TTS(
                # language=
                model="sonic-turbo",
                voice= config.voice_id,#config.voice_id,
                emotion="Excited",
                speed=0.5,
                # volume=config.volume,
            ),
            turn_detection=EnglishModel(),
            tools=tools,
            allow_interruptions=config.allow_interruptions,
            min_endpointing_delay=0.6,
            max_endpointing_delay=0.7,
            # vad=vad,
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

    async def _switch_language(self, language_code: str) -> None:
        """Helper method to switch the language"""
        # if language_code == current_language:
        #     # await session.say(f"I'm already speaking in {language_names[language_code]}.")
        #     return

        if self.session.tts is not None:
            self.session.tts.update_options(language=language_code)

        if self.session.stt is not None:
            deepgram_language = deepgram_language_codes.get(language_code, language_code)
            self.session.stt.update_options(language=deepgram_language)

        current_language = language_code


    @llm.function_tool
    async def switch_to_english(self, reason: str):
        """Switch to speaking English"""
        await self._switch_language("en")

    @llm.function_tool
    async def switch_to_spanish(self, reason: str):
        """Switch to speaking Spanish"""
        await self._switch_language("es")

    @llm.function_tool
    async def switch_to_french(self, reason: str):
        """Switch to speaking French"""
        await self._switch_language("fr")

    @llm.function_tool
    async def switch_to_german(self, reason: str):
        """Switch to speaking German"""
        await self._switch_language("de")

    @llm.function_tool
    async def switch_to_italian(self, reason: str):
        """Switch to speaking Italian"""
        await self._switch_language("it")

    @llm.function_tool
    async def switch_to_hindi(self, reason: str):
        """Switch to speaking Hindi"""
        await self._switch_language("hi")  
class AgentFactory:
    @staticmethod
    async def load_agent_config(user_data, agent_id: str) -> AgentConfig:
        return await helper.load_agent_runtime_config(agent_id, user_data)    

    @staticmethod
    def from_config(cfg: AgentConfig) :#-> Agent: 
        stt = factory.STT.create(cfg)
        llm = factory.LLM.create(cfg)
        tts = factory.TTS.create(cfg)
        return InboundAgent(cfg)

    @staticmethod
    async def get_time_from_phone(phone_number: str):
        try:
            # Parse number
            parsed_number = phonenumbers.parse(phone_number)

            # Get timezone(s)
            timezones = timezone.time_zones_for_number(parsed_number)

            if not timezones:
                return "Timezone not found"

            # Use first timezone (some countries have multiple)
            tz = pytz.timezone(timezones[0])

            # Get current time
            current_time = datetime.now(tz)

            return {
                "timezone": timezones[0],
                "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            return f"Error: {str(e)}"

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

