import logging
from dataclasses import dataclass, field
from phonenumbers import timezone
from datetime import datetime
import pytz
import phonenumbers

from livekit.agents import Agent, llm, stt, tts
from livekit.plugins import deepgram, cartesia, groq, openai, elevenlabs, assemblyai, silero
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from app.agents.prompt.builder import PromptBuilder
from app.agents import AgentConfig, LeadContext
from app.agents.prompt import inbound as inbound_prompt
from app.agents.tools import resolve_tools
from app.models import call_models
from app.shared import schemas
from app.agents import helper
from app.agents import factory
from app.agents.workflow_manager import build_and_run_workflow
import asyncio

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
    def __init__(self,
                 config: AgentConfig):
        tools = resolve_tools(config)
        super().__init__(
            stt=config.lk_plugins.lk_stt,
            llm=config.lk_plugins.lk_llm,
            tts=config.lk_plugins.lk_tts,
            tools=tools,
            instructions=config.system_prompt,
            turn_detection=EnglishModel(),
            allow_interruptions=config.allow_interruptions,
            min_endpointing_delay=0.2,
            max_endpointing_delay=3,
            min_consecutive_speech_delay=0.4,
            use_tts_aligned_transcript=True
            # vad=vad,
        )

        self.config = config

    async def on_enter(self):
        logger.info("Node: on_enter called")
        if getattr(self.config, 'workflow_graph_json', None):
            logger.info("Initializing workflow engine from JSON payload.")
            await build_and_run_workflow(self, self.config.workflow_graph_json)
            
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
    async def get_lead_context(agent_id: str, admin_id: str, user_phone_number: str = "") -> LeadContext:
        lead_data = await helper.get_lead_data(admin_id, user_phone_number)
        summary = await helper.get_lead_agent_and_overall_summary(agent_id=agent_id, user_id=admin_id, phone_number=user_phone_number)
        
        return LeadContext(data=lead_data, summary=summary)
    
    @staticmethod
    async def load_agent_config(agent_id: str) -> AgentConfig:
        return await helper.load_agent_runtime_config(agent_id)    

    @staticmethod
    def from_config(cfg: AgentConfig,
                    lead_ctx: LeadContext) -> InboundAgent:
        cfg.lk_plugins.lk_stt = factory.STT.create(cfg)
        cfg.lk_plugins.lk_llm = factory.LLM.create(cfg)
        cfg.lk_plugins.lk_tts = factory.TTS.create(cfg)
        instructions = PromptBuilder(
            config=cfg,
            lead=lead_ctx,
            call_type=cfg.call_type
        ).build()
        cfg.system_prompt = instructions
        return InboundAgent(config=cfg)

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

