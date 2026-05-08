from typing import Callable, Dict
from livekit.plugins import cartesia, elevenlabs

from app.agents import AgentConfig

class TTS:

    _providers: Dict[str, Callable[[AgentConfig], object]] = {
        "cartesia": lambda cfg: cartesia.TTS(
            model=cfg.models.tts.model,
            voice=cfg.models.tts.voice_id,
            emotion=cfg.models.tts.emotion,
            speed=cfg.models.tts.speed,
            volume=cfg.models.tts.volume,
            language=cfg.models.tts.language,
        ),
        "elevenlabs": lambda cfg: elevenlabs.TTS(
            model=cfg.models.tts.model,
            voice_id=cfg.models.tts.voice_id,
            # voice_settings= elevenlabs.VoiceSettings(
            #     stability=
            #     similarity_boost= 
            #     speed= 
            #     use_speaker_boost= 
            # )
        ),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.models.tts.provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported TTS provider: {cfg.models.tts.provider}")
