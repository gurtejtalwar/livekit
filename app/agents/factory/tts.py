from typing import Callable, Dict
from livekit.plugins import cartesia, elevenlabs

from app.agents import AgentConfig

class TTS:

    _providers: Dict[str, Callable[[AgentConfig], object]] = {
        "cartesia": lambda cfg: cartesia.TTS(
            model=cfg.tts.model,
            voice=cfg.tts.voice_id,
            emotion=cfg.tts.emotion,
            speed=cfg.tts.speed,
            volume=cfg.tts.volume,
        ),
        "elevenlabs": lambda cfg: elevenlabs.TTS(
            model=cfg.tts.model,
            voice_id=cfg.tts.voice_id,
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
            return cls._providers[cfg.tts.provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported TTS provider: {cfg.tts_provider}")
