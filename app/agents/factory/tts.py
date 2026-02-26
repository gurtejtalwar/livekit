from livekit.plugins import cartesia, elevenlabs

from app.agents import AgentConfig

class TTS:

    _providers = {
        "cartesia": lambda cfg: cartesia.TTS(
            model=cfg.model,
            voice=cfg.voice_id,
            emotion=cfg.emotion,
            speed=cfg.speed,
            volume=cfg.volume,
        ),
        "elevenlabs": lambda cfg: elevenlabs.TTS(
            model=cfg.model,
            voice_id=cfg.voice_id,
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
            return cls._providers[cfg.tts_provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported TTS provider: {cfg.tts_provider}")
