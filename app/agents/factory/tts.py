from livekit.plugins import cartesia, elevenlabs

from app.agents import AgentConfig

class TTS:

    _providers = {
        "cartesia": lambda cfg: cartesia.TTS(
            model="sonic-turbo",
            voice=cfg.voice_id,
            emotion=cfg.emotion,
            speed=cfg.speed,
            volume=cfg.volume,
        ),
        "elevenlabs": lambda cfg: elevenlabs.TTS(
            model="eleven_turbo_v2_5",
            voice_id=cfg.voice_id,
        ),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.tts_provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported TTS provider: {cfg.tts_provider}")
