from livekit.plugins import deepgram, assemblyai

from app.agent import AgentConfig

class STT:

    _providers = {
        "deepgram": lambda cfg: deepgram.STT(),
        "assemblyai": lambda cfg: assemblyai.STT(),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.stt_provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported STT provider: {cfg.stt_provider}")
