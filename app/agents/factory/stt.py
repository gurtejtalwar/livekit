from typing import Dict, Callable
from livekit.plugins import deepgram, assemblyai

from app.agents import AgentConfig

class STT:

    _providers: Dict[str, Callable[[AgentConfig], object]]  = {
        "deepgram": lambda cfg: deepgram.STT(
            model=cfg.models.stt.model
        ),
        "deepgram-v2": lambda cfg: deepgram.STTv2(
            model=cfg.models.stt.model
        ),
        "assemblyai": lambda cfg: assemblyai.STT(),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.models.stt.provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported STT provider: {cfg.models.stt.provider}")
