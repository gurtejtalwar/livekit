from typing import Dict, Callable
from livekit.plugins import deepgram, assemblyai

from app.agents import AgentConfig

class STT:

    _providers: Dict[str, Callable[[AgentConfig], object]]  = {
        "deepgram": lambda cfg: deepgram.STTv2(
            model=cfg.stt.model
        ),
        "assemblyai": lambda cfg: assemblyai.STT(),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.stt.provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported STT provider: {cfg.stt_provider}")
