from typing import Dict, Callable
from livekit.plugins import deepgram, assemblyai, elevenlabs

from app.agents import AgentConfig

class STT:

    _providers: Dict[str, Callable[[AgentConfig], object]]  = {
        "deepgram": lambda cfg: deepgram.STT(
            model=cfg.models.stt.model,
            language=cfg.models.stt.language
        ),
        "deepgram-v2": lambda cfg: deepgram.STTv2(
            model=cfg.models.stt.model
        ),
        "assemblyai": lambda cfg: assemblyai.STT(
            model=cfg.models.stt.model,
        ),
        "elevenlabs": lambda cfg: elevenlabs.STT(
            use_realtime=True,
        ),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.models.stt.provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported STT provider: {cfg.models.stt.provider}")
