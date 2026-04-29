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
            # use_realtime=True,
            language_code=cfg.models.stt.language,
            model_id="scribe_v2_realtime",
            server_vad=elevenlabs.stt.VADOptions(
                vad_threshold=0.6,               # Less sensitive to noise
                min_speech_duration_ms=400,      # Ignore brief sounds/pops
                vad_silence_threshold_secs=0.9,   # Wait a beat to be sure
                min_silence_duration_ms=500,     # Prevent rapid segmenting
            )
        ),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.models.stt.provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported STT provider: {cfg.models.stt.provider}")
