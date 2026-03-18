from typing import Dict, Callable
from livekit.plugins import groq

from app.agents import AgentConfig

class LLM:

    _providers: Dict[str, Callable[[AgentConfig], object]] = {
        "groq": lambda cfg: groq.LLM(
            model=cfg.models.llm.model,
            tool_choice="auto",
            max_completion_tokens=cfg.models.llm.max_tokens,
        ),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.models.llm.provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported LLM provider: {cfg.models.llm.provider}")
