from livekit.plugins import groq

from app.agents import AgentConfig

class LLM:

    _providers = {
        "groq": lambda cfg: groq.LLM(
            model=cfg.llm_model,
            tool_choice="auto",
            max_completion_tokens=cfg.max_tokens,
        ),
    }

    @classmethod
    def create(cls, cfg: AgentConfig):
        try:
            return cls._providers[cfg.llm_provider](cfg)
        except KeyError:
            raise ValueError(f"Unsupported LLM provider: {cfg.llm_provider}")
