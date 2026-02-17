from typing import Any, Literal
from pydantic import MongoDsn
from functools import cache
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

class Settings(BaseSettings):
    ENVIRONMENT: Literal["prod", "dev", "qa", "stg", "local"]
    # LiveKit Configuration
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_URL: str

    # Connections
    DEEPGRAM_API_KEY: str
    ASSEMBLYAI_API_KEY: str

    CARTESIA_API_KEY: str
    ELEVEN_API_KEY: str

    GROQ_API_KEY: str
    OPENAI_API_KEY: str

    # Database Configuration
    MONGO_URI: str
    MONGO_DB: str

    # ISC
    N1_ISC_URL: str
    N1_ISC_API_KEY: str

    N3_ISC_URL: str
    N3_ISC_API_KEY: str
    
    P1_ISC_URL: str
    P1_ISC_API_KEY: str

    TTS_CARTESIA_DEFAULT_VOICE_ID: str
    TTS_ELEVENLABS_DEFAULT_VOICE_ID: str
    LLM_GROQ_DEFAULT_MODEL: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, dotenv_settings
    

@cache
def get_settings() -> Settings:
    return Settings()