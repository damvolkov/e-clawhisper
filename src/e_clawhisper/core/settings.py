"""Unified settings for e_clawhisper."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for e-clawhisper voice channel bridge."""

    ENVIRONMENT: Literal["DEV", "PROD"] = "DEV"

    BASE_DIR: ClassVar[Path] = Path(__file__).parent.parent.parent.parent

    ##### AGENT IDENTITY #####

    AGENT_NAME: str = "damien"
    AGENT_ID: str = ""

    ##### CHANNEL BACKEND #####

    CHANNEL_BACKEND: Literal["openfang", "zeroclaw"] = "openfang"
    OPENFANG_HOST: str = "127.0.0.1"
    OPENFANG_PORT: int = 4200
    OPENFANG_TIMEOUT: float = 30.0

    ##### STT (WHISPERLIVE) #####

    STT_HOST: str = "localhost"
    STT_PORT: int = 9090
    STT_MODEL: str = "small"
    STT_LANGUAGE: str = "en"
    STT_TIMEOUT: float = 30.0

    ##### TTS (PIPER) #####

    TTS_HOST: str = "localhost"
    TTS_PORT: int = 10200
    TTS_SAMPLE_RATE: int = 22050
    TTS_SILENCE_PADDING: float = 0.15
    PIPER_VOICE: str = "en_US-lessac-medium"

    ##### LIVEKIT #####

    LIVEKIT_URL: str = "ws://localhost:7880"
    LIVEKIT_API_KEY: str = "devkey"
    LIVEKIT_API_SECRET: str = "secret"

    ##### AUDIO #####

    AUDIO_SAMPLE_RATE: int = 24000
    AUDIO_CHANNELS: int = 1

    ##### CONVERSATION #####

    CONVERSATION_TIMEOUT: float = 30.0

    ##### COMPUTED #####

    @computed_field
    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT == "DEV"

    @computed_field
    @property
    def log_level(self) -> str:
        return "debug" if self.is_dev else "info"

    @computed_field
    @property
    def stt_ws_url(self) -> str:
        return f"ws://{self.STT_HOST}:{self.STT_PORT}"

    @computed_field
    @property
    def openfang_base_url(self) -> str:
        return f"http://{self.OPENFANG_HOST}:{self.OPENFANG_PORT}"

    @computed_field
    @property
    def openfang_ws_url(self) -> str:
        return f"ws://{self.OPENFANG_HOST}:{self.OPENFANG_PORT}"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
