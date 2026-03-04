"""Settings (env) and AppConfig (YAML) for e-clawhisper."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


##### ENUMS #####


class AgentBackend(StrEnum):
    OPENFANG = auto()


class STTBackend(StrEnum):
    WHISPERLIVE = auto()


class TTSBackend(StrEnum):
    PIPER = auto()


class VADBackend(StrEnum):
    TEN_VAD = auto()


##### YAML CONFIG MODELS #####


class OpenFangConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 4200
    timeout: float = 30.0


class BackendsConfig(BaseModel):
    openfang: OpenFangConfig = OpenFangConfig()


class WhisperLiveConfig(BaseModel):
    host: str = "localhost"
    port: int = 9090
    model: str = "small"
    language: str = "en"


class STTConfig(BaseModel):
    backend: STTBackend = STTBackend.WHISPERLIVE
    whisperlive: WhisperLiveConfig = WhisperLiveConfig()


class PiperConfig(BaseModel):
    host: str = "localhost"
    port: int = 10200
    voice: str = "en_US-lessac-medium"
    sample_rate: int = 22050


class TTSConfig(BaseModel):
    backend: TTSBackend = TTSBackend.PIPER
    piper: PiperConfig = PiperConfig()


class VADConfig(BaseModel):
    backend: VADBackend = VADBackend.TEN_VAD
    threshold: float = 0.5
    silence_duration: float = 1.5


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512


class AgentConfig(BaseModel):
    name: str = "damien"
    backend: AgentBackend = AgentBackend.OPENFANG


class AppConfig(BaseModel):
    """Application config loaded from config.yaml."""

    agent: AgentConfig = AgentConfig()
    backends: BackendsConfig = BackendsConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    vad: VADConfig = VADConfig()
    audio: AudioConfig = AudioConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> AppConfig:
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})


##### ENV SETTINGS #####


class Settings(BaseSettings):
    """Environment-level settings (secrets, runtime paths)."""

    ENVIRONMENT: str = "DEV"
    LOG_LEVEL: str = "debug"
    SOCKET_PATH: str = "/tmp/e-claw.sock"

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: ClassVar[Path] = BASE_DIR / "data"
    CONFIG_PATH: ClassVar[Path] = BASE_DIR / "config.yaml"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT == "DEV"


settings = Settings()


def load_config(path: Path | None = None) -> AppConfig:
    """Load AppConfig from YAML, falling back to defaults."""
    config_path = path or settings.CONFIG_PATH
    if config_path.exists():
        return AppConfig.from_yaml(config_path)
    return AppConfig()
