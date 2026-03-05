"""Settings (env) and AppConfig (YAML) for e-clawhisper."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

##### ENUMS #####


class AgentBackend(StrEnum):
    OPENFANG = auto()


class STTBackend(StrEnum):
    WHISPERLIVE = auto()


class TTSBackend(StrEnum):
    PIPER = auto()


##### CONSTANTS #####

_DEFAULT_VOICES: dict[str, str] = {
    "en": "en_US-lessac-medium",
    "es": "es_ES-sharvard-medium",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "it": "it_IT-riccardo-x_low",
    "pt": "pt_BR-faber-medium",
}

##### YAML CONFIG MODELS #####


class WakewordConfig(BaseModel):
    """Wake word detection (OpenWakeWord ONNX)."""

    model: str = "alexa"
    threshold: float = 0.5


class SentinelConfig(BaseModel):
    """Sentinel pipeline — passive listening."""

    energy_floor: float = 0.01
    vad_threshold: float = 0.5
    cooldown: float = 1.5
    wakeword: WakewordConfig = WakewordConfig()


class VADConfig(BaseModel):
    """Turn-pipeline VAD — end-of-speech detection (Silero ONNX)."""

    threshold: float = 0.5
    silence_duration: float = 1.5
    min_recording_time: float = 1.0


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
    finish_timeout: float = 5.0


class STTConfig(BaseModel):
    backend: STTBackend = STTBackend.WHISPERLIVE
    whisperlive: WhisperLiveConfig = WhisperLiveConfig()


class PiperConfig(BaseModel):
    host: str = "localhost"
    port: int = 10200
    voice: str = "en_US-lessac-medium"
    sample_rate: int = 22050
    disconnect_timeout: float = 0.5


class TTSConfig(BaseModel):
    backend: TTSBackend = TTSBackend.PIPER
    piper: PiperConfig = PiperConfig()


class AgentConfig(BaseModel):
    name: str = "damien"
    backend: AgentBackend = AgentBackend.OPENFANG


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    pre_roll_seconds: float = 2.0
    queue_size: int = 100
    pcm_queue_size: int = 20


class LoggingConfig(BaseModel):
    idle_interval: float = 0.5
    turn_interval: float = 0.5


class AppConfig(BaseModel):
    """Application config loaded from config.yaml."""

    language: str = "en"
    turn_timeout: float = 120.0
    agent: AgentConfig = AgentConfig()
    sentinel: SentinelConfig = SentinelConfig()
    vad: VADConfig = VADConfig()
    backends: BackendsConfig = BackendsConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    audio: AudioConfig = AudioConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def cascade_language(self) -> AppConfig:
        """Propagate top-level language to STT and TTS when at defaults."""
        lang = self.language
        if self.stt.whisperlive.language == "en" and lang != "en":
            self.stt.whisperlive.language = lang
        if self.tts.piper.voice == "en_US-lessac-medium" and lang != "en":
            self.tts.piper.voice = _DEFAULT_VOICES.get(lang, self.tts.piper.voice)
        return self

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
    MODELS_DIR: ClassVar[Path] = BASE_DIR / "models"

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
