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
    GENERIC = auto()


class LLMProvider(StrEnum):
    GEMINI = auto()
    OPENAI = auto()
    ANTHROPIC = auto()
    VLLM = auto()


class STTBackend(StrEnum):
    WHISPERLIVE = auto()


class TTSBackend(StrEnum):
    PIPER = auto()
    KOKORO = auto()


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
    model: str = "alexa"
    threshold: float = 0.5


class SentinelConfig(BaseModel):
    energy_floor: float = 0.01
    vad_threshold: float = 0.5
    cooldown: float = 1.5
    wakeword: WakewordConfig = WakewordConfig()


class VADConfig(BaseModel):
    threshold: float = 0.5
    silence_duration: float = 1.5
    min_recording_time: float = 1.0


class OpenFangConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 4200
    timeout: float = 30.0


class GenericLLMConfig(BaseModel):
    """Multi-provider LLM config (PydanticAI)."""

    provider: LLMProvider = LLMProvider.GEMINI
    model: str = "gemini-2.0-flash"
    api_key: str = ""
    timeout: float = 60.0
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    # vLLM only
    host: str = "localhost"
    port: int = 45100


class BackendsConfig(BaseModel):
    openfang: OpenFangConfig = OpenFangConfig()
    generic: GenericLLMConfig = GenericLLMConfig()


class WhisperLiveConfig(BaseModel):
    host: str = "localhost"
    port: int = 45120
    model: str = "large-v3"
    language: str = "en"
    finish_timeout: float = 5.0


class STTConfig(BaseModel):
    backend: STTBackend = STTBackend.WHISPERLIVE
    whisperlive: WhisperLiveConfig = WhisperLiveConfig()


class PiperConfig(BaseModel):
    host: str = "localhost"
    port: int = 45130
    voice: str = "es_ES-davefx-medium"
    sample_rate: int = 22050
    disconnect_timeout: float = 0.5


class KokoroConfig(BaseModel):
    host: str = "localhost"
    port: int = 45130
    model: str = "kokoro"
    voice: str = "em_alex"
    sample_rate: int = 24000
    response_format: str = "pcm"
    timeout: float = 30.0


class TTSConfig(BaseModel):
    backend: TTSBackend = TTSBackend.KOKORO
    piper: PiperConfig = PiperConfig()
    kokoro: KokoroConfig = KokoroConfig()


class AgentConfig(BaseModel):
    name: str = "damien"
    backend: AgentBackend = AgentBackend.OPENFANG


class ConversationConfig(BaseModel):
    enabled: bool = True
    loop_timeout: float = 2.0
    max_turns: int = 10


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512
    queue_size: int = 100
    pcm_queue_size: int = 20
    playback_latency: str = "high"


class LoggingConfig(BaseModel):
    idle_interval: float = 0.25
    turn_interval: float = 0.25


class AppConfig(BaseModel):
    """Application config loaded from config.yaml."""

    language: str = "en"
    turn_timeout: float = 300.0
    agent: AgentConfig = AgentConfig()
    sentinel: SentinelConfig = SentinelConfig()
    vad: VADConfig = VADConfig()
    conversation: ConversationConfig = ConversationConfig()
    backends: BackendsConfig = BackendsConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    audio: AudioConfig = AudioConfig()
    logging: LoggingConfig = LoggingConfig()

    @property
    def tts_sample_rate(self) -> int:
        match self.tts.backend:
            case TTSBackend.KOKORO:
                return self.tts.kokoro.sample_rate
            case _:
                return self.tts.piper.sample_rate

    @model_validator(mode="after")
    def cascade_language(self) -> AppConfig:
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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ENVIRONMENT: str = "DEV"
    LOG_LEVEL: str = "debug"
    SOCKET_PATH: str = "/tmp/e-claw.sock"
    AGENT_BACKEND: AgentBackend | None = None

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: ClassVar[Path] = BASE_DIR / "data"
    CONFIG_PATH: ClassVar[Path] = BASE_DIR / "config.yaml"
    MODELS_DIR: ClassVar[Path] = BASE_DIR / "models"

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT == "DEV"


settings = Settings()


def load_config(path: Path | None = None) -> AppConfig:
    """Load AppConfig from YAML. AGENT_BACKEND env var overrides yaml when set."""
    config_path = path or settings.CONFIG_PATH
    config = AppConfig.from_yaml(config_path) if config_path.exists() else AppConfig()
    if settings.AGENT_BACKEND is not None:
        config.agent.backend = settings.AGENT_BACKEND
    return config
