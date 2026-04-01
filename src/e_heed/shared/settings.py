"""Settings (env) and AppConfig (YAML) for e-heed."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, ClassVar, Literal
from urllib.parse import urlparse

import yaml
from annotated_types import Ge, Gt, Le
from pydantic import AnyUrl, BaseModel, ConfigDict, HttpUrl, UrlConstraints, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

##### TYPES #####

Probability = Annotated[float, Ge(0.0), Le(1.0)]
PositiveSeconds = Annotated[float, Gt(0.0)]
PositiveInt = Annotated[int, Gt(0)]
WebSocketUrl = Annotated[AnyUrl, UrlConstraints(allowed_schemes=["ws", "wss"])]
TcpUrl = Annotated[AnyUrl, UrlConstraints(allowed_schemes=["tcp"])]

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
    EVOICE = auto()


class TTSBackend(StrEnum):
    PIPER = auto()
    KOKORO = auto()
    EVOICE = auto()


class Environment(StrEnum):
    DEV = "DEV"
    PROD = "PROD"


class LogLevel(StrEnum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


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
    model_config = ConfigDict(extra="forbid")

    model: str = "alexa"
    threshold: Probability = 0.5


class SentinelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy_floor: Probability = 0.01
    vad_threshold: Probability = 0.5
    cooldown: PositiveSeconds = 1.5
    wakeword: WakewordConfig = WakewordConfig()


class VADConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: Probability = 0.5
    silence_duration: PositiveSeconds = 1.5
    min_recording_time: PositiveSeconds = 1.0


class OpenFangConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl = "http://127.0.0.1:4200"
    timeout: PositiveSeconds = 30.0

    @property
    def ws_base(self) -> str:
        """Derive WebSocket base from HTTP URL."""
        return str(self.url).rstrip("/").replace("http://", "ws://").replace("https://", "wss://")


class GenericLLMConfig(BaseModel):
    """Multi-provider LLM config (PydanticAI)."""

    model_config = ConfigDict(extra="forbid")

    provider: LLMProvider = LLMProvider.GEMINI
    model: str = "gemini-2.0-flash"
    api_key: str = ""
    timeout: PositiveSeconds = 60.0
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    url: HttpUrl = "http://localhost:45100"


class BackendsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    openfang: OpenFangConfig = OpenFangConfig()
    generic: GenericLLMConfig = GenericLLMConfig()


class WhisperLiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: WebSocketUrl = "ws://localhost:45120"
    model: str = "large-v3"
    language: str = "en"
    finish_timeout: PositiveSeconds = 5.0


class PiperConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: TcpUrl = "tcp://localhost:45130"
    voice: str = "es_ES-davefx-medium"
    sample_rate: PositiveInt = 22050
    disconnect_timeout: PositiveSeconds = 0.5

    @property
    def host(self) -> str:
        return urlparse(str(self.url)).hostname or "localhost"

    @property
    def port(self) -> int:
        return urlparse(str(self.url)).port or 10200


class KokoroConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl = "http://localhost:45130"
    model: str = "kokoro"
    voice: str = "em_alex"
    sample_rate: PositiveInt = 24000
    response_format: Literal["pcm", "wav", "mp3", "opus", "flac"] = "pcm"
    timeout: PositiveSeconds = 30.0


class EVoiceSTTConfig(BaseModel):
    """EVoice STT config — WebSocket streaming transcription."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl = "http://localhost:45140"
    language: str = "es"
    response_format: Literal["json", "text", "verbose_json"] = "json"


class EVoiceTTSConfig(BaseModel):
    """EVoice TTS config — WebSocket streaming synthesis."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl = "http://localhost:45140"
    voice: str = "af_heart"
    speed: float = 1.0
    lang: str = "en-us"
    sample_rate: PositiveInt = 24000


class STTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: STTBackend = STTBackend.EVOICE
    whisperlive: WhisperLiveConfig = WhisperLiveConfig()
    evoice: EVoiceSTTConfig = EVoiceSTTConfig()


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: TTSBackend = TTSBackend.EVOICE
    piper: PiperConfig = PiperConfig()
    kokoro: KokoroConfig = KokoroConfig()
    evoice: EVoiceTTSConfig = EVoiceTTSConfig()


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "damien"
    backend: AgentBackend = AgentBackend.OPENFANG


class ConversationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    loop_timeout: PositiveSeconds = 2.0
    max_turns: PositiveInt = 10


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_rate: PositiveInt = 16000
    channels: PositiveInt = 1
    chunk_size: PositiveInt = 512
    queue_size: PositiveInt = 100
    pcm_queue_size: PositiveInt = 20
    playback_latency: Literal["low", "medium", "high"] = "high"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idle_interval: PositiveSeconds = 0.25
    turn_interval: PositiveSeconds = 0.25


class AppConfig(BaseModel):
    """Application config loaded from config.yaml."""

    model_config = ConfigDict(extra="forbid")

    language: str = "en"
    turn_timeout: PositiveSeconds = 300.0
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
            case TTSBackend.EVOICE:
                return self.tts.evoice.sample_rate
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

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        """Load from YAML with env overrides applied."""
        config_path = path or settings.CONFIG_PATH
        data = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        if (backend := settings.AGENT_BACKEND) is not None:
            data.setdefault("agent", {})["backend"] = backend
        return cls.model_validate(data or {})


##### ENV SETTINGS #####

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

_XDG_CONFIG = Path.home() / ".config" / "e-heed"
_ETC_CONFIG = Path("/etc/e-heed")

_CONFIG_CANDIDATES = (
    _XDG_CONFIG / "config.yaml",
    _ETC_CONFIG / "config.yaml",
    _BASE_DIR / "config.yaml",
)


def _resolve_config_path() -> Path:
    """Resolve config: $CONFIG_PATH env > ~/.config > /etc > ./config.yaml."""
    return next((p for p in _CONFIG_CANDIDATES if p.exists()), _BASE_DIR / "config.yaml")


class Settings(BaseSettings):
    """Environment-level settings (secrets, runtime paths)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ENVIRONMENT: Environment = Environment.DEV
    LOG_LEVEL: LogLevel = LogLevel.DEBUG
    SOCKET_PATH: str = "/tmp/eheed.sock"
    AGENT_BACKEND: AgentBackend | None = None

    # Paths — CONFIG_PATH env var overrides the XDG/etc/dev resolution chain
    CONFIG_PATH: Path | None = None
    MODELS_DIR: Path = _BASE_DIR / "models"
    DATA_DIR: Path = _BASE_DIR / "data"

    BASE_DIR: ClassVar[Path] = _BASE_DIR
    XDG_CONFIG_DIR: ClassVar[Path] = _XDG_CONFIG

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        if self.CONFIG_PATH is None:
            self.CONFIG_PATH = _resolve_config_path()
        return self

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT is Environment.DEV


settings = Settings()
