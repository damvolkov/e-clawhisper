"""Tests for settings and config models — URL types, validation, loading."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pydantic import ValidationError

from e_heed.shared.settings import (
    AgentBackend,
    AppConfig,
    AudioConfig,
    ConversationConfig,
    GenericLLMConfig,
    KokoroConfig,
    LLMProvider,
    LoggingConfig,
    OpenFangConfig,
    PiperConfig,
    Settings,
    STTBackend,
    TTSBackend,
    VADConfig,
    WhisperLiveConfig,
)

##### ENUMS #####


@pytest.mark.parametrize(
    ("enum_cls", "member", "value"),
    [
        (AgentBackend, AgentBackend.OPENFANG, "openfang"),
        (AgentBackend, AgentBackend.GENERIC, "generic"),
        (STTBackend, STTBackend.WHISPERLIVE, "whisperlive"),
        (TTSBackend, TTSBackend.PIPER, "piper"),
        (TTSBackend, TTSBackend.KOKORO, "kokoro"),
        (LLMProvider, LLMProvider.GEMINI, "gemini"),
        (LLMProvider, LLMProvider.VLLM, "vllm"),
    ],
    ids=["agent_openfang", "agent_generic", "stt", "tts_piper", "tts_kokoro", "llm_gemini", "llm_vllm"],
)
def test_enum_values(enum_cls: type, member: object, value: str) -> None:
    assert member == value


##### SETTINGS #####


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.ENVIRONMENT == "DEV"
    assert s.SOCKET_PATH == "/tmp/eheed.sock"
    assert s.is_dev is True


def test_settings_path_defaults_resolve() -> None:
    s = Settings(_env_file=None)
    assert s.CONFIG_PATH.name == "config.yaml"
    assert s.MODELS_DIR.name == "models"
    assert s.DATA_DIR.name == "data"


def test_settings_path_env_override() -> None:
    s = Settings(_env_file=None, CONFIG_PATH=Path("/etc/test/config.yaml"))
    assert Path("/etc/test/config.yaml") == s.CONFIG_PATH


##### APPCONFIG DEFAULTS #####


def test_app_config_defaults() -> None:
    cfg = AppConfig()
    assert cfg.agent.name == "damien"
    assert cfg.agent.backend == AgentBackend.OPENFANG
    assert cfg.stt.backend == STTBackend.EVOICE
    assert cfg.tts.backend == TTSBackend.EVOICE
    assert cfg.vad.threshold == 0.5
    assert cfg.vad.min_recording_time == 1.0
    assert cfg.audio.sample_rate == 16000


def test_app_config_sentinel_defaults() -> None:
    cfg = AppConfig()
    assert cfg.sentinel.energy_floor == 0.01
    assert cfg.sentinel.vad_threshold == 0.5
    assert cfg.sentinel.wakeword.model == "alexa"
    assert cfg.sentinel.wakeword.threshold == 0.5


##### URL TYPES #####


def test_whisperlive_url_default() -> None:
    cfg = WhisperLiveConfig()
    assert "ws://localhost:45120" in str(cfg.url)


def test_kokoro_url_default() -> None:
    cfg = KokoroConfig()
    assert "localhost:45130" in str(cfg.url)


def test_piper_url_default_host_port() -> None:
    cfg = PiperConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 45130


def test_piper_url_custom() -> None:
    cfg = PiperConfig(url="tcp://10.0.0.5:9999")
    assert cfg.host == "10.0.0.5"
    assert cfg.port == 9999


def test_openfang_ws_base() -> None:
    cfg = OpenFangConfig()
    assert cfg.ws_base.startswith("ws://")
    assert "4200" in cfg.ws_base


def test_openfang_ws_base_custom() -> None:
    cfg = OpenFangConfig(url="http://10.0.0.1:5000")
    assert cfg.ws_base == "ws://10.0.0.1:5000"


def test_generic_llm_url_default() -> None:
    cfg = GenericLLMConfig()
    assert "localhost:45100" in str(cfg.url)


##### URL VALIDATION #####


def test_whisperlive_rejects_http_url() -> None:
    with pytest.raises(ValidationError, match="url_scheme"):
        WhisperLiveConfig(url="http://localhost:45120")


def test_piper_rejects_http_url() -> None:
    with pytest.raises(ValidationError, match="url_scheme"):
        PiperConfig(url="http://localhost:45130")


def test_kokoro_rejects_ws_url() -> None:
    with pytest.raises(ValidationError):
        KokoroConfig(url="ws://localhost:45130")


def test_openfang_rejects_ws_url() -> None:
    with pytest.raises(ValidationError):
        OpenFangConfig(url="ws://localhost:4200")


##### EXTRA FIELDS REJECTED #####


def test_extra_field_rejected_on_audio() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AudioConfig(nonexistent_field=42)


def test_extra_field_rejected_on_vad() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VADConfig(fake=True)


def test_extra_field_rejected_on_app_config() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AppConfig(unknown_key="oops")


##### PROBABILITY BOUNDS #####


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, 2.0],
    ids=["negative", "above_1", "far_above"],
)
def test_vad_threshold_rejects_out_of_range(value: float) -> None:
    with pytest.raises(ValidationError):
        VADConfig(threshold=value)


def test_vad_threshold_accepts_boundary_values() -> None:
    cfg_zero = VADConfig(threshold=0.0)
    cfg_one = VADConfig(threshold=1.0)
    assert cfg_zero.threshold == 0.0
    assert cfg_one.threshold == 1.0


##### POSITIVE SECONDS #####


def test_positive_seconds_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        VADConfig(silence_duration=0.0)


def test_positive_seconds_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        VADConfig(silence_duration=-1.0)


##### POSITIVE INT #####


def test_positive_int_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        AudioConfig(sample_rate=0)


def test_positive_int_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        ConversationConfig(max_turns=-1)


##### LITERAL VALIDATION #####


def test_audio_playback_latency_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        AudioConfig(playback_latency="ultra")


def test_kokoro_response_format_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        KokoroConfig(response_format="aac")


##### TTS SAMPLE RATE #####


def test_tts_sample_rate_kokoro_default() -> None:
    cfg = AppConfig()
    assert cfg.tts_sample_rate == 24000


def test_tts_sample_rate_piper_explicit() -> None:
    cfg = AppConfig(tts={"backend": "piper"})
    assert cfg.tts_sample_rate == 22050


##### YAML LOADING #####


def test_app_config_from_yaml_url_format() -> None:
    yaml_content = """
agent:
  name: testbot
  backend: openfang
stt:
  backend: whisperlive
  whisperlive:
    url: ws://10.0.0.1:9999
sentinel:
  energy_floor: 0.02
  wakeword:
    model: hey_jarvis
    threshold: 0.7
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = AppConfig.from_yaml(Path(f.name))

    assert cfg.agent.name == "testbot"
    assert "10.0.0.1:9999" in str(cfg.stt.whisperlive.url)
    assert cfg.tts.backend == TTSBackend.EVOICE
    assert cfg.sentinel.energy_floor == 0.02
    assert cfg.sentinel.wakeword.model == "hey_jarvis"
    assert cfg.sentinel.wakeword.threshold == 0.7


def test_app_config_generic_backend_defaults() -> None:
    cfg = AppConfig()
    assert cfg.backends.generic.provider == LLMProvider.GEMINI
    assert cfg.backends.generic.model == "gemini-2.0-flash"
    assert cfg.backends.generic.timeout == 60.0


def test_app_config_from_yaml_generic_agent() -> None:
    yaml_content = """
agent:
  name: test
  backend: generic
backends:
  generic:
    provider: vllm
    model: qwen2
    url: http://10.0.0.1:8080
tts:
  backend: kokoro
  kokoro:
    voice: es_es/paloma
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = AppConfig.from_yaml(Path(f.name))

    assert cfg.agent.backend == AgentBackend.GENERIC
    assert cfg.backends.generic.provider == LLMProvider.VLLM
    assert "10.0.0.1:8080" in str(cfg.backends.generic.url)
    assert cfg.backends.generic.model == "qwen2"
    assert cfg.tts.backend == TTSBackend.KOKORO
    assert cfg.tts.kokoro.voice == "es_es/paloma"


def test_app_config_language_cascades_stt() -> None:
    yaml_content = """
language: es
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = AppConfig.from_yaml(Path(f.name))

    assert cfg.stt.whisperlive.language == "es"


def test_app_config_language_cascades_piper_voice() -> None:
    yaml_content = """
language: es
tts:
  piper:
    voice: en_US-lessac-medium
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = AppConfig.from_yaml(Path(f.name))

    assert cfg.tts.piper.voice == "es_ES-sharvard-medium"


##### YAML REJECTS OLD HOST+PORT FORMAT #####


def test_yaml_rejects_old_host_port_format() -> None:
    yaml_content = """
stt:
  whisperlive:
    host: localhost
    port: 45120
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AppConfig.from_yaml(Path(f.name))


##### YAML REJECTS INVALID BACKEND #####


def test_yaml_rejects_unknown_backend() -> None:
    yaml_content = """
agent:
  backend: openai_native
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        with pytest.raises(ValidationError):
            AppConfig.from_yaml(Path(f.name))


##### LOGGING CONFIG #####


def test_logging_config_defaults() -> None:
    cfg = LoggingConfig()
    assert cfg.idle_interval == 0.25
    assert cfg.turn_interval == 0.25
