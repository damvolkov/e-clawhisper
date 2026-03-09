"""Tests for settings and config models."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from e_clawhisper.shared.settings import (
    AgentBackend,
    AppConfig,
    Settings,
    STTBackend,
    TTSBackend,
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
    ],
    ids=["agent_openfang", "agent_generic", "stt", "tts_piper", "tts_kokoro"],
)
def test_enum_values(enum_cls: type, member: object, value: str) -> None:
    assert member == value


##### SETTINGS #####


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.ENVIRONMENT == "DEV"
    assert s.SOCKET_PATH == "/tmp/e-claw.sock"
    assert s.is_dev is True


##### APPCONFIG DEFAULTS #####


def test_app_config_defaults() -> None:
    cfg = AppConfig()
    assert cfg.agent.name == "damien"
    assert cfg.agent.backend == AgentBackend.OPENFANG
    assert cfg.stt.backend == STTBackend.WHISPERLIVE
    assert cfg.tts.backend == TTSBackend.KOKORO
    assert cfg.vad.threshold == 0.5
    assert cfg.vad.min_recording_time == 1.0
    assert cfg.audio.sample_rate == 16000


def test_app_config_sentinel_defaults() -> None:
    cfg = AppConfig()
    assert cfg.sentinel.energy_floor == 0.01
    assert cfg.sentinel.vad_threshold == 0.5
    assert cfg.sentinel.wakeword.model == "alexa"
    assert cfg.sentinel.wakeword.threshold == 0.5


##### YAML LOADING #####


def test_app_config_from_yaml() -> None:
    yaml_content = """
agent:
  name: testbot
  backend: openfang
stt:
  backend: whisperlive
  whisperlive:
    port: 9999
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
    assert cfg.stt.whisperlive.port == 9999
    assert cfg.tts.backend == TTSBackend.KOKORO
    assert cfg.sentinel.energy_floor == 0.02
    assert cfg.sentinel.wakeword.model == "hey_jarvis"
    assert cfg.sentinel.wakeword.threshold == 0.7


def test_app_config_generic_backend_defaults() -> None:
    cfg = AppConfig()
    assert cfg.backends.generic.host == "localhost"
    assert cfg.backends.generic.port == 45100
    assert cfg.backends.generic.timeout == 60.0


def test_app_config_kokoro_defaults() -> None:
    cfg = AppConfig()
    assert cfg.tts.kokoro.host == "localhost"
    assert cfg.tts.kokoro.port == 45130
    assert cfg.tts.kokoro.voice == "em_alex"
    assert cfg.tts.kokoro.sample_rate == 24000
    assert cfg.tts.kokoro.response_format == "pcm"


def test_tts_sample_rate_kokoro_default() -> None:
    cfg = AppConfig()
    assert cfg.tts_sample_rate == 24000


def test_tts_sample_rate_piper_explicit() -> None:
    cfg = AppConfig(tts={"backend": "piper"})
    assert cfg.tts_sample_rate == 22050


def test_app_config_from_yaml_generic_agent() -> None:
    yaml_content = """
agent:
  name: test
  backend: generic
backends:
  generic:
    host: 10.0.0.1
    port: 8080
    model: qwen2
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
    assert cfg.backends.generic.host == "10.0.0.1"
    assert cfg.backends.generic.port == 8080
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
