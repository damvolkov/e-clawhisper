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
        (STTBackend, STTBackend.WHISPERLIVE, "whisperlive"),
        (TTSBackend, TTSBackend.PIPER, "piper"),
    ],
    ids=["agent", "stt", "tts"],
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
    assert cfg.tts.backend == TTSBackend.PIPER
    assert cfg.vad.threshold == 0.5
    assert cfg.vad.min_recording_time == 1.0
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.pre_roll_seconds == 2.0


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
    assert cfg.tts.backend == TTSBackend.PIPER
    assert cfg.sentinel.energy_floor == 0.02
    assert cfg.sentinel.wakeword.model == "hey_jarvis"
    assert cfg.sentinel.wakeword.threshold == 0.7


def test_app_config_language_cascades() -> None:
    yaml_content = """
language: es
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = AppConfig.from_yaml(Path(f.name))

    assert cfg.stt.whisperlive.language == "es"
    assert cfg.tts.piper.voice == "es_ES-sharvard-medium"
