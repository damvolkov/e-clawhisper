"""Tests for settings and config models."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from e_clawhisper.shared.settings import (
    AgentBackend,
    AppConfig,
    STTBackend,
    Settings,
    TTSBackend,
    VADBackend,
)


##### ENUMS #####


@pytest.mark.parametrize(
    ("enum_cls", "member", "value"),
    [
        (AgentBackend, AgentBackend.OPENFANG, "openfang"),
        (STTBackend, STTBackend.WHISPERLIVE, "whisperlive"),
        (TTSBackend, TTSBackend.PIPER, "piper"),
        (VADBackend, VADBackend.TEN_VAD, "ten_vad"),
    ],
    ids=["agent", "stt", "tts", "vad"],
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
    assert cfg.audio.sample_rate == 16000


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
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = AppConfig.from_yaml(Path(f.name))

    assert cfg.agent.name == "testbot"
    assert cfg.stt.whisperlive.port == 9999
    assert cfg.tts.backend == TTSBackend.PIPER
