"""Extended tests for PipelineLogger — throttle, methods, configure_logging."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from e_clawhisper.shared.logger import PipelineLogger, PipelineRenderer, configure_logging


##### RENDERER COLOR DISPATCH #####


def test_renderer_warning_color() -> None:
    renderer = PipelineRenderer()
    color = renderer._resolve_color("TURN", "STT", "warning")
    assert color == "255;165;0"


def test_renderer_turn_colors() -> None:
    renderer = PipelineRenderer()
    assert renderer._resolve_color("TURN", "STT", "info") == "144;238;144"
    assert renderer._resolve_color("TURN", "AGENT", "info") == "50;205;50"
    assert renderer._resolve_color("TURN", "TTS", "info") == "0;255;127"
    assert renderer._resolve_color("TURN", "VAD", "info") == "46;139;87"


def test_renderer_loop_colors() -> None:
    renderer = PipelineRenderer()
    assert renderer._resolve_color("LOOP", "SILENCE", "info") == "139;119;42"
    assert renderer._resolve_color("LOOP", "VOICE", "info") == "255;215;0"


def test_renderer_system_colors() -> None:
    renderer = PipelineRenderer()
    assert renderer._resolve_color("SYSTEM", "START", "info") == "255;215;0"
    assert renderer._resolve_color("SYSTEM", "STOP", "info") == "255;165;0"
    assert renderer._resolve_color("SYSTEM", "IPC", "info") == "147;112;219"


def test_renderer_unknown_pipeline_uses_system_default() -> None:
    renderer = PipelineRenderer()
    color = renderer._resolve_color("UNKNOWN", "UNKNOWN", "info")
    assert color == "192;192;192"


def test_renderer_default_turn() -> None:
    renderer = PipelineRenderer()
    assert renderer._resolve_color("TURN", "UNKNOWN", "info") == "60;179;113"


def test_renderer_default_sentinel() -> None:
    renderer = PipelineRenderer()
    assert renderer._resolve_color("SENTINEL", "UNKNOWN", "info") == "70;130;180"


def test_renderer_default_loop() -> None:
    renderer = PipelineRenderer()
    assert renderer._resolve_color("LOOP", "UNKNOWN", "info") == "218;165;32"


##### RENDERER FORMAT #####


def test_renderer_no_step() -> None:
    renderer = PipelineRenderer()
    result = renderer(None, "info", {"event": "msg", "pipeline": "SYSTEM", "step": ""})
    assert "[SYSTEM]" in result
    assert "msg" in result


def test_renderer_no_event() -> None:
    renderer = PipelineRenderer()
    result = renderer(None, "info", {"event": "", "pipeline": "SYSTEM", "step": "OK"})
    assert "[SYSTEM]" in result
    assert "[OK]" in result


def test_renderer_critical_uses_error_color() -> None:
    renderer = PipelineRenderer()
    color = renderer._resolve_color("ANY", "ANY", "critical")
    assert color == "255;69;0"


##### LOGGER METHODS #####


def test_sentinel_logs() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.sentinel("VOICE", "detected")
    log._log.info.assert_called_once_with("detected", pipeline="SENTINEL", step="VOICE")


def test_turn_logs() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.turn("STT", "listening")
    log._log.info.assert_called_once_with("listening", pipeline="TURN", step="STT")


def test_loop_logs() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.loop("SILENCE", "idle")
    log._log.info.assert_called_once_with("idle", pipeline="LOOP", step="SILENCE")


def test_system_logs() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.system("START", "booting")
    log._log.info.assert_called_once_with("booting", pipeline="SYSTEM", step="START")


def test_warning_logs_with_current_pipeline() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.set_pipeline("TURN")
    log.warning("something bad")
    log._log.warning.assert_called_once_with("something bad", pipeline="TURN")


def test_error_logs() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.error("crash")
    log._log.error.assert_called_once_with("crash", pipeline="SYSTEM")


##### THROTTLE #####


def test_sentinel_debug_throttle() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.configure_throttle(idle_interval=100.0, turn_interval=100.0)

    log.sentinel_debug("SILENCE", "a")
    log.sentinel_debug("SILENCE", "b")

    assert log._log.debug.call_count == 1


def test_sentinel_debug_different_step_not_throttled() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.configure_throttle(idle_interval=100.0, turn_interval=100.0)

    log.sentinel_debug("SILENCE", "a")
    log.sentinel_debug("VOICE", "b")

    assert log._log.debug.call_count == 2


def test_turn_debug_throttle() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.configure_throttle(idle_interval=100.0, turn_interval=100.0)

    log.turn_debug("VAD", "a")
    log.turn_debug("VAD", "b")

    assert log._log.debug.call_count == 1


def test_loop_debug_throttle() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.configure_throttle(idle_interval=100.0, turn_interval=100.0)

    log.loop_debug("SILENCE", "a")
    log.loop_debug("SILENCE", "b")

    assert log._log.debug.call_count == 1


def test_loop_debug_different_step_not_throttled() -> None:
    log = PipelineLogger("test")
    log._log = MagicMock()
    log.configure_throttle(idle_interval=100.0, turn_interval=100.0)

    log.loop_debug("SILENCE", "a")
    log.loop_debug("NOISE", "b")

    assert log._log.debug.call_count == 2


##### TRUNCATE #####


def test_truncate_custom_limit() -> None:
    assert PipelineLogger.truncate("abcdef", 3) == "abc..."


def test_truncate_exact_limit() -> None:
    assert PipelineLogger.truncate("abc", 3) == "abc"


##### CONFIGURE LOGGING #####


@patch("e_clawhisper.shared.logger.structlog.configure")
@patch("e_clawhisper.shared.logger.logging.basicConfig")
@patch("e_clawhisper.shared.logger.logging.getLogger")
def test_configure_logging(mock_get: MagicMock, mock_basic: MagicMock, mock_conf: MagicMock) -> None:
    configure_logging("debug", idle_interval=0.5, turn_interval=0.1)
    mock_conf.assert_called_once()
    mock_basic.assert_called_once()
