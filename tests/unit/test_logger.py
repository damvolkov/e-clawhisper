"""Tests for pipeline-aware logger."""

from __future__ import annotations

from e_heed.shared.logger import PipelineLogger, PipelineRenderer

##### RENDERER #####


def test_renderer_formats_pipeline_and_step() -> None:
    renderer = PipelineRenderer()
    event_dict = {"event": "test msg", "pipeline": "SENTINEL", "step": "VAD"}
    result = renderer(None, "info", event_dict)
    assert "[SENTINEL]" in result
    assert "[VAD]" in result
    assert "test msg" in result


def test_renderer_formats_extras() -> None:
    renderer = PipelineRenderer()
    event_dict = {"event": "check", "pipeline": "TURN", "step": "STT", "key": "val"}
    result = renderer(None, "info", event_dict)
    assert "key=val" in result


def test_renderer_error_color() -> None:
    renderer = PipelineRenderer()
    color = renderer._resolve_color("SYSTEM", "OK", "error")
    assert color == "255;69;0"


def test_renderer_sentinel_color() -> None:
    renderer = PipelineRenderer()
    color = renderer._resolve_color("SENTINEL", "WAKEWORD", "info")
    assert color == "34;139;34"


##### PIPELINE LOGGER #####


def test_logger_truncate_short() -> None:
    assert PipelineLogger.truncate("hello", 10) == "hello"


def test_logger_truncate_long() -> None:
    result = PipelineLogger.truncate("a" * 200, 10)
    assert result.endswith("...")
    assert len(result) == 13


def test_logger_set_pipeline() -> None:
    log = PipelineLogger("test")
    log.set_pipeline("TURN")
    assert log._pipeline == "TURN"


def test_logger_configure_throttle() -> None:
    log = PipelineLogger("test")
    log.configure_throttle(idle_interval=0.5, turn_interval=0.1)
    assert log._idle_interval == 0.5
    assert log._turn_interval == 0.1
