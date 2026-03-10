"""Pipeline-aware structured logger — ANSI colored, stderr output."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Final

import structlog

##### ANSI COLORS (RGB) #####

_RESET: Final[str] = "\033[0m"

_SENTINEL_COLORS: dict[str, str] = {
    "SILENCE": "25;25;112",
    "NOISE": "100;149;237",
    "VOICE": "0;206;209",
    "WAKEWORD": "34;139;34",
    "DEFAULT": "70;130;180",
}

_TURN_COLORS: dict[str, str] = {
    "STT": "144;238;144",
    "AGENT": "50;205;50",
    "TTS": "0;255;127",
    "VAD": "46;139;87",
    "DEFAULT": "60;179;113",
}

_LOOP_COLORS: dict[str, str] = {
    "SILENCE": "139;119;42",
    "NOISE": "184;134;11",
    "VOICE": "255;215;0",
    "DEFAULT": "218;165;32",
}

_SYSTEM_COLORS: dict[str, str] = {
    "START": "255;215;0",
    "STOP": "255;165;0",
    "ERROR": "255;69;0",
    "WARN": "255;165;0",
    "IPC": "147;112;219",
    "OK": "50;205;50",
    "DEFAULT": "192;192;192",
}

_MAX_TEXT_LEN: Final[int] = 120


##### RENDERER #####


class PipelineRenderer:
    """Format: HH:MM:SS [PIPELINE] [STEP] message key=value."""

    __slots__ = ()

    def __call__(self, logger_instance: Any, method_name: str, event_dict: dict[str, Any]) -> str:
        pipeline = str(event_dict.pop("pipeline", "SYSTEM")).upper()
        step = str(event_dict.pop("step", "")).upper()
        event = event_dict.pop("event", "")

        color = self._resolve_color(pipeline, step, method_name)
        ts = time.strftime("%H:%M:%S")

        # Build key=value extras
        skip = {"timestamp", "level", "logger"}
        extras = " ".join(f"{k}={v}" for k, v in event_dict.items() if k not in skip and v is not None)

        parts = [f"{ts} [{pipeline}]"]
        if step:
            parts.append(f"[{step}]")
        if event:
            parts.append(str(event))
        if extras:
            parts.append(extras)

        line = " ".join(parts)
        return f"\033[38;2;{color}m{line}{_RESET}"

    @staticmethod
    def _resolve_color(pipeline: str, step: str, level: str) -> str:
        if level in ("error", "critical"):
            return _SYSTEM_COLORS["ERROR"]
        if level == "warning":
            return _SYSTEM_COLORS["WARN"]

        match pipeline:
            case "SENTINEL":
                return _SENTINEL_COLORS.get(step, _SENTINEL_COLORS["DEFAULT"])
            case "TURN":
                return _TURN_COLORS.get(step, _TURN_COLORS["DEFAULT"])
            case "LOOP":
                return _LOOP_COLORS.get(step, _LOOP_COLORS["DEFAULT"])
            case _:
                return _SYSTEM_COLORS.get(step, _SYSTEM_COLORS["DEFAULT"])


##### LOGGER #####


class PipelineLogger:
    """Pipeline-aware logger with throttle support."""

    __slots__ = ("_log", "_idle_interval", "_turn_interval", "_last_debug", "_last_step", "_pipeline")

    def __init__(self, name: str) -> None:
        self._log = structlog.get_logger(name)
        self._idle_interval: float = 0.25
        self._turn_interval: float = 0.25
        self._last_debug: float = 0.0
        self._last_step: str = ""
        self._pipeline: str = "SYSTEM"

    def configure_throttle(self, *, idle_interval: float, turn_interval: float) -> None:
        self._idle_interval = idle_interval
        self._turn_interval = turn_interval

    def set_pipeline(self, pipeline: str) -> None:
        self._pipeline = pipeline

    @staticmethod
    def truncate(text: str, limit: int = _MAX_TEXT_LEN) -> str:
        return f"{text[:limit]}..." if len(text) > limit else text

    ##### SENTINEL #####

    def sentinel(self, step: str, msg: str = "", **kw: Any) -> None:
        self._log.info(msg, pipeline="SENTINEL", step=step, **kw)

    def sentinel_debug(self, step: str, msg: str = "", **kw: Any) -> None:
        now = time.monotonic()
        if step == self._last_step and now - self._last_debug < self._idle_interval:
            return
        self._last_debug = now
        self._last_step = step
        self._log.debug(msg, pipeline="SENTINEL", step=step, **kw)

    ##### TURN #####

    def turn(self, step: str, msg: str = "", **kw: Any) -> None:
        self._log.info(msg, pipeline="TURN", step=step, **kw)

    def turn_debug(self, step: str, msg: str = "", **kw: Any) -> None:
        now = time.monotonic()
        if now - self._last_debug < self._turn_interval:
            return
        self._last_debug = now
        self._log.debug(msg, pipeline="TURN", step=step, **kw)

    ##### LOOP #####

    def loop(self, step: str, msg: str = "", **kw: Any) -> None:
        self._log.info(msg, pipeline="LOOP", step=step, **kw)

    def loop_debug(self, step: str, msg: str = "", **kw: Any) -> None:
        now = time.monotonic()
        if step == self._last_step and now - self._last_debug < self._idle_interval:
            return
        self._last_debug = now
        self._last_step = step
        self._log.debug(msg, pipeline="LOOP", step=step, **kw)

    ##### SYSTEM #####

    def system(self, step: str, msg: str = "", **kw: Any) -> None:
        self._log.info(msg, pipeline="SYSTEM", step=step, **kw)

    def warning(self, msg: str, **kw: Any) -> None:
        self._log.warning(msg, pipeline=self._pipeline, **kw)

    def error(self, msg: str, **kw: Any) -> None:
        self._log.error(msg, pipeline=self._pipeline, **kw)


##### CONFIGURE #####


def configure_logging(
    level: str = "info",
    *,
    idle_interval: float = 0.25,
    turn_interval: float = 0.25,
) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.UnicodeDecoder(),
            PipelineRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")

    for noisy in ("websockets", "httpx", "httpcore", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.configure_throttle(idle_interval=idle_interval, turn_interval=turn_interval)


logger = PipelineLogger("eclaw")
