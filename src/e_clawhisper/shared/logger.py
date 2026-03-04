"""Structured logging."""

from __future__ import annotations

import logging
from enum import StrEnum, auto

import structlog


class LogIcon(StrEnum):
    START = auto()
    STOP = auto()
    AGENT = auto()
    WAKE = auto()
    STT = auto()
    TTS = auto()
    VAD = auto()
    PIPE = auto()
    IPC = auto()
    ERROR = auto()
    OK = auto()


_ICONS: dict[LogIcon, str] = {
    LogIcon.START: "[>>]",
    LogIcon.STOP: "[||]",
    LogIcon.AGENT: "[AG]",
    LogIcon.WAKE: "[WK]",
    LogIcon.STT: "[ST]",
    LogIcon.TTS: "[TS]",
    LogIcon.VAD: "[VD]",
    LogIcon.PIPE: "[PI]",
    LogIcon.IPC: "[IP]",
    LogIcon.ERROR: "[!!]",
    LogIcon.OK: "[OK]",
}


class IconLogger:
    """Logger with icon prefixes."""

    def __init__(self, name: str) -> None:
        self._log = structlog.get_logger(name)

    def _fmt(self, msg: str, icon: LogIcon | None) -> str:
        return f"{_ICONS[icon]} {msg}" if icon else msg

    def info(self, msg: str, *args: object, icon: LogIcon | None = None) -> None:
        self._log.info(self._fmt(msg, icon), *args)

    def debug(self, msg: str, *args: object, icon: LogIcon | None = None) -> None:
        self._log.debug(self._fmt(msg, icon), *args)

    def warning(self, msg: str, *args: object, icon: LogIcon | None = None) -> None:
        self._log.warning(self._fmt(msg, icon), *args)

    def error(self, msg: str, *args: object, icon: LogIcon | None = None) -> None:
        self._log.error(self._fmt(msg, icon), *args)


def configure_logging(level: str = "info") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")


logger = IconLogger("eclaw")
