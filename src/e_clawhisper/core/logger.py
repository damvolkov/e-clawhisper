"""Structured logging with icon prefixes."""

from __future__ import annotations

import logging
from enum import StrEnum, auto

import structlog


class LogIcon(StrEnum):
    """Log icon identifiers."""

    START = auto()
    STOP = auto()
    AGENT = auto()
    CHAT = auto()
    ERROR = auto()
    WAKE = auto()
    STT = auto()
    TTS = auto()
    CHANNEL = auto()
    PROCESSING = auto()
    COMPLETE = auto()


_ICONS: dict[LogIcon, str] = {
    LogIcon.START: "[>>]",
    LogIcon.STOP: "[||]",
    LogIcon.AGENT: "[AG]",
    LogIcon.CHAT: "[CH]",
    LogIcon.ERROR: "[!!]",
    LogIcon.WAKE: "[WK]",
    LogIcon.STT: "[ST]",
    LogIcon.TTS: "[TS]",
    LogIcon.CHANNEL: "[CN]",
    LogIcon.PROCESSING: "[..]",
    LogIcon.COMPLETE: "[OK]",
}


class IconLogger:
    """Logger wrapper with icon-based prefixes."""

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
    """Configure structlog for dev/prod."""
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
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(message)s")


logger = IconLogger("e_clawhisper")
