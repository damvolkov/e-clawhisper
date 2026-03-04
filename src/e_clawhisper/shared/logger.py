"""Structured logging."""

from __future__ import annotations

import logging
from enum import StrEnum, auto

import structlog

##### ENUMS #####


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


##### ICONS #####

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


##### LOGGER #####


class IconLogger:
    """Logger with icon prefixes."""

    __slots__ = ("_log",)

    def __init__(self, name: str) -> None:
        self._log = structlog.get_logger(name)

    @staticmethod
    def format_icon(msg: str, icon: LogIcon | None) -> str:
        return f"{_ICONS[icon]} {msg}" if icon else msg

    def info(self, msg: str, *args: object, icon: LogIcon | None = None) -> None:
        self._log.info(self.format_icon(msg, icon), *args)

    def debug(self, msg: str, *args: object, icon: LogIcon | None = None) -> None:
        self._log.debug(self.format_icon(msg, icon), *args)

    def warning(self, msg: str, *args: object, icon: LogIcon | None = None) -> None:
        self._log.warning(self.format_icon(msg, icon), *args)

    def error(self, msg: str, *args: object, icon: LogIcon | None = None) -> None:
        self._log.error(self.format_icon(msg, icon), *args)


##### CONFIGURE #####


def configure_logging(level: str = "info") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")

    for noisy in ("websockets", "httpx", "httpcore", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = IconLogger("eclaw")
