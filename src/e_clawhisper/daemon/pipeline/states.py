"""Pipeline state machine."""

from __future__ import annotations

from enum import StrEnum, auto


class PipelineState(StrEnum):
    """Audio pipeline processing state."""

    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class ConversationMode(StrEnum):
    """Whether the agent is activated."""

    IDLE = auto()
    ACTIVE = auto()
