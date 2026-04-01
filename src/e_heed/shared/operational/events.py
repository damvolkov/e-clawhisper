"""Inter-pipeline events — data contracts between Sentinel and Turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class WakewordEvent:
    """Sentinel → Orchestrator: wake word detected."""

    timestamp: float
    confidence: float
    energy: float


@dataclass(frozen=True, slots=True)
class TurnComplete:
    """Turn → Orchestrator: turn finished successfully."""

    turn_id: str
    transcript: str
    response: str
    duration: float


@dataclass(frozen=True, slots=True)
class TurnError:
    """Turn → Orchestrator: turn failed."""

    turn_id: str
    reason: str


@dataclass(slots=True)
class Turn:
    """Mutable turn data accumulated during a single STT→Agent→TTS cycle."""

    turn_id: str = field(default_factory=lambda: uuid4().hex[:12])
    transcript: str = ""
    response: str = ""
    started_at: float = field(default_factory=monotonic)

    @property
    def duration(self) -> float:
        return monotonic() - self.started_at

    def to_complete(self) -> TurnComplete:
        return TurnComplete(
            turn_id=self.turn_id,
            transcript=self.transcript,
            response=self.response,
            duration=self.duration,
        )

    def to_error(self, reason: str) -> TurnError:
        return TurnError(turn_id=self.turn_id, reason=reason)
