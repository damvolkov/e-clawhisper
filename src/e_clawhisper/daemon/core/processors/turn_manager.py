"""Turn manager — conversation mode, barge-in, and idle timeout."""

from __future__ import annotations

import time

from e_clawhisper.daemon.pipeline.states import ConversationMode, PipelineState
from e_clawhisper.shared.logger import LogIcon, logger


class TurnManager:
    """Manages conversation activation, barge-in, and idle timeout.

    Barge-in: if VAD detects speech while pipeline is SPEAKING,
    TTS output is interrupted and pipeline returns to STREAMING.
    """

    __slots__ = ("_state", "_mode", "_last_activity", "_conversation_timeout")

    def __init__(self, conversation_timeout: float = 30.0) -> None:
        self._state = PipelineState.IDLE
        self._mode = ConversationMode.IDLE
        self._last_activity: float = 0.0
        self._conversation_timeout = conversation_timeout

    @property
    def state(self) -> PipelineState:
        return self._state

    @state.setter
    def state(self, value: PipelineState) -> None:
        if value != self._state:
            logger.debug("state: %s -> %s", self._state, value, icon=LogIcon.PIPE)
            self._state = value

    @property
    def mode(self) -> ConversationMode:
        return self._mode

    @property
    def is_active(self) -> bool:
        return self._mode == ConversationMode.ACTIVE

    def activate(self) -> None:
        self._mode = ConversationMode.ACTIVE
        self._last_activity = time.monotonic()
        logger.info("conversation_activated", icon=LogIcon.AGENT)

    def deactivate(self) -> None:
        self._mode = ConversationMode.IDLE
        self._state = PipelineState.IDLE
        self._last_activity = 0.0
        logger.info("conversation_deactivated", icon=LogIcon.STOP)

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    def check_timeout(self) -> bool:
        """Return True if conversation timed out and was deactivated."""
        if not self.is_active or self._last_activity == 0.0:
            return False
        elapsed = time.monotonic() - self._last_activity
        if elapsed > self._conversation_timeout:
            logger.info("conversation_timeout elapsed=%.1fs", elapsed, icon=LogIcon.STOP)
            self.deactivate()
            return True
        return False

    def should_barge_in(self, is_speech: bool) -> bool:
        """Check if user is speaking while TTS is playing."""
        if self._state == PipelineState.SPEAKING and is_speech:
            logger.info("barge_in_detected", icon=LogIcon.PIPE)
            self.state = PipelineState.STREAMING
            return True
        return False
