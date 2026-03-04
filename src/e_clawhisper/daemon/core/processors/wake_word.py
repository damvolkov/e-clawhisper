"""Wake-word detector — activates conversation when agent name is spoken."""

from __future__ import annotations

from e_clawhisper.shared.logger import LogIcon, logger


class WakeWordDetector:
    """Simple substring-based wake-word detection.

    Checks if the configured agent name appears in STT transcripts.
    Future: replace with dedicated keyword spotting model.
    """

    __slots__ = ("_wake_word",)

    def __init__(self, wake_word: str) -> None:
        self._wake_word = wake_word.lower().strip()

    @property
    def wake_word(self) -> str:
        return self._wake_word

    def check(self, transcript: str) -> bool:
        found = self._wake_word in transcript.lower()
        if found:
            logger.info("wake_word_detected word=%s", self._wake_word, icon=LogIcon.WAKE)
        return found

    def strip(self, transcript: str) -> str:
        """Remove wake word from transcript and return remaining query."""
        lower = transcript.lower()
        idx = lower.find(self._wake_word)
        if idx == -1:
            return transcript
        before = transcript[:idx].rstrip(" ,!.")
        after = transcript[idx + len(self._wake_word) :].lstrip(" ,!.")
        return f"{before} {after}".strip()
