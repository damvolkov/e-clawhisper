"""TTS interface — text-to-speech contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class TTSBase(ABC):
    """Abstract TTS backend."""

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesize text, yield PCM audio chunks."""

    @abstractmethod
    async def stop(self) -> None:
        """Interrupt current synthesis."""
