"""TTS adapter base — abstract interface for text-to-speech backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class TTSAdapter(ABC):
    """Abstract TTS adapter — synthesize text to PCM audio stream."""

    __slots__ = ()

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @abstractmethod
    async def synthesize(self, text: str) -> AsyncIterator[bytes]: ...

    @abstractmethod
    async def stop(self) -> None: ...
