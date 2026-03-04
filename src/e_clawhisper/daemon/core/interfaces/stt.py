"""STT interface — streaming speech-to-text contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class STTBase(ABC):
    """Abstract streaming STT backend."""

    @abstractmethod
    async def connect(self) -> None:
        """Open connection to STT service."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""

    @abstractmethod
    async def stream_audio(self, audio_chunk: bytes) -> None:
        """Send a raw float32 audio chunk to the STT stream."""

    @abstractmethod
    async def finish_stream(self) -> str:
        """Signal end-of-audio and return final transcript."""

    @abstractmethod
    def transcript_stream(self) -> AsyncIterator[str]:
        """Yield partial transcripts as they arrive."""
