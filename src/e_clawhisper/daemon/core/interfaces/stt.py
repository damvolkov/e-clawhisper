"""STT interface — streaming speech-to-text contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class STTBase(ABC):
    """Abstract streaming STT backend."""

    @abstractmethod
    async def connect(self) -> None:
        """Initial connection (warm up model)."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Full teardown."""

    @abstractmethod
    async def start_utterance(self) -> None:
        """Prepare for a new utterance (reconnect session if needed)."""

    @abstractmethod
    async def stream_audio(self, audio_chunk: bytes) -> None:
        """Send raw float32 audio bytes."""

    @abstractmethod
    async def finish_utterance(self) -> str:
        """Signal end-of-audio and return final transcript."""

    @abstractmethod
    def transcript_stream(self) -> AsyncIterator[str]:
        """Yield partial transcripts as they arrive."""
