"""STT adapter base — abstract interface for speech-to-text backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class STTAdapter(ABC):
    """Abstract STT adapter — connect, stream audio, get transcript."""

    __slots__ = ()

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def start_utterance(self) -> None: ...

    @abstractmethod
    async def stream(self, audio_chunk: bytes) -> None: ...

    @abstractmethod
    async def finish_utterance(self) -> str: ...
