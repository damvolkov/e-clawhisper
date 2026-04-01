"""Agent adapter base — abstract streaming interface for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class AgentAdapter(ABC):
    """Abstract agent adapter — connect, stream, disconnect."""

    __slots__ = ()

    @property
    @abstractmethod
    def agent_id(self) -> str: ...

    @abstractmethod
    async def connect(self, agent_id: str) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def is_connected(self) -> bool: ...

    @abstractmethod
    async def send(self, text: str) -> AsyncIterator[str]: ...

    @abstractmethod
    async def resolve_agent_id(self, agent_name: str) -> str: ...
