"""Agent interface — AgentOS backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class AgentBase(ABC):
    """Abstract AgentOS channel (OpenFang, ZeroClaw, etc.)."""

    @abstractmethod
    async def connect(self, agent_id: str) -> None:
        """Open WebSocket to agent."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""

    @abstractmethod
    async def send_message(self, text: str) -> AsyncIterator[str]:
        """Send user text, yield streaming response chunks."""

    @abstractmethod
    async def resolve_agent_id(self, agent_name: str) -> str:
        """Resolve agent name to ID via REST API."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check connection state."""
