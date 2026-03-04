"""Base channel ABC for agent backend communication."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseChannel(ABC):
    """Abstract voice channel to an AgentOS backend.

    Analogous to a Telegram/Discord channel — a bidirectional
    bridge between a voice pipeline and a remote agent.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the backend."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down connection."""

    @abstractmethod
    async def send_message(self, message: str) -> str:
        """Send user message, return agent response text."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check whether the channel is active."""
