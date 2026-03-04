"""Unit test fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from e_clawhisper.channels.base import BaseChannel


@pytest.fixture
def mock_channel() -> AsyncMock:
    """Fake channel returning canned responses."""
    channel = AsyncMock(spec=BaseChannel)
    channel.send_message.return_value = "Hello! I can help with that."
    channel.is_connected.return_value = True
    return channel
