"""Tests for channel implementations."""

from __future__ import annotations

import pytest

from e_clawhisper.channels.openfang import OpenFangChannel


##### OPENFANG CHANNEL #####


async def test_openfang_channel_builds_message_url() -> None:
    channel = OpenFangChannel(
        base_url="http://localhost:4200",
        agent_id="abc-123",
    )
    assert channel.message_url == "/api/agents/abc-123/message"


async def test_openfang_channel_builds_ws_url() -> None:
    channel = OpenFangChannel(
        base_url="http://localhost:4200",
        agent_id="abc-123",
    )
    assert channel.ws_url == "ws://localhost:4200/api/agents/abc-123/ws"


async def test_openfang_channel_builds_ws_url_from_https() -> None:
    channel = OpenFangChannel(
        base_url="https://agent.example.com",
        agent_id="xyz-789",
    )
    assert channel.ws_url == "wss://agent.example.com/api/agents/xyz-789/ws"


async def test_openfang_channel_strips_trailing_slash() -> None:
    channel = OpenFangChannel(
        base_url="http://localhost:4200/",
        agent_id="abc-123",
    )
    assert channel.message_url == "/api/agents/abc-123/message"


async def test_openfang_channel_not_connected_initially() -> None:
    channel = OpenFangChannel(
        base_url="http://localhost:4200",
        agent_id="abc-123",
    )
    assert not await channel.is_connected()
