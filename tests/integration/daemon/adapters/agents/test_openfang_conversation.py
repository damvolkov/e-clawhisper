"""Integration test — OpenFang adapter real WebSocket conversation.

Requires OpenFang running at 127.0.0.1:4200 with agent 'damien'.
Run with: uv run pytest tests/integration/ -v -s
"""

from __future__ import annotations

import asyncio

import pytest

from e_clawhisper.daemon.adapters.agent import AgentAdapter
from e_clawhisper.shared.settings import OpenFangConfig

_CONFIG = OpenFangConfig(host="127.0.0.1", port=4200, timeout=30.0)
_AGENT_NAME = "damien"

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("httpx", reason="httpx required"),
    reason="integration",
)


##### HELPERS #####


async def _check_openfang_available() -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:4200", timeout=3.0) as client:
            resp = await client.get("/api/agents")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


##### RESOLVE #####


async def test_resolve_agent_id() -> None:
    if not await _check_openfang_available():
        pytest.skip("OpenFang not available at 127.0.0.1:4200")

    adapter = AgentAdapter(_CONFIG)
    agent_id = await adapter.resolve_agent_id(_AGENT_NAME)

    assert agent_id
    assert len(agent_id) > 10
    print(f"\n  resolved: {_AGENT_NAME} -> {agent_id}")


##### CONNECT + DISCONNECT #####


async def test_connect_and_disconnect() -> None:
    if not await _check_openfang_available():
        pytest.skip("OpenFang not available at 127.0.0.1:4200")

    adapter = AgentAdapter(_CONFIG)
    agent_id = await adapter.resolve_agent_id(_AGENT_NAME)
    await adapter.connect(agent_id)

    assert await adapter.is_connected()
    print(f"\n  connected to {agent_id[:12]}")

    await asyncio.sleep(1)
    assert await adapter.is_connected()

    await adapter.disconnect()
    assert not await adapter.is_connected()
    print("  disconnected cleanly")


##### SEND MESSAGE — STREAMING RESPONSE #####


async def test_send_message_streaming() -> None:
    if not await _check_openfang_available():
        pytest.skip("OpenFang not available at 127.0.0.1:4200")

    adapter = AgentAdapter(_CONFIG)
    agent_id = await adapter.resolve_agent_id(_AGENT_NAME)
    await adapter.connect(agent_id)

    chunks: list[str] = []
    print("\n  sending: 'Hello, say hi back in one word'")

    async for chunk in adapter.send("Hello, say hi back in one word"):
        chunks.append(chunk)
        print(f"  chunk: {chunk!r}")

    full_response = "".join(chunks)
    print(f"  full response: {full_response!r}")

    assert len(chunks) > 0, "Expected at least one response chunk"
    assert len(full_response) > 0, "Expected non-empty response"

    await adapter.disconnect()


##### RECV LOOP SURVIVES IDLE #####


async def test_recv_loop_survives_idle_period() -> None:
    """Verify _receive_loop stays alive after 5s of no messages."""
    if not await _check_openfang_available():
        pytest.skip("OpenFang not available at 127.0.0.1:4200")

    adapter = AgentAdapter(_CONFIG)
    agent_id = await adapter.resolve_agent_id(_AGENT_NAME)
    await adapter.connect(agent_id)

    print("\n  connected, waiting 5s idle...")
    await asyncio.sleep(5)

    assert await adapter.is_connected()
    assert adapter._recv_task is not None
    assert not adapter._recv_task.done(), "recv_task died during idle!"

    chunks: list[str] = []
    async for chunk in adapter.send("Say 'test' and nothing else"):
        chunks.append(chunk)

    full = "".join(chunks)
    print(f"  response after idle: {full!r}")
    assert len(full) > 0, "No response after idle period"

    await adapter.disconnect()


##### MULTIPLE MESSAGES IN SEQUENCE #####


async def test_multiple_messages_sequential() -> None:
    if not await _check_openfang_available():
        pytest.skip("OpenFang not available at 127.0.0.1:4200")

    adapter = AgentAdapter(_CONFIG)
    agent_id = await adapter.resolve_agent_id(_AGENT_NAME)
    await adapter.connect(agent_id)

    for i, msg in enumerate(["Say 'one'", "Say 'two'", "Say 'three'"]):
        chunks: list[str] = []
        async for chunk in adapter.send(msg):
            chunks.append(chunk)
        full = "".join(chunks)
        print(f"\n  msg {i + 1}: {msg!r} -> {full!r}")
        assert len(full) > 0, f"Empty response for message {i + 1}"

    await adapter.disconnect()
