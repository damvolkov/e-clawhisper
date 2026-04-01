"""Tests for OpenfangAdapter — init, properties, ws_base derivation."""

from __future__ import annotations

from e_heed.daemon.adapters.agent.openfang import OpenfangAdapter
from e_heed.shared.settings import OpenFangConfig

##### INIT #####


def test_init_derives_base_url() -> None:
    cfg = OpenFangConfig(url="http://192.168.1.100:4200", timeout=15.0)
    adapter = OpenfangAdapter(cfg)
    assert adapter._base_url == "http://192.168.1.100:4200"
    assert adapter._ws_base == "ws://192.168.1.100:4200"


def test_init_https_to_wss() -> None:
    cfg = OpenFangConfig(url="https://example.com:4200")
    adapter = OpenfangAdapter(cfg)
    assert adapter._ws_base == "wss://example.com:4200"


def test_init_strips_trailing_slash() -> None:
    cfg = OpenFangConfig(url="http://localhost:4200/")
    adapter = OpenfangAdapter(cfg)
    assert not adapter._base_url.endswith("/")


def test_agent_id_default_empty() -> None:
    adapter = OpenfangAdapter(OpenFangConfig())
    assert adapter.agent_id == ""


async def test_is_connected_false_by_default() -> None:
    adapter = OpenfangAdapter(OpenFangConfig())
    assert not await adapter.is_connected()


async def test_disconnect_noop_without_connection() -> None:
    adapter = OpenfangAdapter(OpenFangConfig())
    await adapter.disconnect()
    assert not await adapter.is_connected()


##### CONNECT #####


async def test_connect_sets_agent_id_and_ws() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_ws = MagicMock()
    mock_ws.recv = AsyncMock(return_value=b'{"type":"connected"}')

    with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
        adapter = OpenfangAdapter(OpenFangConfig(url="http://localhost:4200"))
        await adapter.connect("agent-abc-123")

    assert adapter.agent_id == "agent-abc-123"
    assert adapter._ws is mock_ws
    assert adapter._recv_task is not None
    adapter._recv_task.cancel()


async def test_connect_builds_correct_ws_url() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_ws = MagicMock()
    mock_ws.recv = AsyncMock(return_value=b'{"type":"connected"}')
    captured: list[str] = []

    async def fake_connect(url: str) -> MagicMock:
        captured.append(url)
        return mock_ws

    with patch("websockets.connect", side_effect=fake_connect):
        adapter = OpenfangAdapter(OpenFangConfig(url="http://localhost:4200"))
        await adapter.connect("my-agent-id")

    assert captured[0] == "ws://localhost:4200/api/agents/my-agent-id/ws"
    adapter._recv_task.cancel()


async def test_connect_logs_warning_on_unexpected_message() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_ws = MagicMock()
    mock_ws.recv = AsyncMock(return_value=b'{"type":"error","msg":"oops"}')

    with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
        adapter = OpenfangAdapter(OpenFangConfig())
        await adapter.connect("abc")

    assert adapter._recv_task is not None
    adapter._recv_task.cancel()


##### DISCONNECT #####


async def test_disconnect_cancels_recv_task_and_clears_ws() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_ws = MagicMock()
    mock_ws.recv = AsyncMock(return_value=b'{"type":"connected"}')
    mock_ws.close = AsyncMock()

    with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
        adapter = OpenfangAdapter(OpenFangConfig())
        await adapter.connect("agent-xyz")

    await adapter.disconnect()

    assert adapter._ws is None
    assert adapter._recv_task is None
    assert not await adapter.is_connected()


async def test_disconnect_suppresses_connection_closed_on_ws_close() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    import websockets.exceptions

    mock_ws = MagicMock()
    mock_ws.recv = AsyncMock(return_value=b'{"type":"connected"}')
    mock_ws.close = AsyncMock(side_effect=websockets.exceptions.ConnectionClosed(None, None))

    with patch("websockets.connect", new=AsyncMock(return_value=mock_ws)):
        adapter = OpenfangAdapter(OpenFangConfig())
        await adapter.connect("agent-xyz")

    await adapter.disconnect()

    assert adapter._ws is None
    assert adapter._recv_task is None


##### RECEIVE LOOP #####


async def test_receive_loop_routes_response_types_to_queue() -> None:
    from unittest.mock import MagicMock

    import orjson

    adapter = OpenfangAdapter(OpenFangConfig())

    frames = [
        orjson.dumps({"type": "text_delta", "content": "hello"}),
        orjson.dumps({"type": "response", "content": "done"}),
        orjson.dumps({"type": "typing"}),
        orjson.dumps({"type": "phase"}),
        orjson.dumps({"type": "tool_start"}),
        orjson.dumps({"type": "tool_result"}),
    ]

    async def _aiter(self: object) -> object:
        for f in frames:
            yield f

    mock_ws = MagicMock()
    mock_ws.__aiter__ = _aiter
    adapter._ws = mock_ws

    await adapter._receive_loop()

    collected: list[str] = []
    while not adapter._response_queue.empty():
        msg = adapter._response_queue.get_nowait()
        collected.append(str(msg.get("type")))

    assert collected == ["text_delta", "response", "typing", "phase", "tool_start", "tool_result"]


async def test_receive_loop_ignores_known_control_types() -> None:
    from unittest.mock import MagicMock

    import orjson

    adapter = OpenfangAdapter(OpenFangConfig())

    frames = [
        orjson.dumps({"type": "agents_updated"}),
        orjson.dumps({"type": "connected"}),
    ]

    async def _aiter(self: object) -> object:
        for f in frames:
            yield f

    mock_ws = MagicMock()
    mock_ws.__aiter__ = _aiter
    adapter._ws = mock_ws

    await adapter._receive_loop()

    assert adapter._response_queue.empty()


async def test_receive_loop_handles_connection_closed() -> None:
    from unittest.mock import MagicMock

    import websockets.exceptions

    adapter = OpenfangAdapter(OpenFangConfig())

    async def _aiter_raises(self: object) -> object:
        raise websockets.exceptions.ConnectionClosed(None, None)
        yield  # make it a generator

    mock_ws = MagicMock()
    mock_ws.__aiter__ = _aiter_raises
    adapter._ws = mock_ws

    await adapter._receive_loop()

    assert adapter._ws is None
    msg = adapter._response_queue.get_nowait()
    assert msg["type"] == "connection_lost"


async def test_receive_loop_handles_generic_exception() -> None:
    from unittest.mock import MagicMock

    adapter = OpenfangAdapter(OpenFangConfig())

    async def _aiter_raises(self: object) -> object:
        raise RuntimeError("boom")
        yield

    mock_ws = MagicMock()
    mock_ws.__aiter__ = _aiter_raises
    adapter._ws = mock_ws

    await adapter._receive_loop()

    assert adapter._ws is None
    msg = adapter._response_queue.get_nowait()
    assert msg["type"] == "connection_lost"


async def test_receive_loop_returns_early_when_ws_none() -> None:
    adapter = OpenfangAdapter(OpenFangConfig())
    adapter._ws = None
    await adapter._receive_loop()
    assert adapter._response_queue.empty()


##### SEND #####


async def test_send_raises_when_not_connected() -> None:
    import pytest

    adapter = OpenfangAdapter(OpenFangConfig())

    with pytest.raises(ConnectionError, match="not connected"):
        async for _ in adapter.send("hello"):
            pass


async def test_send_yields_text_delta_chunks() -> None:
    from unittest.mock import AsyncMock, MagicMock

    adapter = OpenfangAdapter(OpenFangConfig(timeout=5.0))
    mock_ws = MagicMock()

    async def _send_side_effect(text: str) -> None:
        await adapter._response_queue.put({"type": "text_delta", "content": "Hello"})
        await adapter._response_queue.put({"type": "text_delta", "content": " world"})
        await adapter._response_queue.put({"type": "response", "content": ""})

    mock_ws.send = AsyncMock(side_effect=_send_side_effect)
    adapter._ws = mock_ws

    chunks: list[str] = []
    async for chunk in adapter.send("ping"):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]
    mock_ws.send.assert_awaited_once_with("ping")


async def test_send_yields_response_final_when_no_deltas() -> None:
    from unittest.mock import AsyncMock, MagicMock

    adapter = OpenfangAdapter(OpenFangConfig(timeout=5.0))
    mock_ws = MagicMock()

    async def _send_side_effect(text: str) -> None:
        await adapter._response_queue.put({"type": "response", "content": "final answer"})

    mock_ws.send = AsyncMock(side_effect=_send_side_effect)
    adapter._ws = mock_ws

    chunks: list[str] = []
    async for chunk in adapter.send("ping"):
        chunks.append(chunk)

    assert chunks == ["final answer"]


async def test_send_skips_response_final_when_deltas_already_yielded() -> None:
    from unittest.mock import AsyncMock, MagicMock

    adapter = OpenfangAdapter(OpenFangConfig(timeout=5.0))
    mock_ws = MagicMock()

    async def _send_side_effect(text: str) -> None:
        await adapter._response_queue.put({"type": "text_delta", "content": "partial"})
        await adapter._response_queue.put({"type": "response", "content": "final"})

    mock_ws.send = AsyncMock(side_effect=_send_side_effect)
    adapter._ws = mock_ws

    chunks: list[str] = []
    async for chunk in adapter.send("ping"):
        chunks.append(chunk)

    assert chunks == ["partial"]


async def test_send_breaks_on_timeout() -> None:
    from unittest.mock import AsyncMock, MagicMock

    adapter = OpenfangAdapter(OpenFangConfig(timeout=0.01))
    mock_ws = MagicMock()
    mock_ws.send = AsyncMock()
    adapter._ws = mock_ws

    chunks: list[str] = []
    async for chunk in adapter.send("ping"):
        chunks.append(chunk)

    assert chunks == []


async def test_send_breaks_on_connection_lost() -> None:
    from unittest.mock import AsyncMock, MagicMock

    adapter = OpenfangAdapter(OpenFangConfig(timeout=5.0))
    mock_ws = MagicMock()

    async def _send_side_effect(text: str) -> None:
        await adapter._response_queue.put({"type": "connection_lost"})

    mock_ws.send = AsyncMock(side_effect=_send_side_effect)
    adapter._ws = mock_ws

    chunks: list[str] = []
    async for chunk in adapter.send("ping"):
        chunks.append(chunk)

    assert chunks == []


async def test_send_skips_typing_and_phase() -> None:
    from unittest.mock import AsyncMock, MagicMock

    adapter = OpenfangAdapter(OpenFangConfig(timeout=5.0))
    mock_ws = MagicMock()

    async def _send_side_effect(text: str) -> None:
        await adapter._response_queue.put({"type": "typing"})
        await adapter._response_queue.put({"type": "phase"})
        await adapter._response_queue.put({"type": "text_delta", "content": "hi"})
        await adapter._response_queue.put({"type": "response", "content": ""})

    mock_ws.send = AsyncMock(side_effect=_send_side_effect)
    adapter._ws = mock_ws

    chunks: list[str] = []
    async for chunk in adapter.send("ping"):
        chunks.append(chunk)

    assert chunks == ["hi"]


async def test_send_drains_stale_queue_before_sending() -> None:
    from unittest.mock import AsyncMock, MagicMock

    adapter = OpenfangAdapter(OpenFangConfig(timeout=5.0))
    mock_ws = MagicMock()

    async def _send_side_effect(text: str) -> None:
        await adapter._response_queue.put({"type": "response", "content": "fresh"})

    mock_ws.send = AsyncMock(side_effect=_send_side_effect)
    adapter._ws = mock_ws

    # pre-load stale items before calling send — they must be drained
    await adapter._response_queue.put({"type": "text_delta", "content": "stale-a"})
    await adapter._response_queue.put({"type": "text_delta", "content": "stale-b"})

    chunks: list[str] = []
    async for chunk in adapter.send("ping"):
        chunks.append(chunk)

    assert chunks == ["fresh"]


##### RESOLVE #####


async def test_resolve_agent_id_finds_agent_by_name() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(
        return_value=[
            {"id": "uuid-001", "name": "Jarvis"},
            {"id": "uuid-002", "name": "Ada"},
        ]
    )

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        adapter = OpenfangAdapter(OpenFangConfig(url="http://localhost:4200"))
        result = await adapter.resolve_agent_id("jarvis")

    assert result == "uuid-001"


async def test_resolve_agent_id_case_insensitive() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=[{"id": "uuid-999", "name": "SKYNET"}])

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        adapter = OpenfangAdapter(OpenFangConfig())
        result = await adapter.resolve_agent_id("skynet")

    assert result == "uuid-999"


async def test_resolve_agent_id_raises_when_not_found() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    import pytest

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=[{"id": "uuid-001", "name": "Ada"}])

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        adapter = OpenfangAdapter(OpenFangConfig(url="http://localhost:4200"))
        with pytest.raises(ValueError, match="not found on OpenFang"):
            await adapter.resolve_agent_id("unknown-agent")


async def test_resolve_agent_id_handles_dict_response_format() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={"agents": [{"id": "uuid-777", "name": "Cortana"}]})

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        adapter = OpenfangAdapter(OpenFangConfig())
        result = await adapter.resolve_agent_id("cortana")

    assert result == "uuid-777"
