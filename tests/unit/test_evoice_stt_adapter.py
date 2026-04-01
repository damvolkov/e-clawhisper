"""Tests for EVoice STT adapter — unit tests with mocked WebSocket."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from e_heed.daemon.adapters.stt.evoice import EVoiceSTTAdapter
from e_heed.shared.settings import EVoiceSTTConfig


##### HELPERS #####


async def _async_iter(items: list[bytes]) -> AsyncIterator[bytes]:
    for item in items:
        yield item


def _make_adapter(url: str = "http://localhost:45140", language: str = "en") -> EVoiceSTTAdapter:
    config = EVoiceSTTConfig(url=url, language=language)
    return EVoiceSTTAdapter(config)


def _make_ws_mock(messages: list[bytes]) -> MagicMock:
    mock = MagicMock()
    mock.__aiter__ = lambda self: _async_iter(messages)
    mock.send = AsyncMock()
    mock.close = AsyncMock()
    return mock


##### INIT #####


async def test_init_sets_base_url() -> None:
    adapter = _make_adapter()
    assert adapter._base_url == "http://localhost:45140"
    assert adapter._language == "en"
    assert adapter._ws is None
    assert adapter._ready is False


async def test_init_strips_trailing_slash() -> None:
    adapter = _make_adapter(url="http://localhost:45140/")
    assert adapter._base_url == "http://localhost:45140"


##### STREAM #####


async def test_stream_noop_when_not_ready() -> None:
    adapter = _make_adapter()
    await adapter.stream(b"\x00" * 1024)


async def test_stream_sends_when_ready() -> None:
    adapter = _make_adapter()
    adapter._ready = True
    adapter._ws = AsyncMock()
    await adapter.stream(b"\x00" * 1024)
    adapter._ws.send.assert_awaited_once_with(b"\x00" * 1024)


##### FINISH UTTERANCE #####


async def test_finish_utterance_empty_without_session() -> None:
    adapter = _make_adapter()
    result = await adapter.finish_utterance()
    assert result == ""


async def test_finish_utterance_sends_end_of_audio() -> None:
    adapter = _make_adapter()
    adapter._ws = AsyncMock()
    adapter._recv_task = asyncio.create_task(asyncio.sleep(10))
    adapter._final_text = "hello world"

    result = await adapter.finish_utterance()

    adapter._ws.send.assert_awaited_once_with("END_OF_AUDIO")
    assert result == "hello world"


##### RECEIVE LOOP #####


async def test_receive_loop_extracts_transcript_update() -> None:
    adapter = _make_adapter()
    messages = [
        orjson.dumps({"type": "transcript_update", "text": "hello"}),
        orjson.dumps({"type": "session_end", "text": ""}),
    ]
    adapter._ws = _make_ws_mock(messages)

    await adapter._ev_receive_loop()

    assert adapter._final_text == "hello"


async def test_receive_loop_stops_on_session_end() -> None:
    adapter = _make_adapter()
    messages = [orjson.dumps({"type": "session_end", "text": "final"})]
    adapter._ws = _make_ws_mock(messages)

    await adapter._ev_receive_loop()

    assert adapter._final_text == "final"


async def test_receive_loop_noop_without_ws() -> None:
    adapter = _make_adapter()
    adapter._ws = None
    await adapter._ev_receive_loop()


##### DISCONNECT #####


async def test_disconnect_noop_without_connection() -> None:
    adapter = _make_adapter()
    await adapter.disconnect()
    assert adapter._ready is False


##### CLOSE SESSION #####


async def test_close_session_cancels_recv_task() -> None:
    adapter = _make_adapter()
    adapter._recv_task = asyncio.create_task(asyncio.sleep(10))
    adapter._ws = AsyncMock()

    await adapter._close_session()

    assert adapter._recv_task is None
    assert adapter._ws is None
    assert adapter._ready is False
