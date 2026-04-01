"""Tests for EVoice TTS adapter — unit tests with mocked WebSocket."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import orjson

from e_heed.daemon.adapters.tts.evoice import EVoiceTTSAdapter
from e_heed.shared.settings import EVoiceTTSConfig


##### HELPERS #####


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


def _make_adapter(url: str = "http://localhost:45140") -> EVoiceTTSAdapter:
    config = EVoiceTTSConfig(url=url)
    return EVoiceTTSAdapter(config)


def _make_ws_mock(messages: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.__aiter__ = lambda self: _async_iter(messages)
    mock.send = AsyncMock()
    mock.close = AsyncMock()
    return mock


##### INIT #####


async def test_init_defaults() -> None:
    adapter = _make_adapter()
    assert adapter._base_url == "http://localhost:45140"
    assert adapter._voice == "af_heart"
    assert adapter._speed == 1.0
    assert adapter.sample_rate == 24000
    assert adapter._ws is None
    assert adapter._stopped is False


async def test_init_strips_trailing_slash() -> None:
    adapter = _make_adapter(url="http://localhost:45140/")
    assert adapter._base_url == "http://localhost:45140"


##### STOP #####


async def test_stop_sets_flag() -> None:
    adapter = _make_adapter()
    await adapter.stop()
    assert adapter._stopped is True


##### DISCONNECT #####


async def test_disconnect_noop_without_connection() -> None:
    adapter = _make_adapter()
    await adapter.disconnect()
    assert adapter._ws is None


async def test_disconnect_closes_ws() -> None:
    adapter = _make_adapter()
    mock_ws = AsyncMock()
    adapter._ws = mock_ws
    await adapter.disconnect()
    mock_ws.close.assert_awaited_once()
    assert adapter._ws is None


##### SYNTHESIZE #####


async def test_synthesize_yields_audio_chunks() -> None:
    adapter = _make_adapter()

    pcm_data = b"\x00" * 480
    delta_msg = orjson.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(pcm_data).decode()}).decode()
    done_msg = orjson.dumps({"type": "speech.audio.done"}).decode()

    adapter._ws = _make_ws_mock([delta_msg, done_msg])

    chunks: list[bytes] = []
    async for chunk in adapter.synthesize("hello"):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0] == pcm_data


async def test_synthesize_stops_mid_stream() -> None:
    adapter = _make_adapter()

    pcm_data = b"\x00" * 480
    delta_msg = orjson.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(pcm_data).decode()}).decode()

    adapter._ws = _make_ws_mock([delta_msg, delta_msg, delta_msg])

    chunks: list[bytes] = []
    async for chunk in adapter.synthesize("hello"):
        chunks.append(chunk)
        await adapter.stop()

    assert len(chunks) == 1
