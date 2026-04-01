"""Tests for WhisperliveAdapter — init, audio conversion, state management."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import orjson
import pytest
import websockets

from e_heed.daemon.adapters.stt.whisperlive import WhisperliveAdapter
from e_heed.shared.settings import WhisperLiveConfig

##### INIT #####


def test_init_from_config() -> None:
    cfg = WhisperLiveConfig(url="ws://192.168.1.5:9090", model="small", language="es")
    adapter = WhisperliveAdapter(cfg)
    assert adapter._ws_url == "ws://192.168.1.5:9090/"
    assert adapter._model == "small"
    assert adapter._language == "es"
    assert adapter._ws is None
    assert adapter._ready is False


def test_init_defaults() -> None:
    cfg = WhisperLiveConfig()
    adapter = WhisperliveAdapter(cfg)
    assert "45120" in adapter._ws_url
    assert adapter._model == "large-v3"


##### AUDIO CONVERSION #####


def test_audio_to_float32_shape() -> None:
    audio = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    result = WhisperliveAdapter.audio_to_float32(audio)
    assert isinstance(result, bytes)
    assert len(result) == len(audio) * 4  # float32 = 4 bytes


def test_audio_to_float32_values() -> None:
    audio = np.array([0, 32767, -32768], dtype=np.int16)
    result = WhisperliveAdapter.audio_to_float32(audio)
    f32 = np.frombuffer(result, dtype=np.float32)

    assert f32[0] == pytest.approx(0.0)
    assert f32[1] == pytest.approx(32767.0 / 32768.0, abs=1e-4)
    assert f32[2] == pytest.approx(-1.0)


def test_audio_to_float32_silence() -> None:
    silence = np.zeros(512, dtype=np.int16)
    result = WhisperliveAdapter.audio_to_float32(silence)
    f32 = np.frombuffer(result, dtype=np.float32)
    assert np.all(f32 == 0.0)


##### STATE MANAGEMENT #####


async def test_disconnect_noop_without_connection() -> None:
    adapter = WhisperliveAdapter(WhisperLiveConfig())
    await adapter.disconnect()
    assert adapter._ws is None


async def test_stream_noop_when_not_ready() -> None:
    adapter = WhisperliveAdapter(WhisperLiveConfig())
    await adapter.stream(b"audio-data")  # should not raise


async def test_finish_utterance_empty_without_session() -> None:
    adapter = WhisperliveAdapter(WhisperLiveConfig())
    result = await adapter.finish_utterance()
    assert result == ""


##### CONNECT #####


def _make_ws_mock(ready_payload: dict | None = None) -> MagicMock:
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock(return_value=orjson.dumps(ready_payload or {"message": "SERVER_READY"}))
    ws.close = AsyncMock()
    ws.__aiter__ = MagicMock(return_value=iter([]))
    return ws


async def test_connect_sets_ready_true() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter.connect()
    assert adapter._ready is True
    assert adapter._ws is ws


async def test_connect_starts_recv_task() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter.connect()
    assert adapter._recv_task is not None


##### OPEN SESSION #####


async def test_open_session_sends_config_message() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig(model="tiny", language="fr"))
        await adapter._open_session()

    sent_bytes: bytes = ws.send.call_args_list[0].args[0]
    payload = orjson.loads(sent_bytes)
    assert payload["model"] == "tiny"
    assert payload["language"] == "fr"
    assert payload["task"] == "transcribe"
    assert payload["use_vad"] is False
    assert "uid" in payload


async def test_open_session_uid_is_set() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter._open_session()
    assert len(adapter._uid) == 36


async def test_open_session_ready_on_server_ready() -> None:
    ws = _make_ws_mock({"message": "SERVER_READY"})
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter._open_session()
    assert adapter._ready is True


async def test_open_session_ready_on_unexpected_message() -> None:
    ws = _make_ws_mock({"message": "UNEXPECTED"})
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter._open_session()
    assert adapter._ready is True


async def test_open_session_resets_final_text() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        adapter._final_text = "stale text"
        await adapter._open_session()
    assert adapter._final_text == ""


##### CLOSE SESSION #####


async def test_close_session_cancels_recv_task() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter._open_session()

    await adapter._close_session()
    assert adapter._recv_task is None


async def test_close_session_closes_ws() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter._open_session()

    await adapter._close_session()
    ws.close.assert_awaited_once()
    assert adapter._ws is None


async def test_close_session_sets_ready_false() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter._open_session()

    await adapter._close_session()
    assert adapter._ready is False


async def test_close_session_suppresses_connection_closed() -> None:
    ws = _make_ws_mock()
    ws.close = AsyncMock(side_effect=websockets.exceptions.ConnectionClosed(None, None))
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter._open_session()

    await adapter._close_session()
    assert adapter._ws is None


async def test_close_session_noop_without_ws() -> None:
    adapter = WhisperliveAdapter(WhisperLiveConfig())
    await adapter._close_session()
    assert adapter._ws is None
    assert adapter._ready is False


##### START UTTERANCE #####


async def test_start_utterance_reopens_session() -> None:
    ws_first = _make_ws_mock()
    ws_second = _make_ws_mock()
    connect_mock = AsyncMock(side_effect=[ws_first, ws_second])
    with patch("websockets.connect", new=connect_mock):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter.connect()
        await adapter.start_utterance()

    assert adapter._ws is ws_second
    assert adapter._ready is True


async def test_start_utterance_resets_final_text() -> None:
    ws_first = _make_ws_mock()
    ws_second = _make_ws_mock()
    connect_mock = AsyncMock(side_effect=[ws_first, ws_second])
    with patch("websockets.connect", new=connect_mock):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter.connect()
        adapter._final_text = "previous result"
        await adapter.start_utterance()

    assert adapter._final_text == ""


##### STREAM #####


async def test_stream_sends_chunk_when_ready() -> None:
    ws = _make_ws_mock()
    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig())
        await adapter._open_session()

    ws.send.reset_mock()
    await adapter.stream(b"\x00\x01\x02\x03")
    ws.send.assert_awaited_once_with(b"\x00\x01\x02\x03")


async def test_stream_skips_when_ws_none() -> None:
    adapter = WhisperliveAdapter(WhisperLiveConfig())
    await adapter.stream(b"data")


async def test_stream_skips_when_not_ready() -> None:
    ws = _make_ws_mock()
    adapter = WhisperliveAdapter(WhisperLiveConfig())
    adapter._ws = ws
    adapter._ready = False
    await adapter.stream(b"data")
    ws.send.assert_not_awaited()


##### FINISH UTTERANCE #####


async def test_finish_utterance_sends_end_of_audio() -> None:
    ws = _make_ws_mock()

    async def _blocked_recv_task() -> None:
        await asyncio.sleep(10)

    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig(finish_timeout=0.05))
        await adapter._open_session()

    ws.send.reset_mock()
    await adapter.finish_utterance()
    ws.send.assert_any_await('{"type": "END_OF_AUDIO"}')


async def test_finish_utterance_returns_final_text() -> None:
    ws = _make_ws_mock()

    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig(finish_timeout=0.05))
        await adapter._open_session()

    adapter._final_text = "hello world"
    result = await adapter.finish_utterance()
    assert result == "hello world"


async def test_finish_utterance_suppresses_connection_closed_on_send() -> None:
    ws = _make_ws_mock()

    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig(finish_timeout=0.05))
        await adapter._open_session()

    ws.send = AsyncMock(side_effect=websockets.exceptions.ConnectionClosed(None, None))
    adapter._final_text = "partial"
    result = await adapter.finish_utterance()
    assert result == "partial"


async def test_finish_utterance_clears_recv_task() -> None:
    ws = _make_ws_mock()

    with patch("websockets.connect", new=AsyncMock(return_value=ws)):
        adapter = WhisperliveAdapter(WhisperLiveConfig(finish_timeout=0.05))
        await adapter._open_session()

    await adapter.finish_utterance()
    assert adapter._recv_task is None


##### RECEIVE LOOP #####


async def test_receive_loop_extracts_segment_text() -> None:
    msgs = [
        orjson.dumps({"segments": [{"text": "  hello  "}, {"text": "world"}]}),
        orjson.dumps({"eos": True}),
    ]

    async def _gen():
        for m in msgs:
            yield m

    ws = MagicMock()
    ws.__aiter__ = MagicMock(return_value=_gen())

    adapter = WhisperliveAdapter(WhisperLiveConfig())
    adapter._ws = ws
    await adapter._receive_loop()

    assert adapter._final_text == "hello world"


async def test_receive_loop_stops_on_eos() -> None:
    extra_called: list[bool] = []

    async def _gen():
        yield orjson.dumps({"eos": True})
        extra_called.append(True)
        yield orjson.dumps({"segments": [{"text": "after eos"}]})

    ws = MagicMock()
    ws.__aiter__ = MagicMock(return_value=_gen())

    adapter = WhisperliveAdapter(WhisperLiveConfig())
    adapter._ws = ws
    await adapter._receive_loop()

    assert adapter._final_text == ""
    assert extra_called == []


async def test_receive_loop_handles_connection_closed() -> None:
    async def _gen():
        raise websockets.exceptions.ConnectionClosed(None, None)
        yield  # make it an async generator

    ws = MagicMock()
    ws.__aiter__ = MagicMock(return_value=_gen())

    adapter = WhisperliveAdapter(WhisperLiveConfig())
    adapter._ws = ws
    await adapter._receive_loop()

    assert adapter._final_text == ""


async def test_receive_loop_handles_generic_exception() -> None:
    async def _gen():
        raise ValueError("bad payload")
        yield  # make it an async generator

    ws = MagicMock()
    ws.__aiter__ = MagicMock(return_value=_gen())

    adapter = WhisperliveAdapter(WhisperLiveConfig())
    adapter._ws = ws
    await adapter._receive_loop()

    assert adapter._final_text == ""


async def test_receive_loop_skips_empty_segment_text() -> None:
    msgs = [
        orjson.dumps({"segments": [{"text": "  "}, {"text": ""}]}),
        orjson.dumps({"eos": True}),
    ]

    async def _gen():
        for m in msgs:
            yield m

    ws = MagicMock()
    ws.__aiter__ = MagicMock(return_value=_gen())

    adapter = WhisperliveAdapter(WhisperLiveConfig())
    adapter._ws = ws
    await adapter._receive_loop()

    assert adapter._final_text == ""


async def test_receive_loop_noop_without_ws() -> None:
    adapter = WhisperliveAdapter(WhisperLiveConfig())
    await adapter._receive_loop()
    assert adapter._final_text == ""
