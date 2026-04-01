"""Tests for AudioAdapter — mic capture callback, queue, drain, playback."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd

from e_heed.daemon.adapters.audio import AudioAdapter
from e_heed.shared.settings import AudioConfig

##### FIXTURES #####


def _make_config(**kw: object) -> AudioConfig:
    defaults = {"sample_rate": 16000, "channels": 1, "chunk_size": 512, "queue_size": 10}
    defaults.update(kw)
    return AudioConfig(**defaults)


def _make_adapter(config: AudioConfig | None = None) -> AudioAdapter:
    return AudioAdapter(config or _make_config())


##### INIT #####


def test_init_sets_properties() -> None:
    cfg = _make_config(sample_rate=44100, chunk_size=1024)
    adapter = _make_adapter(cfg)
    assert adapter.sample_rate == 44100
    assert adapter.chunk_size == 1024


def test_queue_is_bounded() -> None:
    adapter = _make_adapter(_make_config(queue_size=5))
    assert adapter.queue.maxsize == 5


##### ENQUEUE #####


def test_enqueue_puts_data() -> None:
    adapter = _make_adapter()
    chunk = np.zeros(512, dtype=np.float32)
    adapter._enqueue(chunk)
    assert adapter.queue.qsize() == 1


def test_enqueue_drops_when_full() -> None:
    adapter = _make_adapter(_make_config(queue_size=1))
    adapter._enqueue(np.zeros(512, dtype=np.float32))
    adapter._enqueue(np.zeros(512, dtype=np.float32))
    assert adapter.queue.qsize() == 1


##### AUDIO CALLBACK #####


def test_audio_callback_enqueues_f32_copy() -> None:
    adapter = _make_adapter()
    loop = asyncio.new_event_loop()
    adapter._loop = loop

    indata = np.random.default_rng(42).uniform(-1, 1, (512, 1)).astype(np.float32)
    status = MagicMock()

    # Simulate threadsafe call by directly calling enqueue
    adapter._audio_callback(indata, 512, None, status)
    loop.close()


def test_audio_callback_converts_non_f32() -> None:
    adapter = _make_adapter()
    loop = asyncio.new_event_loop()
    adapter._loop = loop

    indata = np.zeros((512, 1), dtype=np.float64)
    status = MagicMock()

    adapter._audio_callback(indata, 512, None, status)
    loop.close()


def test_audio_callback_noop_without_loop() -> None:
    adapter = _make_adapter()
    indata = np.zeros((512, 1), dtype=np.float32)
    adapter._audio_callback(indata, 512, None, MagicMock())
    assert adapter.queue.qsize() == 0


##### DRAIN #####


def test_drain_clears_queue() -> None:
    adapter = _make_adapter()
    for _ in range(5):
        adapter._enqueue(np.zeros(512, dtype=np.float32))
    assert adapter.queue.qsize() == 5
    adapter.drain()
    assert adapter.queue.qsize() == 0


def test_drain_empty_queue_noop() -> None:
    adapter = _make_adapter()
    adapter.drain()
    assert adapter.queue.qsize() == 0


##### START / STOP #####


@patch("e_heed.daemon.adapters.audio.sd")
async def test_start_creates_stream(mock_sd: MagicMock) -> None:
    adapter = _make_adapter()
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    await adapter.start()

    mock_sd.InputStream.assert_called_once()
    mock_stream.start.assert_called_once()
    assert adapter._stream is not None
    assert adapter._loop is not None


@patch("e_heed.daemon.adapters.audio._BASE_DELAY", 0.0)
@patch("e_heed.daemon.adapters.audio.sd")
async def test_start_retries_on_device_error(mock_sd: MagicMock) -> None:
    mock_stream = MagicMock()
    mock_sd.InputStream.side_effect = [sd.PortAudioError("no device"), sd.PortAudioError("no device"), mock_stream]
    mock_sd.PortAudioError = sd.PortAudioError

    adapter = _make_adapter()
    await adapter.start()

    assert mock_sd.InputStream.call_count == 3
    mock_stream.start.assert_called_once()


@patch("e_heed.daemon.adapters.audio._BASE_DELAY", 0.0)
@patch("e_heed.daemon.adapters.audio._MAX_RETRIES", 2)
@patch("e_heed.daemon.adapters.audio.sd")
async def test_start_raises_after_max_retries(mock_sd: MagicMock) -> None:
    mock_sd.InputStream.side_effect = sd.PortAudioError("no device")
    mock_sd.PortAudioError = sd.PortAudioError

    adapter = _make_adapter()
    with pytest.raises(sd.PortAudioError, match="not available"):
        await adapter.start()


@patch("e_heed.daemon.adapters.audio.sd")
async def test_stop_closes_stream(mock_sd: MagicMock) -> None:
    adapter = _make_adapter()
    mock_stream = MagicMock()
    mock_sd.InputStream.return_value = mock_stream

    await adapter.start()
    await adapter.stop()

    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    assert adapter._stream is None
    assert adapter._loop is None


async def test_stop_noop_without_stream() -> None:
    adapter = _make_adapter()
    await adapter.stop()
    assert adapter._stream is None


##### STOP PLAYBACK #####


@patch("e_heed.daemon.adapters.audio.sd")
def test_stop_playback_sets_flag(mock_sd: MagicMock) -> None:
    adapter = _make_adapter()
    adapter.stop_playback()
    assert adapter._playback_stopped is True
    mock_sd.stop.assert_called_once()


##### PLAY #####


@patch("e_heed.daemon.adapters.audio.sd")
async def test_play_stops_on_playback_flag(mock_sd: MagicMock) -> None:
    """Play loop exits when _playback_stopped is set before sentinel."""
    mock_out = MagicMock()
    mock_sd.RawOutputStream.return_value = mock_out

    adapter = _make_adapter()
    pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    await pcm_queue.put(b"\x00" * 100)

    async def _stop_later() -> None:
        await asyncio.sleep(0.01)
        adapter._playback_stopped = True
        await pcm_queue.put(b"\x00" * 100)  # unblock .get()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_stop_later())
        tg.create_task(adapter.play(pcm_queue, 16000))

    mock_out.start.assert_called_once()


@patch("e_heed.daemon.adapters.audio.sd")
async def test_play_sentinel_exits_loop(mock_sd: MagicMock) -> None:
    """Sentinel (None) exits the read loop; set _playback_stopped to skip drain."""
    mock_out = MagicMock()
    mock_sd.RawOutputStream.return_value = mock_out

    adapter = _make_adapter()
    adapter._playback_stopped = True  # skip drain wait
    pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    await pcm_queue.put(None)  # immediate sentinel

    duration = await adapter.play(pcm_queue, 16000)
    assert duration == 0.0
    mock_out.start.assert_called_once()
