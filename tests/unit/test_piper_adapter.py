"""Tests for PiperAdapter — init, properties, stop flag, synthesize."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from e_heed.daemon.adapters.tts.piper import PiperAdapter
from e_heed.shared.settings import PiperConfig

##### INIT #####


def test_init_extracts_host_port() -> None:
    cfg = PiperConfig(url="tcp://192.168.1.5:10200")
    adapter = PiperAdapter(cfg)
    assert adapter._host == "192.168.1.5"
    assert adapter._port == 10200


def test_init_default_config() -> None:
    cfg = PiperConfig()
    adapter = PiperAdapter(cfg)
    assert adapter._host == "localhost"
    assert adapter._port == 45130


def test_sample_rate_property() -> None:
    cfg = PiperConfig(sample_rate=48000)
    adapter = PiperAdapter(cfg)
    assert adapter.sample_rate == 48000


##### STOP #####


async def test_stop_sets_flag() -> None:
    adapter = PiperAdapter(PiperConfig())
    assert adapter._stopped is False
    await adapter.stop()
    assert adapter._stopped is True


##### SYNTHESIZE #####


@patch("e_heed.daemon.adapters.tts.piper.AsyncTcpClient")
async def test_synthesize_yields_audio_chunks(mock_client_cls: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_client_cls.return_value = mock_client

    audio_event = MagicMock()
    audio_event.type = "audio-chunk"

    stop_event = MagicMock()
    stop_event.type = "audio-stop"

    mock_client.read_event = AsyncMock(side_effect=[audio_event, stop_event])

    with (
        patch("e_heed.daemon.adapters.tts.piper.AudioChunk") as mock_ac,
        patch("e_heed.daemon.adapters.tts.piper.AudioStop") as mock_as,
    ):
        mock_ac.is_type.side_effect = lambda t: t == "audio-chunk"
        mock_as.is_type.side_effect = lambda t: t == "audio-stop"
        chunk_obj = MagicMock()
        chunk_obj.audio = b"\x00" * 1024
        mock_ac.from_event.return_value = chunk_obj

        adapter = PiperAdapter(PiperConfig())
        chunks = [c async for c in adapter.synthesize("hello")]

    assert len(chunks) == 1
    assert chunks[0] == b"\x00" * 1024
    mock_client.connect.assert_called_once()
    mock_client.write_event.assert_called_once()


@patch("e_heed.daemon.adapters.tts.piper.AsyncTcpClient")
async def test_synthesize_stops_on_none_event(mock_client_cls: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_client_cls.return_value = mock_client
    mock_client.read_event = AsyncMock(return_value=None)

    adapter = PiperAdapter(PiperConfig())
    chunks = [c async for c in adapter.synthesize("hello")]
    assert chunks == []


@patch("e_heed.daemon.adapters.tts.piper.AsyncTcpClient")
async def test_synthesize_respects_stop_flag(mock_client_cls: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_client_cls.return_value = mock_client

    adapter = PiperAdapter(PiperConfig())

    # read_event sets _stopped on first call, then returns an event
    # synthesize resets _stopped=False at start, so we set it during execution
    def _set_stopped() -> MagicMock:
        adapter._stopped = True
        evt = MagicMock()
        evt.type = "other"
        return evt

    mock_client.read_event = AsyncMock(side_effect=lambda: _set_stopped())

    with (
        patch("e_heed.daemon.adapters.tts.piper.AudioChunk") as mock_ac,
        patch("e_heed.daemon.adapters.tts.piper.AudioStop") as mock_as,
    ):
        mock_ac.is_type.return_value = False
        mock_as.is_type.return_value = False
        chunks = [c async for c in adapter.synthesize("hello")]

    assert chunks == []
