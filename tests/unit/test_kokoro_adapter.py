"""Tests for KokoroAdapter — init, properties, stop, synthesize."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from e_clawhisper.daemon.adapters.tts.kokoro import KokoroAdapter
from e_clawhisper.shared.settings import KokoroConfig


##### INIT #####


def test_init_from_config() -> None:
    cfg = KokoroConfig(url="http://192.168.1.5:45130", voice="test_voice", sample_rate=48000)
    adapter = KokoroAdapter(cfg)
    assert adapter._base_url == "http://192.168.1.5:45130"
    assert adapter._voice == "test_voice"
    assert adapter._sample_rate == 48000


def test_init_strips_trailing_slash() -> None:
    cfg = KokoroConfig(url="http://localhost:45130/")
    adapter = KokoroAdapter(cfg)
    assert not adapter._base_url.endswith("/")


def test_sample_rate_property() -> None:
    cfg = KokoroConfig(sample_rate=22050)
    adapter = KokoroAdapter(cfg)
    assert adapter.sample_rate == 22050


##### STOP #####


async def test_stop_sets_flag() -> None:
    adapter = KokoroAdapter(KokoroConfig())
    assert adapter._stopped is False
    await adapter.stop()
    assert adapter._stopped is True


##### INIT DEFAULTS #####


def test_default_config() -> None:
    cfg = KokoroConfig()
    adapter = KokoroAdapter(cfg)
    assert adapter._model == "kokoro"
    assert adapter._response_format == "pcm"


##### SYNTHESIZE #####


@patch("e_clawhisper.daemon.adapters.tts.kokoro.httpx.AsyncClient")
async def test_synthesize_yields_chunks(mock_client_cls: MagicMock) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()

    async def _aiter_bytes(chunk_size: int = 4096):  # noqa: ANN001
        yield b"\x00" * 1024
        yield b"\x01" * 512

    mock_response.aiter_bytes = _aiter_bytes

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=AsyncMock())
    mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    adapter = KokoroAdapter(KokoroConfig())
    chunks = [c async for c in adapter.synthesize("hello")]

    assert len(chunks) == 2
    assert chunks[0] == b"\x00" * 1024
    assert chunks[1] == b"\x01" * 512


@patch("e_clawhisper.daemon.adapters.tts.kokoro.httpx.AsyncClient")
async def test_synthesize_stops_on_flag(mock_client_cls: MagicMock) -> None:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()

    adapter = KokoroAdapter(KokoroConfig())

    async def _aiter_bytes(chunk_size: int = 4096):  # noqa: ANN001
        yield b"\x00" * 1024
        adapter._stopped = True  # set during iteration
        yield b"\x01" * 512  # should not be yielded

    mock_response.aiter_bytes = _aiter_bytes

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=AsyncMock())
    mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    chunks = [c async for c in adapter.synthesize("hello")]

    assert len(chunks) == 1
    assert chunks[0] == b"\x00" * 1024
