"""Integration test — Piper TTS via Wyoming protocol.

Requires Piper running at localhost:10200.
Run with: uv run pytest tests/integration/daemon/adapters/tts/ -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from e_clawhisper.daemon.adapters.tts.piper import PiperAdapter
from e_clawhisper.shared.settings import PiperConfig

_CONFIG = PiperConfig(host="localhost", port=10200, sample_rate=22050)
_SAMPLE_RATE = 22050


##### HELPERS #####


async def _piper_available() -> bool:
    import asyncio

    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("localhost", 10200), timeout=2.0)
        writer.close()
        await writer.wait_closed()
    except (OSError, TimeoutError):
        return False
    return True


async def _synthesize_full(adapter: PiperAdapter, text: str) -> tuple[bytes, np.ndarray, float]:
    """Synthesize text, return (raw_pcm, int16_array, duration_seconds)."""
    chunks = [chunk async for chunk in adapter.synthesize(text)]
    pcm = b"".join(chunks)
    audio = np.frombuffer(pcm, dtype=np.int16)
    duration = len(audio) / _SAMPLE_RATE
    return pcm, audio, duration


##### BASIC SYNTHESIS #####


async def test_synthesize_returns_audio() -> None:
    if not await _piper_available():
        pytest.skip("Piper not available at localhost:10200")

    adapter = PiperAdapter(_CONFIG)
    pcm, audio, duration = await _synthesize_full(adapter, "Hola, esto es una prueba")

    assert len(pcm) > 0, "No PCM data returned"
    assert len(audio) > 0, "Empty audio array"
    assert duration > 0.5, f"Audio too short: {duration:.2f}s"
    print(f"\n  pcm={len(pcm)} bytes, samples={len(audio)}, duration={duration:.2f}s")


async def test_synthesize_longer_text_produces_more_audio() -> None:
    if not await _piper_available():
        pytest.skip("Piper not available at localhost:10200")

    adapter = PiperAdapter(_CONFIG)
    _, _, short_dur = await _synthesize_full(adapter, "Hola")
    _, _, long_dur = await _synthesize_full(adapter, "Esta es una frase mucho más larga para comprobar que produce más audio")

    assert long_dur > short_dur, f"Long ({long_dur:.2f}s) should exceed short ({short_dur:.2f}s)"
    print(f"\n  short={short_dur:.2f}s, long={long_dur:.2f}s")


##### AUDIO QUALITY #####


async def test_audio_is_not_silence() -> None:
    if not await _piper_available():
        pytest.skip("Piper not available at localhost:10200")

    adapter = PiperAdapter(_CONFIG)
    _, audio, _ = await _synthesize_full(adapter, "Probando uno dos tres")

    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    assert rms > 100, f"Audio appears silent (RMS={rms:.1f})"
    print(f"\n  RMS={rms:.1f}, max={np.max(np.abs(audio))}")


##### STOP CANCELLATION #####


async def test_stop_interrupts_synthesis() -> None:
    if not await _piper_available():
        pytest.skip("Piper not available at localhost:10200")

    adapter = PiperAdapter(_CONFIG)
    chunks: list[bytes] = []

    async for chunk in adapter.synthesize("Esta es una frase extremadamente larga que debería tardar bastante"):
        chunks.append(chunk)
        if len(chunks) >= 3:
            await adapter.stop()
            break

    _, _, full_dur = await _synthesize_full(adapter, "Esta es una frase extremadamente larga que debería tardar bastante")
    partial_pcm = b"".join(chunks)
    partial_samples = len(partial_pcm) // 2
    partial_dur = partial_samples / _SAMPLE_RATE

    assert partial_dur < full_dur, "Stop should have truncated the output"
    print(f"\n  partial={partial_dur:.2f}s, full={full_dur:.2f}s")


##### SEQUENTIAL CALLS #####


async def test_multiple_sequential_synthesize() -> None:
    if not await _piper_available():
        pytest.skip("Piper not available at localhost:10200")

    adapter = PiperAdapter(_CONFIG)
    texts = ["Primera frase", "Segunda frase", "Tercera frase"]

    for text in texts:
        _, audio, duration = await _synthesize_full(adapter, text)
        assert duration > 0.3, f"'{text}' too short: {duration:.2f}s"
        print(f"\n  '{text}' -> {duration:.2f}s ({len(audio)} samples)")
