"""Integration test — WhisperLive STT round-trip via Piper-generated audio.

Requires both WhisperLive (localhost:9090) and Piper (localhost:10200).
Flow: Text → Piper TTS → PCM → WhisperLive STT → Transcript → Assert match.
Run with: uv run pytest tests/integration/daemon/adapters/stt/ -v -s
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from e_clawhisper.daemon.adapters.stt.whisperlive import WhisperliveAdapter
from e_clawhisper.daemon.adapters.tts.piper import PiperAdapter
from e_clawhisper.shared.settings import PiperConfig, WhisperLiveConfig

_STT_CONFIG = WhisperLiveConfig(url="ws://localhost:9090", model="small", language="es")
_TTS_CONFIG = PiperConfig(url="tcp://localhost:10200", sample_rate=22050)
_STT_SAMPLE_RATE = 16000
_TTS_SAMPLE_RATE = 22050


##### HELPERS #####


async def _stt_available() -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("localhost", 9090), timeout=2.0)
        writer.close()
        await writer.wait_closed()
    except (OSError, TimeoutError):
        return False
    return True


async def _tts_available() -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("localhost", 10200), timeout=2.0)
        writer.close()
        await writer.wait_closed()
    except (OSError, TimeoutError):
        return False
    return True


async def _generate_speech(text: str) -> np.ndarray:
    """Generate int16 PCM at 16kHz from Piper TTS (22050Hz) via resampling."""
    tts = PiperAdapter(_TTS_CONFIG)
    chunks = [chunk async for chunk in tts.synthesize(text)]
    pcm = b"".join(chunks)
    audio_22k = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)

    ratio = _STT_SAMPLE_RATE / _TTS_SAMPLE_RATE
    n_out = int(len(audio_22k) * ratio)
    indices = (np.arange(n_out) / ratio).astype(np.int64)
    indices = np.clip(indices, 0, len(audio_22k) - 1)
    return audio_22k[indices].astype(np.int16)


async def _transcribe(audio_16k: np.ndarray, speed: float = 1.5) -> str:
    """Send int16 audio at 16kHz to WhisperLive at ~speed× real-time."""
    stt = WhisperliveAdapter(_STT_CONFIG)
    await stt.connect()

    try:
        await stt.start_utterance()

        chunk_size = 4096
        chunk_duration = chunk_size / _STT_SAMPLE_RATE
        sleep_per_chunk = chunk_duration / speed

        for i in range(0, len(audio_16k), chunk_size):
            chunk = audio_16k[i : i + chunk_size]
            await stt.stream(WhisperliveAdapter.audio_to_float32(chunk))
            await asyncio.sleep(sleep_per_chunk)

        await asyncio.sleep(1.0)
        return await stt.finish_utterance()
    finally:
        await stt.disconnect()


##### ROUND-TRIP: TTS → STT #####


async def test_roundtrip_simple_phrase() -> None:
    if not await _stt_available() or not await _tts_available():
        pytest.skip("STT or TTS not available")

    source_text = "Hola, esto es una prueba"
    audio = await _generate_speech(source_text)
    transcript = await _transcribe(audio)

    print(f"\n  source:     '{source_text}'")
    print(f"  transcript: '{transcript}'")
    print(f"  audio: {len(audio)} samples, {len(audio) / _STT_SAMPLE_RATE:.2f}s")

    lower = transcript.lower()
    assert any(word in lower for word in ("hola", "prueba")), f"Expected keywords not found in: {transcript}"


async def test_roundtrip_numbers() -> None:
    if not await _stt_available() or not await _tts_available():
        pytest.skip("STT or TTS not available")

    source_text = "Uno, dos, tres, cuatro, cinco"
    audio = await _generate_speech(source_text)
    transcript = await _transcribe(audio)

    print(f"\n  source:     '{source_text}'")
    print(f"  transcript: '{transcript}'")

    lower = transcript.lower()
    word_matches = sum(1 for w in ("uno", "dos", "tres", "cuatro", "cinco") if w in lower)
    digit_matches = sum(1 for d in ("1", "2", "3", "4", "5") if d in transcript)
    total = max(word_matches, digit_matches)
    assert total >= 3, f"Expected at least 3/5 numbers (words or digits), got {total}: {transcript}"


async def test_roundtrip_longer_sentence() -> None:
    if not await _stt_available() or not await _tts_available():
        pytest.skip("STT or TTS not available")

    source_text = "El clima de hoy es soleado y hace mucho calor en la ciudad"
    audio = await _generate_speech(source_text)
    transcript = await _transcribe(audio)

    print(f"\n  source:     '{source_text}'")
    print(f"  transcript: '{transcript}'")

    lower = transcript.lower()
    matches = sum(1 for w in ("clima", "soleado", "calor", "ciudad") if w in lower)
    assert matches >= 2, f"Expected at least 2/4 keywords, got {matches}: {transcript}"


##### STT DIRECT — EMPTY AUDIO #####


async def test_empty_audio_returns_empty_transcript() -> None:
    if not await _stt_available():
        pytest.skip("STT not available")

    silence = np.zeros(16000, dtype=np.int16)
    transcript = await _transcribe(silence)

    print(f"\n  silence transcript: '{transcript}'")
    assert len(transcript.strip()) < 10, f"Unexpected text from silence: {transcript}"


##### STT MULTIPLE UTTERANCES #####


async def test_multiple_utterances_sequential() -> None:
    if not await _stt_available() or not await _tts_available():
        pytest.skip("STT or TTS not available")

    phrases = [
        "Buenos días, qué tal estás hoy",
        "Buenas tardes, hace buen tiempo",
        "Buenas noches, hasta mañana amigo",
    ]

    for phrase in phrases:
        audio = await _generate_speech(phrase)
        transcript = await _transcribe(audio)
        print(f"\n  '{phrase}' -> '{transcript}'")
        assert len(transcript.strip()) > 0, f"Empty transcript for '{phrase}'"
