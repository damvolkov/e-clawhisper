"""Integration test — streaming turn pipeline: Agent → sentence split → TTS → speaker.

Requires OpenFang (127.0.0.1:4200) and Piper TTS (localhost:10200).
Run with: uv run pytest tests/integration/daemon/test_streaming_pipeline.py -v -s
"""

from __future__ import annotations

import asyncio
import re

import httpx
import pytest

from e_clawhisper.daemon.adapters.agent.openfang import OpenfangAdapter
from e_clawhisper.daemon.adapters.tts.piper import PiperAdapter
from e_clawhisper.shared.settings import OpenFangConfig, PiperConfig

_AGENT_CONFIG = OpenFangConfig(url="http://127.0.0.1:4200", timeout=30.0)
_TTS_CONFIG = PiperConfig(url="tcp://localhost:10200", sample_rate=22050)
_AGENT_NAME = "damien"


##### HELPERS #####


async def _openfang_available() -> bool:
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:4200", timeout=3.0) as client:
            resp = await client.get("/api/agents")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


async def _piper_available() -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("localhost", 10200), timeout=2.0)
        writer.close()
        await writer.wait_closed()
    except (OSError, TimeoutError):
        return False
    return True


##### AGENT → SENTENCE SPLIT → TTS STREAMING #####


async def test_agent_to_tts_streaming_produces_audio() -> None:
    """Full streaming: Agent text_delta → sentence split → TTS per sentence → PCM chunks."""
    if not await _openfang_available() or not await _piper_available():
        pytest.skip("OpenFang or Piper not available")

    # Connect agent
    agent = OpenfangAdapter(_AGENT_CONFIG)
    agent_id = await agent.resolve_agent_id(_AGENT_NAME)
    await agent.connect(agent_id)

    tts = PiperAdapter(_TTS_CONFIG)

    try:
        # Stream agent response with sentence splitting
        sentence_re = re.compile(r"(?<=[.!?])\s+|\n")
        buffer = ""
        sentences: list[str] = []
        total_pcm_bytes = 0
        sentence_audio_durations: list[float] = []

        print("\n  Sending: 'Dime tres cosas sobre Python. Sé breve.'")

        async for chunk in agent.send("Dime tres cosas sobre Python. Sé breve."):
            buffer += chunk

            while (m := sentence_re.search(buffer)) is not None:
                sentence = buffer[: m.start()].strip()
                buffer = buffer[m.end() :]
                if sentence:
                    sentences.append(sentence)
                    # Synthesize this sentence immediately
                    pcm_chunks: list[bytes] = []
                    async for pcm in tts.synthesize(sentence):
                        pcm_chunks.append(pcm)
                    pcm_data = b"".join(pcm_chunks)
                    total_pcm_bytes += len(pcm_data)
                    dur = (len(pcm_data) // 2) / _TTS_CONFIG.sample_rate
                    sentence_audio_durations.append(dur)
                    print(f"  sentence: {sentence!r} → {dur:.2f}s audio")

        # Flush remaining
        if remaining := buffer.strip():
            sentences.append(remaining)
            pcm_chunks = []
            async for pcm in tts.synthesize(remaining):
                pcm_chunks.append(pcm)
            pcm_data = b"".join(pcm_chunks)
            total_pcm_bytes += len(pcm_data)
            dur = (len(pcm_data) // 2) / _TTS_CONFIG.sample_rate
            sentence_audio_durations.append(dur)
            print(f"  sentence: {remaining!r} → {dur:.2f}s audio")

        print(f"\n  Total sentences: {len(sentences)}")
        print(f"  Total audio: {sum(sentence_audio_durations):.2f}s")
        print(f"  Total PCM: {total_pcm_bytes} bytes")

        assert len(sentences) >= 2, f"Expected multiple sentences, got {len(sentences)}"
        assert total_pcm_bytes > 0, "No audio produced"
        assert all(d > 0.1 for d in sentence_audio_durations), "Some sentences produced too little audio"

    finally:
        await agent.disconnect()


async def test_tts_streaming_yields_chunks_incrementally() -> None:
    """Verify TTS yields multiple chunks (not all at once) for long text."""
    if not await _piper_available():
        pytest.skip("Piper not available")

    tts = PiperAdapter(_TTS_CONFIG)
    text = "Esta es una frase bastante larga que debería producir múltiples fragmentos de audio para verificar el streaming."

    chunk_count = 0
    chunk_sizes: list[int] = []
    async for pcm in tts.synthesize(text):
        chunk_count += 1
        chunk_sizes.append(len(pcm))

    print(f"\n  chunks: {chunk_count}, sizes: {chunk_sizes[:5]}{'...' if len(chunk_sizes) > 5 else ''}")
    assert chunk_count >= 2, f"Expected multiple TTS chunks, got {chunk_count}"


async def test_sentence_split_tts_sequential_consistency() -> None:
    """Compare full-text TTS vs sentence-split TTS — both should produce audio."""
    if not await _piper_available():
        pytest.skip("Piper not available")

    tts = PiperAdapter(_TTS_CONFIG)
    full_text = "Hola mundo. Esto es una prueba. Funciona correctamente."

    # Full text at once
    full_chunks = [c async for c in tts.synthesize(full_text)]
    full_pcm = b"".join(full_chunks)
    full_samples = len(full_pcm) // 2

    # Sentence by sentence
    sentences = ["Hola mundo", "Esto es una prueba", "Funciona correctamente."]
    split_total = 0
    for sentence in sentences:
        chunks = [c async for c in tts.synthesize(sentence)]
        split_total += sum(len(c) for c in chunks)
    split_samples = split_total // 2

    full_dur = full_samples / _TTS_CONFIG.sample_rate
    split_dur = split_samples / _TTS_CONFIG.sample_rate

    print(f"\n  full={full_dur:.2f}s, split={split_dur:.2f}s")

    # Both should produce meaningful audio (within 50% of each other)
    assert full_dur > 1.0, f"Full too short: {full_dur:.2f}s"
    assert split_dur > 1.0, f"Split too short: {split_dur:.2f}s"
    ratio = split_dur / full_dur
    assert 0.5 < ratio < 2.0, f"Duration ratio too different: {ratio:.2f}"
