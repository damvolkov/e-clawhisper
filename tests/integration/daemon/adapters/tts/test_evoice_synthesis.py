"""Integration tests for EVoice TTS adapter — requires running e-voice at localhost:45140."""

from __future__ import annotations

import pytest

##### ADAPTER SYNTHESIS #####


@pytest.mark.slow
async def test_evoice_tts_adapter_synthesize() -> None:
    """Verify adapter produces audio chunks for text input."""
    from e_heed.daemon.adapters.tts.evoice import EVoiceTTSAdapter
    from e_heed.shared.settings import EVoiceTTSConfig

    config = EVoiceTTSConfig(url="http://localhost:45140")
    adapter = EVoiceTTSAdapter(config)

    chunks: list[bytes] = []
    async for chunk in adapter.synthesize("This is a test sentence."):
        chunks.append(chunk)

    await adapter.disconnect()

    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 0, "Expected PCM audio data"
    assert adapter.sample_rate == 24000


@pytest.mark.slow
async def test_evoice_tts_adapter_multiple_syntheses() -> None:
    """Verify adapter handles multiple consecutive syntheses on same connection."""
    from e_heed.daemon.adapters.tts.evoice import EVoiceTTSAdapter
    from e_heed.shared.settings import EVoiceTTSConfig

    config = EVoiceTTSConfig(url="http://localhost:45140")
    adapter = EVoiceTTSAdapter(config)

    for text in ("Hello.", "How are you?", "Goodbye."):
        chunks: list[bytes] = []
        async for chunk in adapter.synthesize(text):
            chunks.append(chunk)
        assert sum(len(c) for c in chunks) > 0

    await adapter.disconnect()


@pytest.mark.slow
async def test_evoice_tts_adapter_stop() -> None:
    """Verify stop interrupts synthesis."""
    from e_heed.daemon.adapters.tts.evoice import EVoiceTTSAdapter
    from e_heed.shared.settings import EVoiceTTSConfig

    config = EVoiceTTSConfig(url="http://localhost:45140")
    adapter = EVoiceTTSAdapter(config)

    chunk_count = 0
    async for _ in adapter.synthesize("A very long sentence that should produce many chunks of audio data."):
        chunk_count += 1
        if chunk_count >= 2:
            await adapter.stop()
            break

    await adapter.disconnect()
    assert chunk_count >= 1


##### HTTP SYNTHESIS #####


@pytest.mark.slow
async def test_evoice_tts_http_synthesis() -> None:
    """Synthesize via HTTP POST and verify audio response."""
    import httpx

    async with httpx.AsyncClient(base_url="http://localhost:45140", timeout=30.0) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello world.",
                "model": "kokoro",
                "voice": "af_heart",
                "response_format": "pcm",
                "stream": False,
            },
        )
    assert response.status_code == 200
    assert len(response.content) > 0
