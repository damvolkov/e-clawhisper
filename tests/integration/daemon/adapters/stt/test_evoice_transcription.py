"""Integration tests for EVoice STT adapter — requires running e-voice at localhost:45140."""

from __future__ import annotations

import pytest

##### ADAPTER LIFECYCLE #####


@pytest.mark.slow
async def test_evoice_stt_adapter_connect_disconnect() -> None:
    """Verify adapter lifecycle: connect, stream, finish, disconnect."""
    from e_heed.daemon.adapters.stt.evoice import EVoiceSTTAdapter
    from e_heed.shared.settings import EVoiceSTTConfig

    config = EVoiceSTTConfig(url="http://localhost:45140", language="en")
    adapter = EVoiceSTTAdapter(config)

    await adapter.connect()
    await adapter.start_utterance()
    # Send silence (512 samples of PCM16 zeros = 1024 bytes)
    await adapter.stream(b"\x00" * 1024)
    transcript = await adapter.finish_utterance()
    await adapter.disconnect()

    assert isinstance(transcript, str)


@pytest.mark.slow
async def test_evoice_stt_adapter_multiple_utterances() -> None:
    """Verify adapter handles multiple consecutive utterances."""
    from e_heed.daemon.adapters.stt.evoice import EVoiceSTTAdapter
    from e_heed.shared.settings import EVoiceSTTConfig

    config = EVoiceSTTConfig(url="http://localhost:45140", language="en")
    adapter = EVoiceSTTAdapter(config)

    await adapter.connect()
    for _ in range(3):
        await adapter.start_utterance()
        await adapter.stream(b"\x00" * 1024)
        await adapter.finish_utterance()
    await adapter.disconnect()


##### HTTP TRANSCRIPTION #####


@pytest.mark.slow
async def test_evoice_stt_http_transcription() -> None:
    """Transcribe via HTTP POST with a WAV file."""

    import httpx
    from pytest_audioeval.samples.registry import SampleRegistry

    registry = SampleRegistry()
    sample = registry.en_hello_world

    async with httpx.AsyncClient(base_url="http://localhost:45140", timeout=30.0) as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", sample.audio_bytes(), "audio/wav")},
            data={"language": "en", "response_format": "json"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert len(data["text"]) > 0
