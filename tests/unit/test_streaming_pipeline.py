"""Unit tests — 3-stage streaming pipeline: Agent → sentences → TTS → pcm → speaker."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from e_clawhisper.daemon.adapters.agent import AgentAdapter
from e_clawhisper.daemon.adapters.audio import AudioAdapter
from e_clawhisper.daemon.adapters.stt import STTAdapter
from e_clawhisper.daemon.adapters.tts import TTSAdapter
from e_clawhisper.daemon.turn.pipeline import TurnPipeline
from e_clawhisper.shared.settings import VADConfig

_VAD_CFG = VADConfig(threshold=0.5, silence_duration=1.5, min_recording_time=1.0)
_PCM_CHUNK = b"\x00\x01" * 100
_DEADLOCK_TIMEOUT = 3.0


##### HELPERS #####


def _make_agent(chunks: list[str]) -> AgentAdapter:
    agent = AsyncMock(spec=AgentAdapter)

    async def _fake_send(text: str):
        for c in chunks:
            yield c

    agent.send = _fake_send
    return agent


def _make_tts(pcm_per_sentence: int = 2) -> TTSAdapter:
    tts = AsyncMock(spec=TTSAdapter)

    async def _fake_synthesize(text: str):
        for _ in range(pcm_per_sentence):
            yield _PCM_CHUNK

    tts.synthesize = _fake_synthesize
    return tts


def _make_pipeline(agent: AgentAdapter, tts: TTSAdapter) -> TurnPipeline:
    stt = AsyncMock(spec=STTAdapter)
    return TurnPipeline(stt=stt, agent=agent, tts=tts, vad_config=_VAD_CFG, tts_sample_rate=16000)


def _make_audio() -> AudioAdapter:
    """Fake AudioAdapter that drains pcm_queue without actual playback."""
    audio = MagicMock(spec=AudioAdapter)

    async def _fake_play(pcm_queue: asyncio.Queue[bytes | None], sample_rate: int) -> float:
        total = 0
        while (chunk := await pcm_queue.get()) is not None:
            total += len(chunk) // 2
        return total / sample_rate

    audio.play = _fake_play
    return audio


##### STAGE: TTS TO PCM #####


async def test_stage_tts_to_pcm_produces_chunks() -> None:
    """TTS stage reads sentences from instance queue and produces PCM chunks."""
    tts = _make_tts(pcm_per_sentence=3)
    pipeline = _make_pipeline(_make_agent([]), tts)
    pipeline._running = True

    await pipeline._sentence_queue.put("Hello world.")
    await pipeline._sentence_queue.put("Second sentence.")
    await pipeline._sentence_queue.put(None)

    parts: list[str] = []
    await pipeline._stage_tts_to_pcm(parts)

    assert parts == ["Hello world.", "Second sentence."]

    pcm_chunks: list[bytes] = []
    while not pipeline._pcm_queue.empty():
        item = pipeline._pcm_queue.get_nowait()
        if item is None:
            break
        pcm_chunks.append(item)

    assert len(pcm_chunks) == 6  # 3 per sentence * 2 sentences


async def test_stage_tts_to_pcm_sends_poison_pill() -> None:
    """TTS stage always sends None terminator even with empty input."""
    tts = _make_tts()
    pipeline = _make_pipeline(_make_agent([]), tts)
    pipeline._running = True

    await pipeline._sentence_queue.put(None)
    await pipeline._stage_tts_to_pcm([])

    assert pipeline._pcm_queue.get_nowait() is None


async def test_stage_tts_to_pcm_respects_running_flag() -> None:
    """TTS stage stops when _running is False."""
    tts = _make_tts()
    pipeline = _make_pipeline(_make_agent([]), tts)
    pipeline._running = False

    await pipeline._sentence_queue.put("Should be skipped.")
    await pipeline._sentence_queue.put(None)

    parts: list[str] = []
    await pipeline._stage_tts_to_pcm(parts)

    assert parts == []
    assert pipeline._pcm_queue.get_nowait() is None


##### STREAM RESPONSE (3-STAGE) #####


async def test_stream_response_produces_full_text() -> None:
    """3-stage pipeline assembles response from agent chunks."""
    agent = _make_agent(["Hello world. ", "How are you?"])
    tts = _make_tts(pcm_per_sentence=1)
    pipeline = _make_pipeline(agent, tts)
    audio = _make_audio()

    pipeline._running = True
    result = await pipeline._stream_response("test query", audio)

    assert "Hello world." in result
    assert "How are you?" in result


async def test_stream_response_returns_empty_on_no_agent_output() -> None:
    """Empty agent output produces empty response."""
    agent = _make_agent([])
    tts = _make_tts()
    pipeline = _make_pipeline(agent, tts)
    audio = _make_audio()

    pipeline._running = True
    result = await pipeline._stream_response("test", audio)

    assert result == ""


async def test_stream_response_handles_single_sentence_no_punctuation() -> None:
    """Single chunk without punctuation is flushed as response."""
    agent = _make_agent(["Just one sentence"])
    tts = _make_tts(pcm_per_sentence=1)
    pipeline = _make_pipeline(agent, tts)
    audio = _make_audio()

    pipeline._running = True
    result = await pipeline._stream_response("test", audio)

    assert result == "Just one sentence"


async def test_stream_response_concurrent_execution() -> None:
    """Verify all 3 stages run as concurrent tasks (not sequential)."""
    execution_order: list[str] = []

    agent = AsyncMock(spec=AgentAdapter)

    async def _slow_agent(text: str):
        execution_order.append("agent_start")
        yield "Hello. "
        await asyncio.sleep(0.01)
        yield "World."
        execution_order.append("agent_end")

    agent.send = _slow_agent

    tts = AsyncMock(spec=TTSAdapter)

    async def _slow_tts(text: str):
        execution_order.append(f"tts_{text[:5]}")
        yield _PCM_CHUNK

    tts.synthesize = _slow_tts

    pipeline = _make_pipeline(agent, tts)
    pipeline._tts = tts

    audio = _make_audio()
    pipeline._running = True

    await pipeline._stream_response("test", audio)

    assert "agent_start" in execution_order
    assert "agent_end" in execution_order
    assert any(e.startswith("tts_") for e in execution_order)


##### RACE CONDITIONS & DEADLOCKS #####


async def test_agent_failure_does_not_deadlock() -> None:
    """Agent error mid-stream must cancel TTS+speaker — no infinite wait."""
    agent = AsyncMock(spec=AgentAdapter)

    async def _failing_agent(text: str):
        yield "Hello. "
        raise RuntimeError("agent_crashed")

    agent.send = _failing_agent

    tts = _make_tts(pcm_per_sentence=1)
    pipeline = _make_pipeline(agent, tts)
    audio = _make_audio()
    pipeline._running = True

    with pytest.raises(RuntimeError, match="agent_crashed"):
        async with asyncio.timeout(_DEADLOCK_TIMEOUT):
            await pipeline._stream_response("test", audio)


async def test_tts_failure_does_not_deadlock() -> None:
    """TTS error during synthesis must cancel agent+speaker — no infinite wait."""
    agent = _make_agent(["First sentence. ", "Second sentence. ", "Third."])

    tts = AsyncMock(spec=TTSAdapter)
    call_count = 0

    async def _failing_tts(text: str):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise ConnectionError("tts_down")
        yield _PCM_CHUNK

    tts.synthesize = _failing_tts

    pipeline = _make_pipeline(agent, tts)
    audio = _make_audio()
    pipeline._running = True

    with pytest.raises(ConnectionError, match="tts_down"):
        async with asyncio.timeout(_DEADLOCK_TIMEOUT):
            await pipeline._stream_response("test", audio)


async def test_speaker_failure_does_not_deadlock() -> None:
    """Speaker error must cancel agent+tts — no infinite wait."""
    agent = _make_agent(["Hello world. ", "More text."])
    tts = _make_tts(pcm_per_sentence=1)
    pipeline = _make_pipeline(agent, tts)

    audio = MagicMock(spec=AudioAdapter)

    async def _failing_play(pcm_queue: asyncio.Queue[bytes | None], sample_rate: int) -> float:
        await pcm_queue.get()
        raise OSError("speaker_broken")

    audio.play = _failing_play
    pipeline._running = True

    with pytest.raises(OSError, match="speaker_broken"):
        async with asyncio.timeout(_DEADLOCK_TIMEOUT):
            await pipeline._stream_response("test", audio)


async def test_agent_failure_propagates_through_run() -> None:
    """Agent crash during run() returns TurnError (caught by run's handler), not hang."""
    agent = AsyncMock(spec=AgentAdapter)

    async def _crashing_agent(text: str):
        yield "partial "
        raise ConnectionError("ws_closed")

    agent.send = _crashing_agent

    tts = _make_tts(pcm_per_sentence=1)
    stt = AsyncMock(spec=STTAdapter)
    stt.finish_utterance = AsyncMock(return_value="user said something")
    stt.start_utterance = AsyncMock()
    stt.stream = AsyncMock()

    pipeline = TurnPipeline(stt=stt, agent=agent, tts=tts, vad_config=_VAD_CFG, tts_sample_rate=16000)

    audio = MagicMock(spec=AudioAdapter)
    # Provide a queue with one VAD-triggering chunk, then stop
    audio.queue = asyncio.Queue()

    async def _fake_play(pcm_queue: asyncio.Queue[bytes | None], sample_rate: int) -> float:
        while (await pcm_queue.get()) is not None:
            pass
        return 0.0

    audio.play = _fake_play

    # _stream_response raises the ConnectionError — run() catches it as TurnError
    with pytest.raises(ConnectionError, match="ws_closed"):
        async with asyncio.timeout(_DEADLOCK_TIMEOUT):
            await pipeline._stream_response("user said something", audio)


async def test_all_stages_cancelled_on_first_failure() -> None:
    """Verify TaskGroup cancels sibling tasks when one fails."""
    cancelled_stages: list[str] = []

    agent = AsyncMock(spec=AgentAdapter)

    async def _slow_agent(text: str):
        try:
            yield "Hello. "
            await asyncio.sleep(10)  # Simulates slow streaming
            yield "Never reached."
        except asyncio.CancelledError:
            cancelled_stages.append("agent")
            raise

    agent.send = _slow_agent

    tts = AsyncMock(spec=TTSAdapter)

    async def _ok_tts(text: str):
        yield _PCM_CHUNK

    tts.synthesize = _ok_tts

    pipeline = _make_pipeline(agent, tts)

    audio = MagicMock(spec=AudioAdapter)

    async def _crashing_play(pcm_queue: asyncio.Queue[bytes | None], sample_rate: int) -> float:
        await pcm_queue.get()  # get one chunk
        raise OSError("alsa_xrun")

    audio.play = _crashing_play
    pipeline._running = True

    with pytest.raises(OSError, match="alsa_xrun"):
        async with asyncio.timeout(_DEADLOCK_TIMEOUT):
            await pipeline._stream_response("test", audio)

    assert "agent" in cancelled_stages
