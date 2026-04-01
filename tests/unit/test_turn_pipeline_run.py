"""Tests for TurnPipeline.run — listen phase, empty transcript, stop, drain."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from e_clawhisper.daemon.adapters.audio import AudioAdapter
from e_clawhisper.daemon.adapters.base import AgentPort, STTPort, TTSPort
from e_clawhisper.daemon.turn.pipeline import TurnPipeline
from e_clawhisper.daemon.turn.vad import EndOfSpeechResult
from e_clawhisper.shared.operational.events import TurnComplete, TurnError
from e_clawhisper.shared.settings import VADConfig

_VAD_CFG = VADConfig(threshold=0.5, silence_duration=1.5, min_recording_time=1.0)
_PCM_CHUNK = b"\x00\x01" * 100


##### HELPERS #####


def _make_stt(transcript: str = "hello world") -> STTPort:
    stt = AsyncMock(spec=STTPort)
    stt.finish_utterance = AsyncMock(return_value=transcript)
    stt.start_utterance = AsyncMock()
    stt.stream = AsyncMock()
    return stt


def _make_agent(chunks: list[str]) -> AgentPort:
    agent = AsyncMock(spec=AgentPort)

    async def _fake_send(text: str):
        for c in chunks:
            yield c

    agent.send = _fake_send
    return agent


def _make_tts() -> TTSPort:
    tts = AsyncMock(spec=TTSPort)

    async def _fake_synth(text: str):
        yield _PCM_CHUNK

    tts.synthesize = _fake_synth
    return tts


def _make_audio_with_vad_stop() -> AudioAdapter:
    """Audio adapter that provides one chunk, then pipeline stops via VAD."""
    audio = MagicMock(spec=AudioAdapter)
    q: asyncio.Queue[np.ndarray] = asyncio.Queue()
    q.put_nowait(np.zeros(512, dtype=np.float32))
    audio.queue = q

    async def _fake_play(pcm_queue: asyncio.Queue[bytes | None], sample_rate: int) -> float:
        total = 0
        while (chunk := await pcm_queue.get()) is not None:
            total += len(chunk) // 2
        return total / sample_rate

    audio.play = _fake_play
    return audio


##### EMPTY TRANSCRIPT #####


@patch("e_clawhisper.daemon.turn.pipeline.EndOfSpeechDetector")
async def test_run_empty_transcript_returns_error(mock_vad_cls: MagicMock) -> None:
    mock_vad = MagicMock()
    mock_vad.process.return_value = EndOfSpeechResult(is_speech=False, should_stop=True, probability=0.1)
    mock_vad.reset = MagicMock()
    mock_vad_cls.return_value = mock_vad

    stt = _make_stt(transcript="  ")
    agent = _make_agent(["response"])
    tts = _make_tts()
    pipeline = TurnPipeline(stt=stt, agent=agent, tts=tts, vad_config=_VAD_CFG, tts_sample_rate=16000)
    audio = _make_audio_with_vad_stop()

    result = await pipeline.run(audio)
    assert isinstance(result, TurnError)


##### SUCCESSFUL TURN #####


@patch("e_clawhisper.daemon.turn.pipeline.EndOfSpeechDetector")
async def test_run_successful_turn(mock_vad_cls: MagicMock) -> None:
    mock_vad = MagicMock()
    mock_vad.process.return_value = EndOfSpeechResult(is_speech=True, should_stop=True, probability=0.9)
    mock_vad.reset = MagicMock()
    mock_vad_cls.return_value = mock_vad

    stt = _make_stt(transcript="hello world")
    agent = _make_agent(["Hi there!"])
    tts = _make_tts()
    pipeline = TurnPipeline(stt=stt, agent=agent, tts=tts, vad_config=_VAD_CFG, tts_sample_rate=16000)
    audio = _make_audio_with_vad_stop()

    result = await pipeline.run(audio)
    assert isinstance(result, TurnComplete)
    assert result.transcript == "hello world"


##### EMPTY RESPONSE #####


@patch("e_clawhisper.daemon.turn.pipeline.EndOfSpeechDetector")
async def test_run_empty_response_returns_error(mock_vad_cls: MagicMock) -> None:
    mock_vad = MagicMock()
    mock_vad.process.return_value = EndOfSpeechResult(is_speech=True, should_stop=True, probability=0.9)
    mock_vad.reset = MagicMock()
    mock_vad_cls.return_value = mock_vad

    stt = _make_stt(transcript="some text")
    agent = _make_agent([])  # empty response
    tts = _make_tts()
    pipeline = TurnPipeline(stt=stt, agent=agent, tts=tts, vad_config=_VAD_CFG, tts_sample_rate=16000)
    audio = _make_audio_with_vad_stop()

    result = await pipeline.run(audio)
    assert isinstance(result, TurnError)


##### EXCEPTION DURING RUN #####


@patch("e_clawhisper.daemon.turn.pipeline.EndOfSpeechDetector")
async def test_run_catches_exception_as_turn_error(mock_vad_cls: MagicMock) -> None:
    mock_vad = MagicMock()
    mock_vad.process.side_effect = RuntimeError("vad boom")
    mock_vad.reset = MagicMock()
    mock_vad_cls.return_value = mock_vad

    stt = _make_stt()
    agent = _make_agent([])
    tts = _make_tts()
    pipeline = TurnPipeline(stt=stt, agent=agent, tts=tts, vad_config=_VAD_CFG, tts_sample_rate=16000)
    audio = _make_audio_with_vad_stop()

    result = await pipeline.run(audio)
    assert isinstance(result, TurnError)


##### STOP #####


async def test_stop_sets_flag_and_shuts_executor() -> None:
    pipeline = TurnPipeline(
        stt=AsyncMock(spec=STTPort),
        agent=AsyncMock(spec=AgentPort),
        tts=AsyncMock(spec=TTSPort),
        vad_config=_VAD_CFG,
    )
    await pipeline.stop()
    assert pipeline._running is False


##### DRAIN QUEUES #####


async def test_drain_queues_clears_both() -> None:
    pipeline = TurnPipeline(
        stt=AsyncMock(spec=STTPort),
        agent=AsyncMock(spec=AgentPort),
        tts=AsyncMock(spec=TTSPort),
        vad_config=_VAD_CFG,
    )
    await pipeline._sentence_queue.put("test")
    await pipeline._pcm_queue.put(b"data")

    pipeline.drain_queues()

    assert pipeline._sentence_queue.empty()
    assert pipeline._pcm_queue.empty()


##### PROPERTIES #####


async def test_queue_properties() -> None:
    pipeline = TurnPipeline(
        stt=AsyncMock(spec=STTPort),
        agent=AsyncMock(spec=AgentPort),
        tts=AsyncMock(spec=TTSPort),
        vad_config=_VAD_CFG,
    )
    assert pipeline.sentence_queue is pipeline._sentence_queue
    assert pipeline.pcm_queue is pipeline._pcm_queue
