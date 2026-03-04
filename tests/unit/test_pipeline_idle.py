"""Tests for pipeline IDLE → wake word → STREAMING flow.

All external adapters (STT, TTS, Agent, AudioDevice) are mocked.
VAD is mocked to control speech/silence transitions deterministically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from e_clawhisper.daemon.core.models import VADResult
from e_clawhisper.daemon.core.processors.turn_manager import TurnManager
from e_clawhisper.daemon.core.processors.wake_word import WakeWordDetector
from e_clawhisper.daemon.pipeline.runner import PipelineRunner
from e_clawhisper.daemon.pipeline.states import ConversationMode, PipelineState

_CHUNK = np.zeros(512, dtype=np.int16)


##### FIXTURES #####


def _vad_speech() -> VADResult:
    return VADResult(is_speech=True, should_stop=False, probability=0.9)


def _vad_silence() -> VADResult:
    return VADResult(is_speech=False, should_stop=False, probability=0.1)


def _vad_stop() -> VADResult:
    return VADResult(is_speech=False, should_stop=True, probability=0.1)


def _make_runner(
    vad_sequence: list[VADResult],
    stt_transcript: str = "",
) -> PipelineRunner:
    """Build a PipelineRunner with mocked components and a finite VAD sequence."""
    call_count = 0
    max_calls = len(vad_sequence)
    runner_ref: list[PipelineRunner] = []

    audio = AsyncMock()

    async def _read_chunk() -> np.ndarray:
        nonlocal call_count
        if call_count >= max_calls:
            runner_ref[0]._running = False
            return _CHUNK
        call_count += 1
        return _CHUNK

    audio.read_chunk = _read_chunk
    audio.start = AsyncMock()
    audio.stop = AsyncMock()
    audio.play_audio = MagicMock()
    audio.stop_playback = MagicMock()

    vad = MagicMock()
    seq_iter = iter(vad_sequence)
    vad.process.side_effect = lambda chunk: next(seq_iter, _vad_silence())
    vad.reset = MagicMock()

    stt = AsyncMock()
    stt.start_utterance = AsyncMock()
    stt.finish_utterance = AsyncMock(return_value=stt_transcript)
    stt.stream_audio = AsyncMock()
    stt.disconnect = AsyncMock()

    async def _mock_agent_response(text: str) -> AsyncIterator[str]:
        yield "Mocked response."

    agent = AsyncMock()
    agent.send_message = _mock_agent_response
    agent.disconnect = AsyncMock()

    async def _mock_synth(text: str) -> AsyncIterator[bytes]:
        yield b"\x00" * 100

    tts = AsyncMock()
    tts.synthesize = _mock_synth
    tts.stop = AsyncMock()

    wake = WakeWordDetector(wake_word="damien")
    turn = TurnManager(conversation_timeout=30.0)

    runner = PipelineRunner(
        audio_device=audio,
        vad=vad,
        stt=stt,
        tts=tts,
        agent=agent,
        wake_word=wake,
        turn_manager=turn,
        pre_roll_capacity=32000,
        tts_sample_rate=22050,
    )
    runner_ref.append(runner)
    return runner


##### IDLE — NO SPEECH #####


async def test_idle_stays_idle_on_silence() -> None:
    runner = _make_runner([_vad_silence(), _vad_silence()])
    await runner.start()

    assert runner.state == PipelineState.IDLE
    assert runner.conversation_mode == ConversationMode.IDLE


##### IDLE — SPEECH WITHOUT WAKE WORD #####


async def test_idle_speech_no_wake_word_stays_idle() -> None:
    runner = _make_runner(
        [_vad_speech(), _vad_stop()],
        stt_transcript="hello world how are you",
    )
    await runner.start()

    assert runner.state == PipelineState.IDLE
    assert runner.conversation_mode == ConversationMode.IDLE


##### IDLE — WAKE WORD DETECTED #####


async def test_wake_word_activates_conversation() -> None:
    runner = _make_runner(
        [_vad_speech(), _vad_stop()],
        stt_transcript="hey damien what time is it",
    )
    await runner.start()

    assert runner.conversation_mode == ConversationMode.ACTIVE


##### IDLE — WAKE WORD ONLY (NO QUERY) #####


async def test_wake_word_only_responds_listening() -> None:
    runner = _make_runner(
        [_vad_speech(), _vad_stop()],
        stt_transcript="damien",
    )
    await runner.start()

    assert runner.conversation_mode == ConversationMode.ACTIVE


##### STT LIFECYCLE #####


async def test_stt_opens_on_speech() -> None:
    runner = _make_runner([_vad_speech()])
    await runner.start()

    runner._stt.start_utterance.assert_awaited_once()


async def test_stt_not_opened_on_silence() -> None:
    runner = _make_runner([_vad_silence(), _vad_silence()])
    await runner.start()

    runner._stt.start_utterance.assert_not_awaited()


async def test_stt_finalized_on_should_stop() -> None:
    runner = _make_runner(
        [_vad_speech(), _vad_stop()],
        stt_transcript="random words",
    )
    await runner.start()

    runner._stt.finish_utterance.assert_awaited_once()


##### AGENT INTERACTION #####


async def test_agent_receives_query_after_wake_word() -> None:
    calls: list[str] = []

    async def _tracking_agent(text: str) -> AsyncIterator[str]:
        calls.append(text)
        yield "ok"

    runner = _make_runner(
        [_vad_speech(), _vad_stop()],
        stt_transcript="damien what is the weather",
    )
    runner._agent.send_message = _tracking_agent
    await runner.start()

    assert len(calls) == 1
    assert "weather" in calls[0]
    assert "damien" not in calls[0]
