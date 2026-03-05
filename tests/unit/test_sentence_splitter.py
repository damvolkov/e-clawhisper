"""Unit tests — sentence boundary splitting used by TurnPipeline streaming."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from e_clawhisper.daemon.adapters.agent import AgentAdapter
from e_clawhisper.daemon.adapters.stt import STTAdapter
from e_clawhisper.daemon.adapters.tts import TTSAdapter
from e_clawhisper.daemon.turn.pipeline import _SENTENCE_RE, TurnPipeline
from e_clawhisper.shared.settings import VADConfig

##### REGEX SPLITS #####


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world. How are you?", ["Hello world.", "How are you?"]),
        ("One! Two! Three!", ["One!", "Two!", "Three!"]),
        ("No punctuation here", ["No punctuation here"]),
        ("First.\nSecond.", ["First.", "Second."]),
        ("A. B. C.", ["A.", "B.", "C."]),
        ("Hello...", ["Hello..."]),
        ("Wait... Really? Yes!", ["Wait...", "Really?", "Yes!"]),
    ],
    ids=[
        "period_space",
        "exclamation",
        "no_split",
        "newline",
        "short_sentences",
        "ellipsis_no_split",
        "mixed_punctuation",
    ],
)
def test_sentence_regex_splits(text: str, expected: list[str]) -> None:
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    assert parts == expected


##### HELPERS #####


def _make_pipeline(agent: AgentAdapter) -> TurnPipeline:
    stt = AsyncMock(spec=STTAdapter)
    tts = AsyncMock(spec=TTSAdapter)
    vad_cfg = VADConfig(threshold=0.5, silence_duration=1.5, min_recording_time=1.0)
    return TurnPipeline(stt=stt, agent=agent, tts=tts, vad_config=vad_cfg)


async def _collect_sentences(pipeline: TurnPipeline, transcript: str) -> list[str]:
    await pipeline._stage_agent_to_sentences(transcript)

    sentences: list[str] = []
    while not pipeline._sentence_queue.empty():
        item = pipeline._sentence_queue.get_nowait()
        if item is None:
            break
        sentences.append(item)
    return sentences


##### STREAMING SENTENCE PRODUCER #####


async def test_producer_yields_sentences_from_chunks() -> None:
    """Simulate agent streaming text_delta → sentence queue."""
    agent = AsyncMock(spec=AgentAdapter)

    async def _fake_send(text: str):
        for chunk in ["Hello ", "world. ", "How are ", "you? ", "Fine."]:
            yield chunk

    agent.send = _fake_send
    pipeline = _make_pipeline(agent)

    sentences = await _collect_sentences(pipeline, "test")
    assert sentences == ["Hello world.", "How are you?", "Fine."]


async def test_producer_flushes_remaining_buffer() -> None:
    """Text without trailing punctuation is flushed as final sentence."""
    agent = AsyncMock(spec=AgentAdapter)

    async def _fake_send(text: str):
        for chunk in ["No punctuation ", "at the end"]:
            yield chunk

    agent.send = _fake_send
    pipeline = _make_pipeline(agent)

    sentences = await _collect_sentences(pipeline, "test")
    assert sentences == ["No punctuation at the end"]


async def test_producer_handles_newline_splits() -> None:
    """Newlines also trigger sentence boundaries."""
    agent = AsyncMock(spec=AgentAdapter)

    async def _fake_send(text: str):
        for chunk in ["Line one\n", "Line two\n", "Line three"]:
            yield chunk

    agent.send = _fake_send
    pipeline = _make_pipeline(agent)

    sentences = await _collect_sentences(pipeline, "test")
    assert sentences == ["Line one", "Line two", "Line three"]


async def test_producer_empty_response() -> None:
    """Empty agent response produces no sentences (only poison pill)."""
    agent = AsyncMock(spec=AgentAdapter)

    async def _fake_send(text: str):
        return
        yield  # make it an async generator

    agent.send = _fake_send
    pipeline = _make_pipeline(agent)

    sentences = await _collect_sentences(pipeline, "test")
    assert sentences == []
