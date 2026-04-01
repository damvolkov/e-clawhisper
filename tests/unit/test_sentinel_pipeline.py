"""Tests for SentinelPipeline — init, properties, cooldown, stop, run, wait_for_voice."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np

from e_heed.daemon.sentinel.pipeline import SentinelPipeline
from e_heed.shared.settings import SentinelConfig

##### FIXTURES #####


class _StubVAD:
    """Thread-safe VAD stub returning a fixed probability."""

    __slots__ = ("threshold", "prob")

    def __init__(self, threshold: float, prob: float = 0.0) -> None:
        self.threshold = threshold
        self.prob = prob

    def __call__(self, audio: np.ndarray) -> float:
        return self.prob

    def reset(self) -> None:
        pass


class _StubWW:
    """Thread-safe wakeword stub returning a fixed score."""

    __slots__ = ("name", "threshold", "score")

    def __init__(self, name: str, threshold: float, score: float = 0.0) -> None:
        self.name = name
        self.threshold = threshold
        self.score = score

    def feed(self, audio: np.ndarray) -> float:
        return self.score

    def reset(self) -> None:
        pass


@patch("e_heed.daemon.sentinel.pipeline.WakeWordDetector")
@patch("e_heed.daemon.sentinel.pipeline.SileroVAD")
def _make_pipeline(
    mock_vad_cls: MagicMock,
    mock_ww_cls: MagicMock,
    *,
    energy_floor: float = 0.01,
    vad_threshold: float = 0.5,
    ww_threshold: float = 0.5,
    vad_prob: float = 0.0,
    ww_score: float = 0.0,
) -> SentinelPipeline:
    cfg = SentinelConfig(energy_floor=energy_floor, vad_threshold=vad_threshold, cooldown=0.001)
    cfg.wakeword.threshold = ww_threshold

    mock_vad_cls.return_value = _StubVAD(vad_threshold, vad_prob)
    mock_ww_cls.return_value = _StubWW("alexa", ww_threshold, ww_score)

    pipeline = SentinelPipeline(cfg)
    # Replace thread-safe stubs after construction (since __init__ creates them)
    pipeline._vad = _StubVAD(vad_threshold, vad_prob)
    pipeline._ww = _StubWW("alexa", ww_threshold, ww_score)
    return pipeline


def _silent_chunk() -> np.ndarray:
    return np.zeros(512, dtype=np.float32)


def _loud_chunk(val: float = 0.5) -> np.ndarray:
    return np.full(512, val, dtype=np.float32)


##### INIT #####


def test_init_properties() -> None:
    pipeline = _make_pipeline()
    assert pipeline.wakeword_name == "alexa"
    assert pipeline._running is False
    assert pipeline.last_event is None


def test_init_wakeword_event_clear() -> None:
    pipeline = _make_pipeline()
    assert not pipeline.wakeword_detected.is_set()


##### STOP #####


async def test_stop_sets_flag() -> None:
    pipeline = _make_pipeline()
    pipeline._running = True
    await pipeline.stop()
    assert pipeline._running is False


##### DRAIN COOLDOWN #####


async def test_drain_cooldown_consumes_frames() -> None:
    pipeline = _make_pipeline()
    pipeline._running = True
    q: asyncio.Queue[np.ndarray] = asyncio.Queue()

    async def _feeder() -> None:
        for _ in range(50):
            await q.put(_silent_chunk())
            await asyncio.sleep(0.0001)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_feeder())
        tg.create_task(pipeline._drain_cooldown(q))


async def test_drain_cooldown_stops_on_running_false() -> None:
    pipeline = _make_pipeline()
    pipeline._running = False
    q: asyncio.Queue[np.ndarray] = asyncio.Queue()
    await q.put(_silent_chunk())
    await pipeline._drain_cooldown(q)


##### RUN — FULL PIPELINE #####


async def test_run_wakeword_detected() -> None:
    pipeline = _make_pipeline(energy_floor=0.001, vad_prob=0.9, ww_score=0.8, ww_threshold=0.5)

    q: asyncio.Queue[np.ndarray] = asyncio.Queue()

    async def _feeder() -> None:
        for _ in range(100):
            await q.put(_loud_chunk(0.5))
            await asyncio.sleep(0.001)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_feeder())
        tg.create_task(pipeline.run(q))

    assert pipeline.last_event is not None
    assert pipeline.wakeword_detected.is_set()


async def test_run_silence_below_energy_floor() -> None:
    pipeline = _make_pipeline(energy_floor=0.9, vad_prob=0.0, ww_score=0.0)

    q: asyncio.Queue[np.ndarray] = asyncio.Queue()
    for _ in range(5):
        await q.put(np.full(512, 0.01, dtype=np.float32))

    async def _stop_later() -> None:
        await asyncio.sleep(0.05)
        pipeline._running = False
        await q.put(_silent_chunk())

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_stop_later())
        tg.create_task(pipeline.run(q))

    assert pipeline.last_event is None


async def test_run_noise_below_vad_threshold() -> None:
    pipeline = _make_pipeline(energy_floor=0.001, vad_prob=0.2, ww_score=0.0, vad_threshold=0.5)

    q: asyncio.Queue[np.ndarray] = asyncio.Queue()
    for _ in range(5):
        await q.put(_loud_chunk(0.5))

    async def _stop_later() -> None:
        await asyncio.sleep(0.05)
        pipeline._running = False
        await q.put(_loud_chunk(0.5))

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_stop_later())
        tg.create_task(pipeline.run(q))

    assert pipeline.last_event is None


async def test_run_voice_but_no_wakeword() -> None:
    pipeline = _make_pipeline(energy_floor=0.001, vad_prob=0.9, ww_score=0.1, ww_threshold=0.5)

    q: asyncio.Queue[np.ndarray] = asyncio.Queue()
    for _ in range(5):
        await q.put(_loud_chunk(0.5))

    async def _stop_later() -> None:
        await asyncio.sleep(0.05)
        pipeline._running = False
        await q.put(_loud_chunk(0.5))

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_stop_later())
        tg.create_task(pipeline.run(q))

    assert pipeline.last_event is None


##### WAIT FOR VOICE — PARTIAL PIPELINE #####


async def test_wait_for_voice_detected() -> None:
    pipeline = _make_pipeline(energy_floor=0.001, vad_prob=0.9, vad_threshold=0.5)

    q: asyncio.Queue[np.ndarray] = asyncio.Queue()

    async def _feeder() -> None:
        for _ in range(100):
            await q.put(_loud_chunk(0.5))
            await asyncio.sleep(0.001)

    result = False

    async def _run() -> None:
        nonlocal result
        result = await pipeline.wait_for_voice(q, timeout=2.0)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_feeder())
        tg.create_task(_run())

    assert result is True


async def test_wait_for_voice_timeout_silence() -> None:
    pipeline = _make_pipeline(energy_floor=0.5)

    q: asyncio.Queue[np.ndarray] = asyncio.Queue()

    async def _feeder() -> None:
        for _ in range(100):
            await q.put(_silent_chunk())
            await asyncio.sleep(0.001)

    result = True

    async def _run() -> None:
        nonlocal result
        result = await pipeline.wait_for_voice(q, timeout=0.05)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_feeder())
        tg.create_task(_run())

    assert result is False


async def test_wait_for_voice_noise_below_vad() -> None:
    pipeline = _make_pipeline(energy_floor=0.001, vad_prob=0.2, vad_threshold=0.5)

    q: asyncio.Queue[np.ndarray] = asyncio.Queue()

    async def _feeder() -> None:
        for _ in range(100):
            await q.put(_loud_chunk(0.5))
            await asyncio.sleep(0.001)

    result = True

    async def _run() -> None:
        nonlocal result
        result = await pipeline.wait_for_voice(q, timeout=0.05)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_feeder())
        tg.create_task(_run())

    assert result is False
