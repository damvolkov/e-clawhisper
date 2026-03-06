"""Sentinel pipeline — passive listening loop.

3-layer parallel pipeline: Energy gate → [Silero VAD ‖ OpenWakeWord].
Runs until wake word detected, then signals orchestrator via asyncio.Event.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import numpy as np

from e_clawhisper.daemon.sentinel.vad import SileroVAD
from e_clawhisper.daemon.sentinel.wakeword import WakeWordDetector
from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.operational.events import WakewordEvent
from e_clawhisper.shared.settings import SentinelConfig


class SentinelPipeline:
    """Async listening loop: mic chunks → energy gate → [VAD ‖ OWW] → WakewordEvent."""

    __slots__ = (
        "_vad",
        "_ww",
        "_energy_floor",
        "_cooldown",
        "_executor",
        "_running",
        "_audio_queue",
        "wakeword_detected",
        "last_event",
    )

    def __init__(self, config: SentinelConfig) -> None:
        self._vad = SileroVAD(threshold=config.vad_threshold)
        self._ww = WakeWordDetector(
            model_name=config.wakeword.model,
            threshold=config.wakeword.threshold,
        )
        self._energy_floor = config.energy_floor
        self._cooldown = config.cooldown
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._running = False
        self._audio_queue: asyncio.Queue[np.ndarray] | None = None

        # Signal to orchestrator
        self.wakeword_detected = asyncio.Event()
        self.last_event: WakewordEvent | None = None

    @property
    def wakeword_name(self) -> str:
        return self._ww.name

    async def run(self, audio_queue: asyncio.Queue[np.ndarray]) -> None:
        """Main loop — reads from shared audio queue until wakeword or stopped."""
        self._audio_queue = audio_queue
        self._running = True
        self.wakeword_detected.clear()
        self.last_event = None
        self._ww.reset()
        loop = asyncio.get_running_loop()

        # Cooldown: discard audio frames to let speaker echo dissipate
        if self._cooldown > 0:
            deadline = monotonic() + self._cooldown
            while monotonic() < deadline and self._running:
                await audio_queue.get()

        logger.sentinel("DEFAULT", f"listening for '{self._ww.name}'")

        while self._running:
            audio = await audio_queue.get()
            energy = float(np.sqrt(np.mean(audio**2)))

            # OWW ALWAYS fed (needs continuous mel spectrogram)
            ww_future = loop.run_in_executor(self._executor, self._ww.feed, audio)

            if energy < self._energy_floor:
                await ww_future  # drain to keep OWW state advancing
                continue

            # Parallel: Silero VAD + OWW (already submitted)
            vad_future = loop.run_in_executor(self._executor, self._vad, audio)

            vad_prob = await vad_future
            ww_score = await ww_future

            if vad_prob < self._vad.threshold:
                logger.sentinel_debug("NOISE", vad=f"{vad_prob:.2f}", e=f"{energy:.4f}")
                continue

            logger.sentinel_debug("VOICE", vad=f"{vad_prob:.2f}", e=f"{energy:.4f}")

            if ww_score >= self._ww.threshold:
                logger.sentinel("WAKEWORD", f"'{self._ww.name}' detected", ww=f"{ww_score:.2f}")
                self.last_event = WakewordEvent(
                    timestamp=monotonic(),
                    confidence=ww_score,
                    energy=energy,
                )
                self.wakeword_detected.set()
                return

    async def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
