"""Sentinel pipeline — passive listening with layered architecture.

3-layer parallel pipeline: Energy gate → [Silero VAD ‖ OpenWakeWord].
Layers can be used independently:
  - run()            → full pipeline (layers 1-2-3): energy → VAD → wakeword
  - wait_for_voice() → partial pipeline (layers 1-2): energy → VAD only
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import numpy as np

from e_heed.daemon.sentinel.vad import SileroVAD
from e_heed.daemon.sentinel.wakeword import WakeWordDetector
from e_heed.shared.logger import logger
from e_heed.shared.operational.events import WakewordEvent
from e_heed.shared.settings import SentinelConfig


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

    ##### FULL PIPELINE (LAYERS 1-2-3) #####

    async def run(self, audio_queue: asyncio.Queue[np.ndarray]) -> None:
        """Full pipeline — energy gate → VAD → wakeword. Blocks until wakeword or stopped."""
        self._audio_queue = audio_queue
        self._running = True
        self.wakeword_detected.clear()
        self.last_event = None
        self._ww.reset()
        loop = asyncio.get_running_loop()

        await self._drain_cooldown(audio_queue)

        logger.sentinel("DEFAULT", f"listening for '{self._ww.name}'")

        while self._running:
            audio = await audio_queue.get()
            energy = float(np.sqrt(np.mean(audio**2)))

            # OWW ALWAYS fed (needs continuous mel spectrogram)
            ww_future = loop.run_in_executor(self._executor, self._ww.feed, audio)

            if energy < self._energy_floor:
                await ww_future
                logger.sentinel_debug("SILENCE", e=f"{energy:.4f}")
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

    ##### PARTIAL PIPELINE (LAYERS 1-2) #####

    async def wait_for_voice(
        self,
        audio_queue: asyncio.Queue[np.ndarray],
        timeout: float,
    ) -> bool:
        """Layers 1-2 only — energy gate → VAD. Returns True if voice detected within timeout."""
        self._running = True
        self._vad.reset()
        loop = asyncio.get_running_loop()

        await self._drain_cooldown(audio_queue)

        logger.loop("DEFAULT", f"waiting for voice ({timeout:.1f}s)")

        deadline = monotonic() + timeout
        while self._running:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break

            try:
                audio = await asyncio.wait_for(audio_queue.get(), timeout=remaining)
            except TimeoutError:
                break

            energy = float(np.sqrt(np.mean(audio**2)))

            if energy < self._energy_floor:
                logger.loop_debug("SILENCE", e=f"{energy:.4f}")
                continue

            vad_prob = await loop.run_in_executor(self._executor, self._vad, audio)

            if vad_prob < self._vad.threshold:
                logger.loop_debug("NOISE", vad=f"{vad_prob:.2f}", e=f"{energy:.4f}")
                continue

            logger.loop("VOICE", "speech detected, continuing", vad=f"{vad_prob:.2f}")
            return True

        logger.loop("DEFAULT", "silence timeout, ending conversation")
        return False

    ##### SHARED #####

    async def _drain_cooldown(self, audio_queue: asyncio.Queue[np.ndarray]) -> None:
        """Discard audio frames during cooldown to let speaker echo dissipate."""
        if self._cooldown <= 0:
            return
        deadline = monotonic() + self._cooldown
        while monotonic() < deadline and self._running:
            await audio_queue.get()

    async def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
