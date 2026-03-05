"""Turn pipeline — single STT → LLM → TTS cycle.

Flow:
  1. Open STT WebSocket session
  2. Stream audio chunks to STT ‖ VAD monitors end-of-speech (parallel)
  3. Silence detected → close STT → get transcript
  4. Send transcript to LLM → collect response
  5. TTS response → play audio
  6. Return TurnComplete or TurnError
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from e_clawhisper.daemon.adapters.llm import LLMAdapter
from e_clawhisper.daemon.adapters.stt import STTAdapter
from e_clawhisper.daemon.adapters.tts import TTSAdapter
from e_clawhisper.daemon.turn.vad import EndOfSpeechDetector
from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.operational.events import Turn, TurnComplete, TurnError
from e_clawhisper.shared.settings import VADConfig


class TurnPipeline:
    """Executes one complete turn: listen → transcribe → agent → speak."""

    __slots__ = (
        "_stt",
        "_llm",
        "_tts",
        "_vad",
        "_executor",
        "_tts_sample_rate",
        "_running",
    )

    def __init__(
        self,
        *,
        stt: STTAdapter,
        llm: LLMAdapter,
        tts: TTSAdapter,
        vad_config: VADConfig,
        tts_sample_rate: int = 22050,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._vad = EndOfSpeechDetector(vad_config)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._tts_sample_rate = tts_sample_rate
        self._running = False

    async def run(
        self,
        audio_queue: asyncio.Queue[np.ndarray],
        play_fn: callable,
    ) -> TurnComplete | TurnError:
        """Execute one turn cycle. Returns event for orchestrator."""
        self._running = True
        self._vad.reset()
        turn = Turn()
        loop = asyncio.get_running_loop()

        try:
            # Phase 1: Listen — STT streaming ‖ VAD end-of-speech
            logger.turn("STT", "listening...", turn_id=turn.turn_id)
            await self._stt.start_utterance()

            while self._running:
                audio = await audio_queue.get()

                # Parallel: stream float32 bytes to STT + VAD end-of-speech check
                stt_task = asyncio.create_task(self._stt.stream_audio(audio.tobytes()))
                vad_result = await loop.run_in_executor(self._executor, self._vad.process, audio)
                await stt_task

                if vad_result.is_speech:
                    logger.turn_debug("VAD", "speech", prob=f"{vad_result.probability:.2f}")

                if vad_result.should_stop:
                    logger.turn("VAD", "end-of-speech detected")
                    break

            # Phase 2: Get transcript
            turn.transcript = await self._stt.finish_utterance()

            if not turn.transcript.strip():
                logger.turn("STT", "(empty transcript)")
                return turn.to_error("empty_transcript")

            logger.turn("STT", logger.truncate(turn.transcript))

            # Phase 3: Agent
            logger.turn("AGENT", f"query: {logger.truncate(turn.transcript)}")
            chunks: list[str] = []
            async for chunk_text in self._llm.send_message(turn.transcript):
                chunks.append(chunk_text)

            turn.response = "".join(chunks)
            if not turn.response.strip():
                return turn.to_error("empty_response")

            logger.turn("AGENT", f"response: {logger.truncate(turn.response)}")

            # Phase 4: TTS
            logger.turn("TTS", f"synthesizing: {logger.truncate(turn.response)}")
            audio_chunks: list[bytes] = []
            async for pcm_chunk in self._tts.synthesize(turn.response):
                audio_chunks.append(pcm_chunk)

            if audio_chunks:
                pcm = b"".join(audio_chunks)
                audio_array = np.frombuffer(pcm, dtype=np.int16)
                play_fn(audio_array, self._tts_sample_rate)
                duration = len(audio_array) / self._tts_sample_rate
                logger.turn("TTS", f"playing {duration:.1f}s")
                await asyncio.sleep(duration)

            return turn.to_complete()

        except Exception as exc:
            logger.error(f"turn_error: {exc}", turn_id=turn.turn_id)
            return turn.to_error(str(exc))

    async def stop(self) -> None:
        self._running = False

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
