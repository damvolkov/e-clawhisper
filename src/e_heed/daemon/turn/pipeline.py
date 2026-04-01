"""Turn pipeline — single STT → Agent → TTS cycle with 3-stage streaming.

Flow:
  1. Open STT WebSocket session
  2. Stream audio chunks to STT ‖ VAD monitors end-of-speech (parallel)
  3. Silence detected → close STT → get transcript
  4. Three concurrent stages:
     a) Agent text_delta → sentence splitter → sentence_queue
     b) sentence_queue → TTS synthesize → pcm_queue
     c) pcm_queue → sd.RawOutputStream (speaker)
  5. Return TurnComplete or TurnError
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

from e_heed.daemon.adapters.audio import AudioAdapter
from e_heed.daemon.adapters.base import AgentPort, STTPort, TTSPort
from e_heed.daemon.turn.vad import EndOfSpeechDetector
from e_heed.shared.logger import logger
from e_heed.shared.operational.events import Turn, TurnComplete, TurnError
from e_heed.shared.settings import VADConfig

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n")
_SENTINEL: None = None


class TurnPipeline:
    """Executes one complete turn: listen → transcribe → agent → speak (3-stage streaming)."""

    __slots__ = (
        "_stt",
        "_agent",
        "_tts",
        "_vad",
        "_executor",
        "_tts_sample_rate",
        "_sentence_queue",
        "_pcm_queue",
        "_running",
    )

    _SENTENCE_QUEUE_SIZE: int = 10

    def __init__(
        self,
        *,
        stt: STTPort,
        agent: AgentPort,
        tts: TTSPort,
        vad_config: VADConfig,
        tts_sample_rate: int = 22050,
        pcm_queue_size: int = 20,
    ) -> None:
        self._stt = stt
        self._agent = agent
        self._tts = tts
        self._vad = EndOfSpeechDetector(vad_config)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._tts_sample_rate = tts_sample_rate
        self._sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self._SENTENCE_QUEUE_SIZE)
        self._pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=pcm_queue_size)
        self._running = False

    @property
    def sentence_queue(self) -> asyncio.Queue[str | None]:
        return self._sentence_queue

    @property
    def pcm_queue(self) -> asyncio.Queue[bytes | None]:
        return self._pcm_queue

    async def run(self, audio: AudioAdapter) -> TurnComplete | TurnError:
        """Execute one turn cycle."""
        self._running = True
        self._vad.reset()
        turn = Turn()
        loop = asyncio.get_running_loop()

        try:
            # Phase 1: Listen — STT streaming ‖ VAD end-of-speech
            logger.turn("STT", "listening...", turn_id=turn.turn_id)
            await self._stt.start_utterance()

            while self._running:
                audio_chunk = await audio.queue.get()

                stt_task = asyncio.create_task(self._stt.stream(audio_chunk.tobytes()))
                vad_result = await loop.run_in_executor(self._executor, self._vad.process, audio_chunk)
                await stt_task

                if vad_result.is_speech:
                    logger.turn_debug("VAD", "speech", prob=f"{vad_result.probability:.2f}")

                if vad_result.should_stop:
                    logger.turn("VAD", "end-of-speech detected")
                    break

            # Phase 2: Transcript
            turn.transcript = await self._stt.finish_utterance()

            if not turn.transcript.strip():
                logger.turn("STT", "(empty transcript)")
                return turn.to_error("empty_transcript")

            logger.turn("STT", logger.truncate(turn.transcript))

            # Phase 3: 3-stage streaming — Agent ‖ TTS ‖ speaker
            turn.response = await self._stream_response(turn.transcript, audio)

            if not turn.response.strip():
                return turn.to_error("empty_response")

            logger.turn("AGENT", f"response: {logger.truncate(turn.response)}")
            return turn.to_complete()

        except Exception as exc:
            logger.error(f"turn_error: {exc}", turn_id=turn.turn_id)
            return turn.to_error(str(exc))

    async def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)

    ##### STREAMING #####

    def drain_queues(self) -> None:
        """Discard all pending items from sentence and pcm queues."""
        for q in (self._sentence_queue, self._pcm_queue):
            while not q.empty():
                q.get_nowait()

    async def _stream_response(self, transcript: str, audio: AudioAdapter) -> str:
        """3 concurrent stages: Agent → sentence_queue → TTS → pcm_queue → speaker.

        Uses TaskGroup for structured concurrency — if any stage fails, all others
        are cancelled immediately, preventing deadlocks from missing sentinels.
        """
        logger.turn("AGENT", f"query: {logger.truncate(transcript)}")
        self.drain_queues()
        response_parts: list[str] = []
        duration = 0.0

        async def _speaker_stage() -> None:
            nonlocal duration
            duration = await audio.play(self._pcm_queue, self._tts_sample_rate)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._stage_agent_to_sentences(transcript))
                tg.create_task(self._stage_tts_to_pcm(response_parts))
                tg.create_task(_speaker_stage())
        except ExceptionGroup as eg:
            # Re-raise first real error (not CancelledError from sibling cancellation)
            real_errors = [e for e in eg.exceptions if not isinstance(e, asyncio.CancelledError)]
            if real_errors:
                raise real_errors[0] from eg

        logger.turn("TTS", f"played {duration:.1f}s streaming")
        return " ".join(response_parts)

    async def _stage_agent_to_sentences(self, transcript: str) -> None:
        """Agent text_delta → sentence boundary → sentence_queue."""
        buffer = ""
        try:
            async for chunk in self._agent.send(transcript):
                buffer += chunk

                while (m := _SENTENCE_RE.search(buffer)) is not None:
                    sentence = buffer[: m.start()].strip()
                    buffer = buffer[m.end() :]
                    if sentence:
                        logger.turn_debug("STREAM", f"sentence: {logger.truncate(sentence, 50)}")
                        await self._sentence_queue.put(sentence)

            if remaining := buffer.strip():
                await self._sentence_queue.put(remaining)
        finally:
            await self._sentence_queue.put(_SENTINEL)

    async def _stage_tts_to_pcm(self, response_parts: list[str]) -> None:
        """sentence_queue → TTS synthesize → pcm_queue."""
        try:
            while (sentence := await self._sentence_queue.get()) is not _SENTINEL:
                if not self._running:
                    break
                response_parts.append(sentence)
                logger.turn("TTS", f"synthesizing: {logger.truncate(sentence, 50)}")
                async for pcm_chunk in self._tts.synthesize(sentence):
                    await self._pcm_queue.put(pcm_chunk)
        finally:
            await self._pcm_queue.put(_SENTINEL)
