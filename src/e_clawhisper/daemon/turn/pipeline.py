"""Turn pipeline — single STT → LLM → TTS cycle with 3-stage streaming.

Flow:
  1. Open STT WebSocket session
  2. Stream audio chunks to STT ‖ VAD monitors end-of-speech (parallel)
  3. Silence detected → close STT → get transcript
  4. Three concurrent stages:
     a) LLM text_delta → sentence splitter → sentence_queue
     b) sentence_queue → TTS synthesize → pcm_queue
     c) pcm_queue → sd.RawOutputStream (speaker)
  5. Return TurnComplete or TurnError
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

from e_clawhisper.daemon.adapters.audio import AudioAdapter
from e_clawhisper.daemon.adapters.llm import LLMAdapter
from e_clawhisper.daemon.adapters.stt import STTAdapter
from e_clawhisper.daemon.adapters.tts import TTSAdapter
from e_clawhisper.daemon.turn.vad import EndOfSpeechDetector
from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.operational.events import Turn, TurnComplete, TurnError
from e_clawhisper.shared.settings import VADConfig

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n")
_SENTINEL: None = None


class TurnPipeline:
    """Executes one complete turn: listen → transcribe → agent → speak (3-stage streaming)."""

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

    ##### PUBLIC #####

    async def run(self, audio: AudioAdapter) -> TurnComplete | TurnError:
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
                audio_chunk = await audio.queue.get()

                stt_task = asyncio.create_task(self._stt.stream_audio(audio_chunk.tobytes()))
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

            # Phase 3: 3-stage streaming — LLM ‖ TTS ‖ speaker
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

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    ##### STREAMING STAGES #####

    async def _stream_response(self, transcript: str, audio: AudioAdapter) -> str:
        """Orchestrate 3 concurrent stages: LLM → TTS → speaker."""
        logger.turn("AGENT", f"query: {logger.truncate(transcript)}")

        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=20)
        response_parts: list[str] = []

        stage_llm = asyncio.create_task(self._stage_llm_to_sentences(transcript, sentence_queue))
        stage_tts = asyncio.create_task(
            self._stage_tts_to_pcm(sentence_queue, pcm_queue, response_parts)
        )
        stage_speaker = asyncio.create_task(
            audio.play_pcm_queue(pcm_queue, self._tts_sample_rate)
        )

        duration = 0.0
        try:
            duration = await stage_speaker
            await asyncio.gather(stage_llm, stage_tts)
        except Exception:
            stage_llm.cancel()
            stage_tts.cancel()
            stage_speaker.cancel()
            raise

        logger.turn("TTS", f"played {duration:.1f}s streaming")
        return " ".join(response_parts)

    async def _stage_llm_to_sentences(
        self,
        transcript: str,
        sentence_queue: asyncio.Queue[str | None],
    ) -> None:
        """Stage 1: LLM text_delta → sentence boundary detection → sentence_queue."""
        buffer = ""
        try:
            async for chunk in self._llm.send_message(transcript):
                buffer += chunk

                while (m := _SENTENCE_RE.search(buffer)) is not None:
                    sentence = buffer[: m.start()].strip()
                    buffer = buffer[m.end() :]
                    if sentence:
                        logger.turn_debug("STREAM", f"sentence: {logger.truncate(sentence, 50)}")
                        await sentence_queue.put(sentence)

            if (remaining := buffer.strip()):
                await sentence_queue.put(remaining)
        finally:
            await sentence_queue.put(_SENTINEL)

    async def _stage_tts_to_pcm(
        self,
        sentence_queue: asyncio.Queue[str | None],
        pcm_queue: asyncio.Queue[bytes | None],
        response_parts: list[str],
    ) -> None:
        """Stage 2: sentence_queue → TTS synthesize → pcm_queue."""
        try:
            while (sentence := await sentence_queue.get()) is not _SENTINEL:
                if not self._running:
                    break
                response_parts.append(sentence)
                logger.turn("TTS", f"synthesizing: {logger.truncate(sentence, 50)}")
                async for pcm_chunk in self._tts.synthesize(sentence):
                    await pcm_queue.put(pcm_chunk)
        finally:
            await pcm_queue.put(_SENTINEL)
