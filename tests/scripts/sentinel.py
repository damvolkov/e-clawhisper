"""Standalone sentinel test script — uses project modules, no torch dependency.

Usage: make script sentinel
"""

from __future__ import annotations

import sys
import time
import threading
from enum import StrEnum, auto
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue

import numpy as np
import sounddevice as sd

from e_clawhisper.daemon.sentinel.vad import SileroVAD
from e_clawhisper.daemon.sentinel.wakeword import WakeWordDetector

_CHUNK_SIZE = 512  # 32ms @ 16kHz — required by Silero VAD
_SAMPLE_RATE = 16000
_CHUNKS_PER_LOG = _SAMPLE_RATE // _CHUNK_SIZE  # ~31 chunks ≈ 1s

_SILENCE_CEIL = 0.01  # RMS below this = dead silence (skip VAD)
_VAD_THRESHOLD = 0.5
_WW_THRESHOLD = 0.5

_SENTINEL = None  # Poison pill for worker shutdown


class AudioLabel(StrEnum):
    SILENCE = auto()
    NOISE = auto()
    VOICE = auto()
    WAKEWORD = auto()


_PRIORITY: dict[AudioLabel, int] = {
    AudioLabel.SILENCE: 0,
    AudioLabel.NOISE: 1,
    AudioLabel.VOICE: 2,
    AudioLabel.WAKEWORD: 3,
}

_COLORS: dict[AudioLabel, str] = {
    AudioLabel.SILENCE: "25;25;112",
    AudioLabel.NOISE: "100;149;237",
    AudioLabel.VOICE: "0;206;209",
    AudioLabel.WAKEWORD: "34;139;34",
}


class ActiveListeningProcessor:
    """3-layer parallel pipeline: Energy gate → [Silero VAD ‖ OpenWakeWord]."""

    __slots__ = (
        "_vad", "_ww", "_queue", "_executor", "_worker",
        "_chunk_count", "_peak_label", "_peak_energy", "_peak_vad_score", "_peak_ww_score",
    )

    def __init__(self, wake_word_model: str = "alexa") -> None:
        self._vad = SileroVAD(threshold=_VAD_THRESHOLD)
        self._ww = WakeWordDetector(model_name=wake_word_model, threshold=_WW_THRESHOLD)

        self._queue: SimpleQueue[np.ndarray | None] = SimpleQueue()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._worker = threading.Thread(target=self._process_loop, daemon=True)

        self._chunk_count = 0
        self._peak_label = AudioLabel.SILENCE
        self._peak_energy = 0.0
        self._peak_vad_score = 0.0
        self._peak_ww_score = 0.0

        _log_info(
            f"Wake Word: '{wake_word_model}' | "
            f"silence<{_SILENCE_CEIL} | vad>={_VAD_THRESHOLD} | ww>={_WW_THRESHOLD}"
        )

    def _promote(self, label: AudioLabel) -> None:
        if _PRIORITY[label] > _PRIORITY[self._peak_label]:
            self._peak_label = label

    def _infer_vad(self, audio: np.ndarray) -> float:
        return self._vad(audio)

    def _infer_ww(self, audio: np.ndarray) -> float:
        return self._ww.feed(audio)

    def _process_chunk(self, audio: np.ndarray) -> None:
        energy = float(np.sqrt(np.mean(audio**2)))
        self._peak_energy = max(self._peak_energy, energy)

        # OWW ALWAYS fed — needs continuous mel spectrogram stream.
        # Silero only on non-silent chunks (energy gate saves ~0.5ms on silence).
        ww_future = self._executor.submit(self._infer_ww, audio)

        if energy < _SILENCE_CEIL:
            ww_future.result()
            return

        vad_future = self._executor.submit(self._infer_vad, audio)

        vad_prob = vad_future.result()
        ww_score = ww_future.result()

        self._peak_vad_score = max(self._peak_vad_score, vad_prob)

        if vad_prob < _VAD_THRESHOLD:
            self._promote(AudioLabel.NOISE)
            return

        self._promote(AudioLabel.VOICE)

        if ww_score > _WW_THRESHOLD:
            self._promote(AudioLabel.WAKEWORD)
            self._peak_ww_score = max(self._peak_ww_score, ww_score)

    def _process_loop(self) -> None:
        while (audio := self._queue.get()) is not _SENTINEL:
            self._process_chunk(audio)
            self._chunk_count += 1

            if self._chunk_count >= _CHUNKS_PER_LOG:
                self._flush_log()

    def _flush_log(self) -> None:
        label = self._peak_label
        color = _COLORS[label]
        tag = label.value.upper()
        ts = time.strftime("%H:%M:%S")

        extras: list[str] = []
        if self._peak_vad_score > 0:
            extras.append(f"vad={self._peak_vad_score:.2f}")
        if label == AudioLabel.WAKEWORD:
            extras.append(f"ww={self._peak_ww_score:.2f}")
        extra_str = f" {' '.join(extras)}" if extras else ""

        print(
            f"\033[38;2;{color}m{ts} [{tag}]{extra_str} e={self._peak_energy:.4f}\033[0m",
            file=sys.stderr,
        )

        self._peak_label = AudioLabel.SILENCE
        self._peak_energy = 0.0
        self._peak_vad_score = 0.0
        self._peak_ww_score = 0.0
        self._chunk_count = 0

    def process_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            print(f"\033[31mAudio status: {status}\033[0m", file=sys.stderr)
        self._queue.put_nowait(indata.flatten().astype(np.float32))

    def start(self) -> None:
        self._worker.start()
        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                blocksize=_CHUNK_SIZE,
                callback=self.process_callback,
                dtype="float32",
            ):
                _log_info("Listening... Ctrl+C to stop")
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            print("\n\033[33mStopped by user.\033[0m", file=sys.stderr)
        finally:
            self._queue.put(_SENTINEL)
            self._worker.join(timeout=2.0)
            self._executor.shutdown(wait=False)


def _log_info(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"\033[38;2;50;205;50m{ts} [INFO] {msg}\033[0m", file=sys.stderr)


##### MAIN #####

if __name__ == "__main__":
    processor = ActiveListeningProcessor(wake_word_model="alexa")
    processor.start()
