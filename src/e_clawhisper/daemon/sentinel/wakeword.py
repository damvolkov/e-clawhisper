"""OpenWakeWord detector — ONNX wake word detection.

Requires continuous audio feeding to maintain internal mel spectrogram state.
Accepts int16 PCM input. Any chunk size >= 400 samples.
"""

from __future__ import annotations

import os

import numpy as np
import openwakeword
from openwakeword.model import Model

from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.operational.exceptions import ModelNotFoundError
from e_clawhisper.shared.settings import settings

_CUSTOM_MODELS_DIR = settings.MODELS_DIR / "ww"


def _resolve_model_path(name: str) -> str:
    """Find OWW model: models/ww/{name}.onnx → pretrained fallback."""
    custom = _CUSTOM_MODELS_DIR / f"{name}.onnx"
    if custom.exists():
        return str(custom)

    for path in openwakeword.get_pretrained_model_paths():
        if name in os.path.basename(path):
            return path

    available = [os.path.basename(p).split("_v0")[0] for p in openwakeword.get_pretrained_model_paths()]
    msg = f"WakeWord model '{name}' not found. Checked: {custom} | Pretrained: {available}"
    raise ModelNotFoundError(msg)


class WakeWordDetector:
    """OpenWakeWord ONNX wrapper — feed every chunk, check score."""

    __slots__ = ("_model", "_threshold", "_name")

    def __init__(self, model_name: str = "alexa", threshold: float = 0.5) -> None:
        model_path = _resolve_model_path(model_name)
        self._model = Model(wakeword_model_paths=[model_path])
        self._threshold = threshold
        self._name = model_name

        logger.system("OK", f"WakeWord loaded model='{model_name}' threshold={threshold}")

    @property
    def name(self) -> str:
        return self._name

    def feed(self, audio_float32: np.ndarray) -> float:
        """Feed audio chunk (float32) → return max wakeword score.

        OWW requires int16 PCM internally for mel spectrogram.
        Must be called on EVERY chunk (including silence) to maintain state.
        """
        audio_int16 = (audio_float32 * 32767).astype(np.int16)
        prediction = self._model.predict(audio_int16)
        return max(prediction.values(), default=0.0)

    def detect(self, audio_float32: np.ndarray) -> tuple[float, bool]:
        """Feed chunk → return (score, is_wakeword)."""
        score = self.feed(audio_float32)
        return score, score >= self._threshold

    def reset(self) -> None:
        self._model.reset()
