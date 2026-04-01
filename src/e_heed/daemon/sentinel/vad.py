"""Silero VAD — pure numpy ONNX wrapper for voice classification.

No torch dependency. Accepts exactly 512 samples @ 16kHz per call.
Returns speech probability [0.0, 1.0].
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import numpy as np
import onnxruntime

from e_heed.shared.logger import logger
from e_heed.shared.operational.exceptions import ModelNotFoundError
from e_heed.shared.settings import settings

_CONTEXT_SIZE = 64
_WINDOW_SIZE = 512
_SAMPLE_RATE = 16000
_STATE_SHAPE = (2, 1, 128)

_MODEL_FILENAME = "silero_vad.onnx"


def _resolve_model_path() -> Path:
    """Find silero_vad.onnx: models/vad/ → package fallback → project root."""
    local = settings.MODELS_DIR / "vad" / _MODEL_FILENAME
    if local.exists():
        return local

    # Copy from silero_vad package if installed
    try:
        pkg_spec = importlib.util.find_spec("silero_vad")
        if pkg_spec and pkg_spec.origin:
            pkg_data = Path(pkg_spec.origin).parent / "data" / _MODEL_FILENAME
            if pkg_data.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pkg_data, local)
                logger.system("OK", f"Copied silero model to {local}")
                return local
    except (ImportError, ValueError):
        pass

    # Fallback to project root
    root = settings.BASE_DIR / _MODEL_FILENAME
    if root.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root, local)
        return local

    msg = f"Silero VAD model not found. Place {_MODEL_FILENAME} in {local.parent}/"
    raise ModelNotFoundError(msg)


class SileroVAD:
    """Pure numpy Silero VAD ONNX — O(1) per 512-sample chunk."""

    __slots__ = ("_session", "_state", "_context", "_threshold")

    def __init__(self, threshold: float = 0.5) -> None:
        model_path = _resolve_model_path()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._threshold = threshold
        self.reset()

        logger.system("OK", f"SileroVAD loaded threshold={threshold}")

    @property
    def threshold(self) -> float:
        return self._threshold

    def reset(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> float:
        """Process 512 float32 samples @ 16kHz → speech probability."""
        x = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        if x.ndim == 1:
            x = x[np.newaxis, :]

        x = np.concatenate([self._context, x], axis=1)

        out, state = self._session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": np.array(_SAMPLE_RATE, dtype=np.int64),
            },
        )

        self._state = state
        self._context = x[:, -_CONTEXT_SIZE:]

        return float(out[0][0])

    def is_voice(self, audio: np.ndarray) -> tuple[float, bool]:
        """Return (probability, is_voice) for classification."""
        prob = self(audio)
        return prob, prob >= self._threshold
