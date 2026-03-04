"""Processor interface — base for VAD, turn detection, wake word."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ProcessorBase(ABC):
    """Abstract audio processor (VAD, wake-word, turn-detection)."""

    @abstractmethod
    def process(self, audio_chunk: np.ndarray) -> Any:
        """Process one audio frame, return processor-specific result."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
