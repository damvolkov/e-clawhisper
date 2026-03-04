"""Lock-free ring buffer for streaming audio chunks."""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Fixed-size circular buffer for int16 audio samples."""

    __slots__ = ("_buf", "_capacity", "_write_pos", "_count")

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.int16)
        self._capacity = capacity
        self._write_pos = 0
        self._count = 0

    @property
    def available(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count >= self._capacity

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        if n > self._capacity:
            data = data[-self._capacity :]
            n = self._capacity

        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos : end] = data
        else:
            first = self._capacity - self._write_pos
            self._buf[self._write_pos :] = data[:first]
            self._buf[: n - first] = data[first:]

        self._write_pos = end % self._capacity
        self._count = min(self._count + n, self._capacity)

    def read_all(self) -> np.ndarray:
        if self._count == 0:
            return np.array([], dtype=np.int16)

        start = (self._write_pos - self._count) % self._capacity
        if start + self._count <= self._capacity:
            result = self._buf[start : start + self._count].copy()
        else:
            first = self._capacity - start
            result = np.concatenate([self._buf[start:], self._buf[: self._count - first]])

        return result

    def clear(self) -> None:
        self._write_pos = 0
        self._count = 0
