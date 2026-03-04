"""Tests for ring buffer."""

from __future__ import annotations

import numpy as np
import pytest

from e_clawhisper.shared.operational.buffer import RingBuffer


##### BASIC OPS #####


def test_empty_buffer_read() -> None:
    buf = RingBuffer(capacity=1024)
    assert buf.available == 0
    assert len(buf.read_all()) == 0


def test_write_and_read() -> None:
    buf = RingBuffer(capacity=1024)
    data = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    buf.write(data)
    assert buf.available == 5
    result = buf.read_all()
    np.testing.assert_array_equal(result, data)


##### WRAP-AROUND #####


def test_ring_buffer_wraps() -> None:
    buf = RingBuffer(capacity=8)
    buf.write(np.array([1, 2, 3, 4, 5, 6], dtype=np.int16))
    buf.write(np.array([7, 8, 9, 10], dtype=np.int16))
    result = buf.read_all()
    np.testing.assert_array_equal(result, np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int16))


def test_overflow_keeps_latest() -> None:
    buf = RingBuffer(capacity=4)
    buf.write(np.arange(10, dtype=np.int16))
    assert buf.available == 4
    result = buf.read_all()
    np.testing.assert_array_equal(result, np.array([6, 7, 8, 9], dtype=np.int16))


##### CLEAR #####


def test_clear_resets() -> None:
    buf = RingBuffer(capacity=16)
    buf.write(np.ones(10, dtype=np.int16))
    buf.clear()
    assert buf.available == 0
    assert len(buf.read_all()) == 0
