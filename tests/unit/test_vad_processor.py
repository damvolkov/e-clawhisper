"""Tests for TenVADProcessor — frame splitting, silence tracking, should_stop."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from e_clawhisper.shared.settings import VADConfig

_HOP = 256
_SR = 16000


##### FIXTURES #####


def _make_config(
    threshold: float = 0.5,
    silence_duration: float = 1.5,
    min_recording_time: float = 1.0,
) -> VADConfig:
    return VADConfig(
        threshold=threshold,
        silence_duration=silence_duration,
        min_recording_time=min_recording_time,
    )


def _silence(n_samples: int = 512) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.int16)


def _speech(n_samples: int = 512, amplitude: int = 10000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.uniform(-1, 1, n_samples) * amplitude).astype(np.int16)


def _make_processor(config: VADConfig | None = None, speech_flags: list[int] | None = None):
    """Create a TenVADProcessor with mocked TenVad.process()."""
    cfg = config or _make_config()
    flag_iter = iter(speech_flags) if speech_flags else None

    def mock_process(frame: np.ndarray) -> tuple[float, int]:
        flag = next(flag_iter, 0) if flag_iter is not None else (1 if np.abs(frame).mean() > 500 else 0)
        return (0.9 if flag == 1 else 0.1), flag

    with patch("e_clawhisper.daemon.core.processors.vad.TenVad") as mock_cls:
        mock_vad = MagicMock()
        mock_vad.process.side_effect = mock_process
        mock_cls.return_value = mock_vad

        from e_clawhisper.daemon.core.processors.vad import TenVADProcessor

        proc = TenVADProcessor(cfg, sample_rate=_SR)
        proc._vad = mock_vad
        return proc


##### FRAME SPLITTING #####


def test_exact_hop_size_no_remainder() -> None:
    proc = _make_processor(speech_flags=[0])
    chunk = _silence(_HOP)
    result = proc.process(chunk)
    assert not result.is_speech
    assert len(proc._remainder) == 0


def test_double_hop_processes_two_frames() -> None:
    proc = _make_processor(speech_flags=[0, 1])
    chunk = _speech(_HOP * 2)
    result = proc.process(chunk)
    assert result.is_speech
    assert proc._vad.process.call_count == 2


def test_odd_chunk_saves_remainder() -> None:
    proc = _make_processor(speech_flags=[1])
    chunk = _speech(_HOP + 100)
    proc.process(chunk)
    assert len(proc._remainder) == 100


def test_remainder_carried_to_next_call() -> None:
    proc = _make_processor(speech_flags=[1, 1])
    proc.process(_speech(300))
    assert len(proc._remainder) == 300 - _HOP
    assert proc._vad.process.call_count == 1

    proc.process(_speech(300))
    remaining = (300 - _HOP + 300) - _HOP
    assert len(proc._remainder) == remaining
    assert proc._vad.process.call_count == 2


def test_chunk_smaller_than_hop_buffered() -> None:
    proc = _make_processor()
    chunk = _silence(100)
    result = proc.process(chunk)
    assert not result.is_speech
    assert len(proc._remainder) == 100
    assert proc._vad.process.call_count == 0


##### SPEECH DETECTION #####


def test_speech_detected_sets_has_speech() -> None:
    proc = _make_processor(speech_flags=[1])
    result = proc.process(_speech(_HOP))
    assert result.is_speech
    assert proc.has_speech


def test_silence_not_detected_as_speech() -> None:
    proc = _make_processor(speech_flags=[0, 0])
    result = proc.process(_silence(_HOP * 2))
    assert not result.is_speech
    assert not proc.has_speech


def test_mixed_speech_and_silence() -> None:
    proc = _make_processor(speech_flags=[1, 0])
    result = proc.process(_speech(_HOP * 2))
    assert result.is_speech
    assert proc.has_speech


##### SILENCE COUNTING #####


def test_silence_frames_increment() -> None:
    proc = _make_processor(speech_flags=[0, 0, 0])
    proc.process(_silence(_HOP * 3))
    assert proc._silence_frames == 3


def test_speech_resets_silence_count() -> None:
    proc = _make_processor(speech_flags=[0, 0, 1])
    proc.process(_silence(_HOP * 3))
    assert proc._silence_frames == 0


##### SHOULD STOP #####


def test_should_stop_requires_speech_then_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    max_sil = 3
    cfg = _make_config(silence_duration=max_sil * _HOP / _SR, min_recording_time=0.0)
    flags = [1] + [0] * max_sil
    proc = _make_processor(config=cfg, speech_flags=flags)

    monkeypatch.setattr("e_clawhisper.daemon.core.processors.vad.time.monotonic", lambda: 100.0)
    proc.process(_speech(_HOP))
    monkeypatch.setattr("e_clawhisper.daemon.core.processors.vad.time.monotonic", lambda: 101.0)
    result = proc.process(_silence(_HOP * max_sil))
    assert result.should_stop


def test_should_stop_false_without_prior_speech() -> None:
    cfg = _make_config(silence_duration=0.0, min_recording_time=0.0)
    proc = _make_processor(config=cfg, speech_flags=[0, 0])
    result = proc.process(_silence(_HOP * 2))
    assert not result.should_stop


def test_min_recording_time_blocks_early_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(silence_duration=0.016, min_recording_time=2.0)
    proc = _make_processor(config=cfg, speech_flags=[1, 0])

    monkeypatch.setattr("e_clawhisper.daemon.core.processors.vad.time.monotonic", lambda: 100.0)
    proc.process(_speech(_HOP))

    monkeypatch.setattr("e_clawhisper.daemon.core.processors.vad.time.monotonic", lambda: 100.5)
    result = proc.process(_silence(_HOP))
    assert not result.should_stop


def test_min_recording_time_allows_stop_after_elapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(silence_duration=0.016, min_recording_time=1.0)
    proc = _make_processor(config=cfg, speech_flags=[1, 0])

    monkeypatch.setattr("e_clawhisper.daemon.core.processors.vad.time.monotonic", lambda: 100.0)
    proc.process(_speech(_HOP))

    monkeypatch.setattr("e_clawhisper.daemon.core.processors.vad.time.monotonic", lambda: 101.5)
    result = proc.process(_silence(_HOP))
    assert result.should_stop


##### RESET #####


def test_reset_clears_state() -> None:
    proc = _make_processor(speech_flags=[1])
    proc.process(_speech(_HOP))
    assert proc.has_speech

    proc.reset()
    assert not proc.has_speech
    assert proc._silence_frames == 0
    assert proc._recording_start == 0.0
    assert len(proc._remainder) == 0


##### PROBABILITY #####


def test_peak_probability_tracked() -> None:
    def mock_process(frame: np.ndarray) -> tuple[float, int]:
        return 0.85, 1

    proc = _make_processor(speech_flags=[])
    proc._vad.process.side_effect = mock_process
    result = proc.process(_speech(_HOP))
    assert result.probability == pytest.approx(0.85)
