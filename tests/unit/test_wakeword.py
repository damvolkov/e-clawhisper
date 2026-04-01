"""Tests for WakeWordDetector — model resolution, feed, detect, reset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from e_clawhisper.daemon.sentinel.wakeword import WakeWordDetector, _resolve_model_path
from e_clawhisper.shared.operational.exceptions import ModelNotFoundError


##### MODEL RESOLUTION #####


def test_resolve_local_model(tmp_path: Path) -> None:
    ww_dir = tmp_path / "ww"
    ww_dir.mkdir()
    model = ww_dir / "alexa.onnx"
    model.write_bytes(b"fake")

    with patch("e_clawhisper.daemon.sentinel.wakeword._MODELS_DIR", ww_dir):
        result = _resolve_model_path("alexa")
    assert result == model


@patch("e_clawhisper.daemon.sentinel.wakeword.openwakeword.get_pretrained_model_paths")
def test_resolve_from_pretrained(mock_pretrained: MagicMock, tmp_path: Path) -> None:
    ww_dir = tmp_path / "ww"

    pretrained_path = tmp_path / "pretrained" / "alexa_v0.1.onnx"
    pretrained_path.parent.mkdir(parents=True)
    pretrained_path.write_bytes(b"pretrained-model")

    mock_pretrained.return_value = [str(pretrained_path)]

    with patch("e_clawhisper.daemon.sentinel.wakeword._MODELS_DIR", ww_dir):
        result = _resolve_model_path("alexa")
    assert result.exists()
    assert result == ww_dir / "alexa.onnx"


@patch("e_clawhisper.daemon.sentinel.wakeword.openwakeword.get_pretrained_model_paths")
def test_resolve_not_found(mock_pretrained: MagicMock, tmp_path: Path) -> None:
    mock_pretrained.return_value = ["/fake/other_model_v0.1.onnx"]

    with patch("e_clawhisper.daemon.sentinel.wakeword._MODELS_DIR", tmp_path / "ww"):
        with pytest.raises(ModelNotFoundError, match="not found"):
            _resolve_model_path("nonexistent")


##### DETECTOR (MOCKED OWW) #####


@patch("e_clawhisper.daemon.sentinel.wakeword._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.wakeword.Model")
def test_detector_init(mock_model_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "alexa.onnx"
    detector = WakeWordDetector(model_name="alexa", threshold=0.6)
    assert detector.name == "alexa"
    assert detector.threshold == 0.6
    mock_model_cls.assert_called_once()


@patch("e_clawhisper.daemon.sentinel.wakeword._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.wakeword.Model")
def test_detector_feed(mock_model_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "alexa.onnx"
    mock_model = MagicMock()
    mock_model.predict.return_value = {"alexa": 0.85}
    mock_model_cls.return_value = mock_model

    detector = WakeWordDetector()
    score = detector.feed(np.zeros(512, dtype=np.float32))
    assert score == pytest.approx(0.85)
    mock_model.predict.assert_called_once()


@patch("e_clawhisper.daemon.sentinel.wakeword._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.wakeword.Model")
def test_detector_detect_above_threshold(mock_model_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "alexa.onnx"
    mock_model = MagicMock()
    mock_model.predict.return_value = {"alexa": 0.8}
    mock_model_cls.return_value = mock_model

    detector = WakeWordDetector(threshold=0.5)
    score, detected = detector.detect(np.zeros(512, dtype=np.float32))
    assert detected is True
    assert score == pytest.approx(0.8)


@patch("e_clawhisper.daemon.sentinel.wakeword._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.wakeword.Model")
def test_detector_detect_below_threshold(mock_model_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "alexa.onnx"
    mock_model = MagicMock()
    mock_model.predict.return_value = {"alexa": 0.2}
    mock_model_cls.return_value = mock_model

    detector = WakeWordDetector(threshold=0.5)
    score, detected = detector.detect(np.zeros(512, dtype=np.float32))
    assert detected is False


@patch("e_clawhisper.daemon.sentinel.wakeword._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.wakeword.Model")
def test_detector_feed_empty_prediction(mock_model_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "alexa.onnx"
    mock_model = MagicMock()
    mock_model.predict.return_value = {}
    mock_model_cls.return_value = mock_model

    detector = WakeWordDetector()
    score = detector.feed(np.zeros(512, dtype=np.float32))
    assert score == 0.0


@patch("e_clawhisper.daemon.sentinel.wakeword._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.wakeword.Model")
def test_detector_reset(mock_model_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "alexa.onnx"
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model

    detector = WakeWordDetector()
    detector.reset()
    mock_model.reset.assert_called_once()


@patch("e_clawhisper.daemon.sentinel.wakeword._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.wakeword.Model")
def test_detector_feed_converts_to_int16(mock_model_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "alexa.onnx"
    mock_model = MagicMock()
    mock_model.predict.return_value = {"alexa": 0.1}
    mock_model_cls.return_value = mock_model

    detector = WakeWordDetector()
    audio = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    detector.feed(audio)

    call_args = mock_model.predict.call_args[0][0]
    assert call_args.dtype == np.int16
