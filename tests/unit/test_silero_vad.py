"""Tests for SileroVAD — model resolution, inference, state reset."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from e_clawhisper.daemon.sentinel.vad import SileroVAD, _resolve_model_path
from e_clawhisper.shared.operational.exceptions import ModelNotFoundError


##### MODEL RESOLUTION #####


@patch("e_clawhisper.daemon.sentinel.vad.settings")
def test_resolve_local_model(mock_settings: MagicMock, tmp_path: Path) -> None:
    model_dir = tmp_path / "vad"
    model_dir.mkdir()
    model_file = model_dir / "silero_vad.onnx"
    model_file.write_bytes(b"fake-onnx")

    mock_settings.MODELS_DIR = tmp_path
    mock_settings.BASE_DIR = tmp_path

    result = _resolve_model_path()
    assert result == model_file


@patch("e_clawhisper.daemon.sentinel.vad.settings")
def test_resolve_fallback_to_root(mock_settings: MagicMock, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    mock_settings.MODELS_DIR = models_dir

    root_model = tmp_path / "silero_vad.onnx"
    root_model.write_bytes(b"fake-onnx-root")
    mock_settings.BASE_DIR = tmp_path

    result = _resolve_model_path()
    assert result.exists()
    assert result == models_dir / "vad" / "silero_vad.onnx"


@patch("e_clawhisper.daemon.sentinel.vad.importlib.util.find_spec", return_value=None)
@patch("e_clawhisper.daemon.sentinel.vad.settings")
def test_resolve_raises_when_not_found(mock_settings: MagicMock, _mock_spec: MagicMock, tmp_path: Path) -> None:
    mock_settings.MODELS_DIR = tmp_path / "models"
    mock_settings.BASE_DIR = tmp_path

    with pytest.raises(ModelNotFoundError, match="not found"):
        _resolve_model_path()


@patch("e_clawhisper.daemon.sentinel.vad.settings")
@patch("e_clawhisper.daemon.sentinel.vad.importlib.util.find_spec")
def test_resolve_from_package(mock_find: MagicMock, mock_settings: MagicMock, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    mock_settings.MODELS_DIR = models_dir
    mock_settings.BASE_DIR = tmp_path

    # Setup fake package
    pkg_dir = tmp_path / "pkg" / "data"
    pkg_dir.mkdir(parents=True)
    pkg_model = pkg_dir / "silero_vad.onnx"
    pkg_model.write_bytes(b"fake-onnx-pkg")

    mock_spec = MagicMock()
    mock_spec.origin = str(tmp_path / "pkg" / "__init__.py")
    mock_find.return_value = mock_spec

    result = _resolve_model_path()
    assert result.exists()
    assert result == models_dir / "vad" / "silero_vad.onnx"


##### SILERO VAD (mocked ONNX) #####


@patch("e_clawhisper.daemon.sentinel.vad._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.vad.onnxruntime.InferenceSession")
def test_vad_init(mock_session_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "model.onnx"
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session

    vad = SileroVAD(threshold=0.6)
    assert vad.threshold == 0.6


@patch("e_clawhisper.daemon.sentinel.vad._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.vad.onnxruntime.InferenceSession")
def test_vad_call_returns_float(mock_session_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "model.onnx"
    mock_session = MagicMock()
    mock_session.run.return_value = (np.array([[0.75]]), np.zeros((2, 1, 128), dtype=np.float32))
    mock_session_cls.return_value = mock_session

    vad = SileroVAD()
    prob = vad(np.zeros(512, dtype=np.float32))
    assert prob == pytest.approx(0.75)


@patch("e_clawhisper.daemon.sentinel.vad._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.vad.onnxruntime.InferenceSession")
def test_vad_is_voice(mock_session_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "model.onnx"
    mock_session = MagicMock()
    mock_session.run.return_value = (np.array([[0.75]]), np.zeros((2, 1, 128), dtype=np.float32))
    mock_session_cls.return_value = mock_session

    vad = SileroVAD(threshold=0.5)
    prob, is_v = vad.is_voice(np.zeros(512, dtype=np.float32))
    assert is_v is True
    assert prob == pytest.approx(0.75)


@patch("e_clawhisper.daemon.sentinel.vad._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.vad.onnxruntime.InferenceSession")
def test_vad_is_voice_below_threshold(mock_session_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "model.onnx"
    mock_session = MagicMock()
    mock_session.run.return_value = (np.array([[0.3]]), np.zeros((2, 1, 128), dtype=np.float32))
    mock_session_cls.return_value = mock_session

    vad = SileroVAD(threshold=0.5)
    prob, is_v = vad.is_voice(np.zeros(512, dtype=np.float32))
    assert is_v is False
    assert prob == pytest.approx(0.3)


@patch("e_clawhisper.daemon.sentinel.vad._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.vad.onnxruntime.InferenceSession")
def test_vad_reset_clears_state(mock_session_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "model.onnx"
    mock_session = MagicMock()
    mock_session.run.return_value = (np.array([[0.9]]), np.ones((2, 1, 128), dtype=np.float32))
    mock_session_cls.return_value = mock_session

    vad = SileroVAD()
    vad(np.zeros(512, dtype=np.float32))
    # State is now non-zero
    assert np.any(vad._state != 0)

    vad.reset()
    assert np.all(vad._state == 0)
    assert np.all(vad._context == 0)


@patch("e_clawhisper.daemon.sentinel.vad._resolve_model_path")
@patch("e_clawhisper.daemon.sentinel.vad.onnxruntime.InferenceSession")
def test_vad_handles_1d_and_2d_input(mock_session_cls: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
    mock_resolve.return_value = tmp_path / "model.onnx"
    mock_session = MagicMock()
    mock_session.run.return_value = (np.array([[0.5]]), np.zeros((2, 1, 128), dtype=np.float32))
    mock_session_cls.return_value = mock_session

    vad = SileroVAD()

    # 1D input
    prob1 = vad(np.zeros(512, dtype=np.float32))
    assert isinstance(prob1, float)

    # 2D input (already batched)
    prob2 = vad(np.zeros((1, 512), dtype=np.float32))
    assert isinstance(prob2, float)
