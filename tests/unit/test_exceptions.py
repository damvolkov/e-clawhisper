"""Tests for exception hierarchy."""

from __future__ import annotations

import pytest

from e_clawhisper.shared.operational.exceptions import (
    AdapterError,
    AppError,
    ConfigError,
    HealthCheckError,
    ModelNotFoundError,
    PipelineError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [AdapterError, PipelineError, ModelNotFoundError, ConfigError, HealthCheckError],
    ids=["adapter", "pipeline", "model", "config", "health"],
)
def test_all_exceptions_inherit_app_error(exc_cls: type[AppError]) -> None:
    exc = exc_cls("test message")
    assert isinstance(exc, AppError)
    assert str(exc) == "test message"


def test_app_error_is_base_exception() -> None:
    assert issubclass(AppError, Exception)
