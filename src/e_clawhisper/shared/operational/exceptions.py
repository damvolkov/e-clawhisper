"""Exception hierarchy for e-clawhisper."""

from __future__ import annotations


class AppError(Exception):
    """Base for all project exceptions."""


class AdapterError(AppError):
    """External service communication failure."""


class PipelineError(AppError):
    """Pipeline processing failure."""


class ModelNotFoundError(AppError):
    """ML model file not found."""
