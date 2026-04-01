"""Unit test fixtures."""

from __future__ import annotations

import pytest

from e_heed.shared.settings import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig()
