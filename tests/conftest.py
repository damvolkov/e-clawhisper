"""Root-level test fixtures."""

from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "DEV")
os.environ.setdefault("AGENT_NAME", "testbot")
os.environ.setdefault("AGENT_ID", "test-agent-00000000-0000-0000-0000-000000000000")
os.environ.setdefault("CHANNEL_BACKEND", "openfang")
