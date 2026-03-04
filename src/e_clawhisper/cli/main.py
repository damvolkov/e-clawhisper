"""Cyclopts CLI entry point for eclaw."""

from __future__ import annotations

import cyclopts

from e_clawhisper.cli.commands.config import app as config_app
from e_clawhisper.cli.commands.session import app as session_app

app = cyclopts.App(
    name="eclaw",
    help="e-clawhisper: always-on voice channel daemon.",
)

app.command(session_app)
app.command(config_app)
