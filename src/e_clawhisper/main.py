"""e-clawhisper entry point — CLI + LiveKit agent runner."""

from __future__ import annotations

import sys
from typing import Literal

import typer
from livekit import agents

from e_clawhisper.core.logger import LogIcon, configure_logging, logger
from e_clawhisper.core.settings import settings as st
from e_clawhisper.sessions.voice import VoiceSession

cli = typer.Typer(help="e-clawhisper: always-on voice channel bridge")


@cli.command("run")
def cmd_run(
    mode: Literal["dev", "console"] = typer.Option("dev", "--mode", "-m", help="LiveKit run mode"),
) -> None:
    """Start the voice channel bridge."""
    configure_logging(st.log_level)

    logger.info(
        "STARTING e-clawhisper | agent=%s | channel=%s | mode=%s",
        st.AGENT_NAME,
        st.CHANNEL_BACKEND,
        mode,
        icon=LogIcon.START,
    )

    server = agents.AgentServer()
    session = VoiceSession()
    server.rtc_session()(session.entrypoint)

    sys.argv = [sys.argv[0], mode]
    agents.cli.run_app(server)


@cli.command("info")
def cmd_info() -> None:
    """Show current configuration."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="e-clawhisper configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Agent Name (wake word)", st.AGENT_NAME)
    table.add_row("Agent ID", st.AGENT_ID or "(not set)")
    table.add_row("Channel Backend", st.CHANNEL_BACKEND)
    table.add_row("OpenFang URL", st.openfang_base_url)
    table.add_row("STT (WhisperLive)", st.stt_ws_url)
    table.add_row("TTS (Piper)", f"{st.TTS_HOST}:{st.TTS_PORT}")
    table.add_row("LiveKit", st.LIVEKIT_URL)
    table.add_row("Conversation Timeout", f"{st.CONVERSATION_TIMEOUT}s")
    table.add_row("Environment", st.ENVIRONMENT)

    console.print(table)


if __name__ == "__main__":
    cli()
