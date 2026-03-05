"""Config commands — inspect and display current configuration."""

from __future__ import annotations

import cyclopts
from rich.console import Console
from rich.table import Table

from e_clawhisper.shared.settings import AgentBackend, STTBackend, TTSBackend, load_config, settings

app = cyclopts.App(name="config", help="Configuration inspection commands.")
console = Console()


##### COMMANDS #####


@app.command
def info() -> None:
    """Show current configuration."""
    config = load_config()

    table = Table(title="eclaw configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Agent Name", config.agent.name)
    table.add_row("Agent Backend", config.agent.backend)
    table.add_row("Wakeword", config.sentinel.wakeword.model)
    table.add_row("STT Backend", config.stt.backend)
    table.add_row("STT Host", f"{config.stt.whisperlive.host}:{config.stt.whisperlive.port}")
    table.add_row("TTS Backend", config.tts.backend)
    table.add_row("TTS Host", f"{config.tts.piper.host}:{config.tts.piper.port}")
    table.add_row("VAD Threshold", str(config.vad.threshold))
    table.add_row("Audio Sample Rate", str(config.audio.sample_rate))
    table.add_row("Socket Path", settings.SOCKET_PATH)
    table.add_row("Environment", settings.ENVIRONMENT)
    table.add_row("Config File", str(settings.CONFIG_PATH))

    console.print(table)


@app.command
def backends() -> None:
    """List available backends."""
    table = Table(title="Available Backends")
    table.add_column("Type", style="cyan")
    table.add_column("Backends", style="green")

    table.add_row("Agent", ", ".join(b.value for b in AgentBackend))
    table.add_row("STT", ", ".join(b.value for b in STTBackend))
    table.add_row("TTS", ", ".join(b.value for b in TTSBackend))

    console.print(table)
