"""Config commands — inspect, display, and initialize configuration."""

from __future__ import annotations

import shutil
from pathlib import Path

import cyclopts
from rich.console import Console
from rich.table import Table

from e_heed.shared.settings import AgentBackend, AppConfig, STTBackend, TTSBackend, settings

app = cyclopts.App(name="config", help="Configuration inspection commands.")
console = Console(stderr=True)

_ETC_CONFIG = "/etc/e-heed/config.yaml"

##### COMMANDS #####


@app.command
def info() -> None:
    """Show current configuration."""
    config = AppConfig.load()

    table = Table(title="eheed configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Config File", str(settings.CONFIG_PATH))
    table.add_row("Agent Name", config.agent.name)
    table.add_row("Agent Backend", config.agent.backend)
    table.add_row("Language", config.language)
    table.add_row("Wakeword", config.sentinel.wakeword.model)
    table.add_row("STT Backend", config.stt.backend)
    match config.stt.backend:
        case STTBackend.EVOICE:
            table.add_row("STT URL", str(config.stt.evoice.url))
        case STTBackend.WHISPERLIVE:
            table.add_row("STT URL", str(config.stt.whisperlive.url))
    table.add_row("TTS Backend", config.tts.backend)
    match config.tts.backend:
        case TTSBackend.EVOICE:
            table.add_row("TTS URL", str(config.tts.evoice.url))
        case TTSBackend.KOKORO:
            table.add_row("TTS URL", str(config.tts.kokoro.url))
        case TTSBackend.PIPER:
            table.add_row("TTS URL", str(config.tts.piper.url))
    table.add_row("Conversation", str(config.conversation.enabled))
    table.add_row("VAD Threshold", str(config.vad.threshold))
    table.add_row("Audio Sample Rate", str(config.audio.sample_rate))
    table.add_row("Socket Path", settings.SOCKET_PATH)
    table.add_row("Environment", settings.ENVIRONMENT)

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


@app.command
def init(*, force: bool = False) -> None:
    """Copy default config to ~/.config/e-heed/config.yaml."""
    xdg_dir = settings.XDG_CONFIG_DIR
    target = xdg_dir / "config.yaml"

    if target.exists() and not force:
        console.print(f"[yellow]Config already exists:[/yellow] {target}")
        console.print("Use [cyan]--force[/cyan] to overwrite.")
        return

    # Source: /etc system default or project-local config.yaml
    source = Path(_ETC_CONFIG) if Path(_ETC_CONFIG).exists() else settings.BASE_DIR / "config.yaml"

    if not source.exists():
        console.print("[red]No source config found to copy.[/red]")
        return

    xdg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    console.print(f"[green]Config initialized:[/green] {target}")
    console.print("Edit it and restart the daemon to apply changes.")
