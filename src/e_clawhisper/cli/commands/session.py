"""Session commands — start / stop / status / health / logs."""

from __future__ import annotations

import asyncio
import os
import sys

import cyclopts
import orjson
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from e_clawhisper.daemon.server import DaemonServer
from e_clawhisper.shared.operational.exceptions import HealthCheckError
from e_clawhisper.shared.settings import AppConfig, settings

app = cyclopts.App(name="session", help="Daemon lifecycle commands.")
console = Console(stderr=True)

##### HELPERS #####


async def _send_ipc(command: str) -> dict[str, object]:
    """Send a command to the daemon via Unix socket."""
    try:
        reader, writer = await asyncio.open_unix_connection(settings.SOCKET_PATH)
        writer.write(orjson.dumps({"command": command}))
        await writer.drain()
        data = await reader.read(4096)
        writer.close()
        await writer.wait_closed()
        return orjson.loads(data)
    except (ConnectionRefusedError, FileNotFoundError):
        return {"status": "error", "message": "daemon not running"}


def _print_tagged(pipeline: str, step: str, msg: str, *, error: bool = False) -> None:
    """Print CLI messages with daemon-style tagging."""
    style = "red" if error else "green"
    console.print(f"[cyan]\\[{pipeline}][/cyan] [bold {style}]\\[{step}][/bold {style}] {msg}")


##### COMMANDS #####


@app.command
def start() -> None:
    """Start the voice daemon (foreground)."""
    try:
        config = AppConfig.load()
    except ValidationError as exc:
        _print_tagged("SYSTEM", "ERROR", "invalid config.yaml", error=True)
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"])
            console.print(f"  [red]{loc}:[/red] {err['msg']}")
        sys.exit(1)

    try:
        server = DaemonServer(config)
        asyncio.run(server.run())
    except HealthCheckError as exc:
        _print_tagged("SYSTEM", "HEALTH", str(exc), error=True)
        sys.exit(2)
    except KeyboardInterrupt:
        _print_tagged("SYSTEM", "STOP", "interrupted")


@app.command
def stop() -> None:
    """Stop the voice daemon."""
    result = asyncio.run(_send_ipc("stop"))
    if result.get("status") == "ok":
        _print_tagged("SYSTEM", "STOP", "daemon stopping...")
    else:
        _print_tagged("SYSTEM", "ERROR", str(result.get("message", "unknown error")), error=True)


@app.command
def status() -> None:
    """Show daemon status."""
    result = asyncio.run(_send_ipc("status"))

    if result.get("status") != "ok":
        _print_tagged("SYSTEM", "ERROR", str(result.get("message", "unknown error")), error=True)
        return

    if not isinstance(raw := result.get("data"), dict):
        _print_tagged("SYSTEM", "ERROR", "invalid response data", error=True)
        return

    table = Table(title="eclaw daemon status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Running", str(raw.get("running", False)))
    table.add_row("Phase", str(raw.get("phase", "unknown")))
    table.add_row("Agent", str(raw.get("agent_name", "")))
    table.add_row("Backend", str(raw.get("agent_backend", "")))
    table.add_row("Wakeword", str(raw.get("wakeword", "")))

    console.print(table)


@app.command
def health() -> None:
    """Check health of dependent services."""
    from e_clawhisper.health import check_services

    try:
        config = AppConfig.load()
    except ValidationError as exc:
        _print_tagged("SYSTEM", "ERROR", f"invalid config: {exc.error_count()} errors", error=True)
        sys.exit(1)

    result = asyncio.run(check_services(config))

    for svc in result.services:
        if svc.healthy:
            _print_tagged("HEALTH", "OK", f"{svc.name} {svc.detail}")
        else:
            _print_tagged("HEALTH", "FAIL", f"{svc.name} {svc.detail}", error=True)

    if not result.healthy:
        sys.exit(1)


@app.command
def logs(*, lines: int = 50) -> None:
    """Follow daemon logs in real time (journalctl wrapper)."""
    os.execvp("journalctl", ["journalctl", "--user-unit", "e-clawhisper", "-f", "-n", str(lines), "--no-hostname", "-o", "short-iso"])
