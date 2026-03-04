"""Session commands — start / stop / status."""

from __future__ import annotations

import asyncio

import cyclopts
import orjson
from rich.console import Console
from rich.table import Table

from e_clawhisper.shared.settings import settings

app = cyclopts.App(name="session", help="Daemon lifecycle commands.")
console = Console()


async def _send_ipc(command: str) -> dict:
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


@app.command
def start() -> None:
    """Start the voice daemon (foreground)."""
    from e_clawhisper.daemon.server import DaemonServer
    from e_clawhisper.shared.logger import configure_logging
    from e_clawhisper.shared.settings import load_config

    configure_logging(settings.LOG_LEVEL)
    config = load_config()
    server = DaemonServer(config)
    asyncio.run(server.run())


@app.command
def stop() -> None:
    """Stop the voice daemon."""
    result = asyncio.run(_send_ipc("stop"))
    if result.get("status") == "ok":
        console.print("[green]Daemon stopping...[/green]")
    else:
        console.print(f"[red]{result.get('message', 'unknown error')}[/red]")


@app.command
def status() -> None:
    """Show daemon status."""
    result = asyncio.run(_send_ipc("status"))

    if result.get("status") != "ok":
        console.print(f"[red]{result.get('message', 'unknown error')}[/red]")
        return

    data = result.get("data", {})
    table = Table(title="eclaw daemon status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Running", str(data.get("running", False)))
    table.add_row("Pipeline State", data.get("pipeline_state", "unknown"))
    table.add_row("Conversation Active", str(data.get("conversation_active", False)))
    table.add_row("Agent Name", data.get("agent_name", ""))
    table.add_row("Agent Backend", data.get("agent_backend", ""))

    console.print(table)
