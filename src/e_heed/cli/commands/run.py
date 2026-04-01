"""Run command — start the full voice daemon foreground with agent backend override."""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated

import cyclopts
from pydantic import ValidationError
from rich.console import Console

from e_heed.daemon.server import DaemonServer
from e_heed.shared.operational.exceptions import HealthCheckError
from e_heed.shared.settings import AgentBackend, AppConfig

app = cyclopts.App(name="run", help="Start the full voice daemon (foreground).")
console = Console(stderr=True)


def _print_tagged(pipeline: str, step: str, msg: str, *, error: bool = False) -> None:
    style = "red" if error else "green"
    console.print(f"[cyan]\\[{pipeline}][/cyan] [bold {style}]\\[{step}][/bold {style}] {msg}")


@app.default
def run(backend: Annotated[AgentBackend, cyclopts.Parameter(name="backend")] = AgentBackend.GENERIC) -> None:
    """Start the full daemon with the specified agent backend.

    Usage:
      eheed run              # generic (default)
      eheed run generic      # explicit generic (Gemini/OpenAI/etc.)
      eheed run openfang     # OpenFang agent OS
    """
    try:
        config = AppConfig.load()
    except ValidationError as exc:
        _print_tagged("SYSTEM", "ERROR", "invalid config.yaml", error=True)
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"])
            console.print(f"  [red]{loc}:[/red] {err['msg']}")
        sys.exit(1)

    config.agent.backend = backend

    try:
        server = DaemonServer(config)
        asyncio.run(server.run())
    except HealthCheckError as exc:
        _print_tagged("SYSTEM", "HEALTH", str(exc), error=True)
        sys.exit(2)
    except KeyboardInterrupt:
        _print_tagged("SYSTEM", "STOP", "interrupted")
