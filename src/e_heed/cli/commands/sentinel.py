"""Sentinel command — run only the wakeword detection pipeline (foreground)."""

from __future__ import annotations

import asyncio
import sys

import cyclopts
from pydantic import ValidationError
from rich.console import Console

from e_heed.shared.settings import AppConfig

app = cyclopts.App(name="sentinel", help="Run the sentinel pipeline only (wakeword testing).")
console = Console(stderr=True)


def _print_tagged(pipeline: str, step: str, msg: str, *, error: bool = False) -> None:
    style = "red" if error else "green"
    console.print(f"[cyan]\\[{pipeline}][/cyan] [bold {style}]\\[{step}][/bold {style}] {msg}")


async def _run_sentinel_loop(config: AppConfig) -> None:
    """Standalone sentinel loop — detects wakewords, logs, resets, repeats."""
    from e_heed.daemon.adapters.audio import AudioAdapter
    from e_heed.daemon.sentinel.pipeline import SentinelPipeline
    from e_heed.shared.logger import configure_logging, logger

    configure_logging(
        "debug",
        idle_interval=config.logging.idle_interval,
        turn_interval=config.logging.turn_interval,
    )

    audio = AudioAdapter(config.audio)
    sentinel = SentinelPipeline(config.sentinel)

    logger.system("START", "starting audio device...")
    await audio.start()

    logger.system("START", f"sentinel-only mode — listening for '{sentinel.wakeword_name}'")

    try:
        while True:
            await sentinel.run(audio.queue)

            if sentinel.last_event:
                event = sentinel.last_event
                logger.sentinel(
                    "WAKEWORD",
                    f"detected! confidence={event.confidence:.2f} energy={event.energy:.4f}",
                )
    except asyncio.CancelledError:
        pass
    finally:
        await sentinel.stop()
        await audio.stop()
        logger.system("STOP", "sentinel stopped")


@app.default
def sentinel() -> None:
    """Run sentinel pipeline only — listens for wakewords without activating STT/LLM/TTS.

    Useful for testing wakeword detection, VAD thresholds, and energy gate tuning.
    No agent, STT, or TTS backends required.

    Usage:
      eheed sentinel
    """
    try:
        config = AppConfig.load()
    except ValidationError as exc:
        _print_tagged("SYSTEM", "ERROR", "invalid config.yaml", error=True)
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"])
            console.print(f"  [red]{loc}:[/red] {err['msg']}")
        sys.exit(1)

    try:
        asyncio.run(_run_sentinel_loop(config))
    except KeyboardInterrupt:
        _print_tagged("SYSTEM", "STOP", "interrupted")
