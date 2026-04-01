"""Standalone sentinel test script — reuses project SentinelPipeline.

Usage:
  make script sentinel          # uses config.yaml wakeword
  make script sentinel maia     # override model
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

from e_heed.daemon.adapters.audio import AudioAdapter
from e_heed.daemon.sentinel.pipeline import SentinelPipeline
from e_heed.shared.logger import configure_logging, logger
from e_heed.shared.settings import load_config


async def _run(model_override: str | None = None) -> None:
    config = load_config()
    configure_logging(
        level="debug", idle_interval=config.logging.idle_interval, turn_interval=config.logging.turn_interval
    )

    if model_override:
        config.sentinel.wakeword.model = model_override

    sentinel = SentinelPipeline(config.sentinel)
    audio = AudioAdapter(config.audio)

    logger.system("START", f"sentinel script model='{sentinel.wakeword_name}'")

    await audio.start()
    try:
        while True:
            await sentinel.run(audio.queue)
            if sentinel.last_event:
                logger.sentinel(
                    "WAKEWORD",
                    f"'{sentinel.wakeword_name}' detected — continuing",
                    ww=f"{sentinel.last_event.confidence:.2f}",
                )
            sentinel.wakeword_detected.clear()
    except asyncio.CancelledError:
        pass
    finally:
        await sentinel.stop()
        await audio.stop()
        logger.system("STOP", "sentinel script stopped")


##### MAIN #####

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run(model))
