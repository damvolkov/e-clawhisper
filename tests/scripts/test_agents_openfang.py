"""Interactive WebSocket streaming chat with OpenFang agents."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import TYPE_CHECKING

import websockets

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

BASE_URL = "ws://127.0.0.1:4200/api/agents"

# ping disabled + no close timeout = nunca se cae por inactividad
WS_KWARGS: dict = {
    "max_size": 2**22,
    "ping_interval": None,
    "ping_timeout": None,
    "close_timeout": None,
}

AGENTS: dict[str, str] = {
    "damien": "9c031c0b-94d0-4f29-b48b-9711fc5740e0",
    "tatan": "c99cb8fd-a492-4943-9bdf-0e3df4c45c20",
    "instructor": "66da1cbe-9644-41c9-b989-f9e0abbc7b62",
    "assistant": "66fe65fe-227f-4373-b6e6-68442b41c61c",
}


async def _drain_until_connected(ws: ClientConnection) -> None:
    """Consume initial 'connected' frame."""
    raw = await asyncio.wait_for(ws.recv(), timeout=10)
    frame = json.loads(raw)
    if frame.get("type") != "connected":
        print(f"  ⚠ Unexpected init frame: {frame}")


async def _stream_response(ws: ClientConnection, agent_name: str) -> str:
    """Consume streaming frames until full response arrives."""
    chunks: list[str] = []

    async for raw in ws:
        frame = json.loads(raw)
        msg_type = frame.get("type", "")

        match msg_type:
            case "text_delta":
                chunk = frame.get("content", "")
                print(chunk, end="", flush=True)
                chunks.append(chunk)

            case "phase":
                match frame.get("phase", ""):
                    case "thinking":
                        print("  💭 pensando...", end="", flush=True)
                    case _:
                        pass

            case "tool_start":
                tool_name = frame.get("tool", frame.get("name", "?"))
                print(f"\n  ⚙ [{tool_name}] ", end="", flush=True)

            case "tool_result":
                status = frame.get("status", "done")
                print(f"→ {status}", flush=True)

            case "response":
                print(flush=True)
                return "".join(chunks)

            case "error":
                err = frame.get("content", frame.get("message", str(frame)))
                print(f"\n✗ Error: {err}", flush=True)
                return ""

            case "typing" | "agents_updated":
                pass

            case _:
                pass

    return "".join(chunks)


async def _connect_with_retry(url: str, max_retries: int = 3) -> ClientConnection:
    """Connect with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            return await websockets.connect(url, **WS_KWARGS)
        except (OSError, websockets.exceptions.WebSocketException) as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt
            print(f"  ⚠ Conexión fallida ({exc}), reintentando en {wait}s...")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def chat_loop(agent_name: str) -> None:
    agent_id = AGENTS.get(agent_name)
    if not agent_id:
        print(f"✗ Agente '{agent_name}' no encontrado. Disponibles: {', '.join(AGENTS)}")
        return

    url = f"{BASE_URL}/{agent_id}/ws"
    print(f"Conectando a {agent_name} ({agent_id[:8]}...) en {url}")
    print("Escribe 'exit' o Ctrl+C para salir.\n")

    ws = await _connect_with_retry(url)
    try:
        await _drain_until_connected(ws)
        print(f"✓ Conectado a {agent_name}.\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "[tú] → ")
            except (EOFError, KeyboardInterrupt):
                print("\n\nDesconectando...")
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in {"exit", "quit", "salir"}:
                print("Chao.")
                break

            # Reconnect if WS dropped mid-session
            if ws.close_code is not None:
                print("  ⚠ Reconectando...")
                ws = await _connect_with_retry(url)
                await _drain_until_connected(ws)
                print("  ✓ Reconectado.\n")

            await ws.send(stripped)
            print(f"\n[{agent_name}] → ", end="", flush=True)

            try:
                await _stream_response(ws, agent_name)
            except websockets.exceptions.ConnectionClosed:
                print("\n  ⚠ Conexión perdida. Reconectando...")
                ws = await _connect_with_retry(url)
                await _drain_until_connected(ws)
                print("  ✓ Reconectado. Reenvía tu mensaje.\n")

            print()
    finally:
        await ws.close()


def main() -> None:
    agent_name = sys.argv[1] if len(sys.argv) > 1 else "damien"
    agent_name = agent_name.lower()

    try:
        asyncio.run(chat_loop(agent_name))
    except KeyboardInterrupt:
        print("\nAbortado.")


if __name__ == "__main__":
    main()
