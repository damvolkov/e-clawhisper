"""OpenFang agent adapter — persistent WebSocket with background receive."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import httpx
import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.settings import OpenFangConfig

_RESPONSE_TYPES = frozenset({"text_delta", "response", "typing", "tool_start", "tool_result", "phase"})


class OpenfangAdapter:
    """Persistent WebSocket to OpenFang agent with background message dispatch."""

    __slots__ = ("_timeout", "_base_url", "_ws_base", "_ws", "_agent_id", "_response_queue", "_recv_task")

    def __init__(self, config: OpenFangConfig) -> None:
        self._timeout = config.timeout
        self._base_url = str(config.url).rstrip("/")
        self._ws_base = config.ws_base
        self._ws: ClientConnection | None = None
        self._agent_id: str = ""
        self._response_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._recv_task: asyncio.Task[None] | None = None

    @property
    def agent_id(self) -> str:
        return self._agent_id

    ##### CONNECTION #####

    async def connect(self, agent_id: str) -> None:
        self._agent_id = agent_id
        ws_url = f"{self._ws_base}/api/agents/{agent_id}/ws"
        self._ws = await websockets.connect(ws_url)

        connected_msg = await self._ws.recv()
        data = orjson.loads(connected_msg)
        if data.get("type") != "connected":
            logger.warning(f"agent unexpected connect: {data}")

        self._recv_task = asyncio.create_task(self._receive_loop())
        logger.system("OK", f"agent connected agent_id={agent_id[:12]}")

    async def disconnect(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        if self._ws:
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await self._ws.close()
            self._ws = None
            logger.system("STOP", "agent disconnected")

    async def is_connected(self) -> bool:
        return self._ws is not None and self._recv_task is not None

    ##### RECEIVE #####

    async def _receive_loop(self) -> None:
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                data: dict[str, object] = orjson.loads(raw)
                msg_type = data.get("type", "")

                if msg_type in _RESPONSE_TYPES:
                    await self._response_queue.put(data)
                elif msg_type not in ("agents_updated", "connected"):
                    logger.turn_debug("AGENT", f"unknown: {msg_type}")
        except websockets.exceptions.ConnectionClosed:
            self._ws = None
            await self._response_queue.put({"type": "connection_lost"})
            logger.warning("agent connection lost")
        except Exception as exc:
            self._ws = None
            await self._response_queue.put({"type": "connection_lost"})
            logger.error(f"agent receive error: {exc}")

    ##### MESSAGING #####

    async def send(self, text: str) -> AsyncIterator[str]:
        if not self._ws:
            msg = "agent WebSocket not connected"
            raise ConnectionError(msg)

        while not self._response_queue.empty():
            self._response_queue.get_nowait()

        await self._ws.send(text)

        full_response = ""
        while True:
            try:
                data = await asyncio.wait_for(self._response_queue.get(), timeout=self._timeout)
            except TimeoutError:
                logger.warning(f"agent response timeout after {self._timeout:.0f}s")
                break

            match data.get("type", ""):
                case "text_delta":
                    chunk = str(data.get("content", ""))
                    full_response += chunk
                    yield chunk
                case "response":
                    if (final := str(data.get("content", ""))) and not full_response:
                        yield final
                    break
                case "typing" | "phase":
                    continue
                case "tool_start" | "tool_result":
                    logger.turn_debug("AGENT", f"tool: {data.get('type')}")
                case "connection_lost":
                    logger.warning("agent connection lost during response")
                    break

    ##### RESOLUTION #####

    async def resolve_agent_id(self, agent_name: str) -> str:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.get("/api/agents")
            resp.raise_for_status()
            agents = resp.json()

            if isinstance(agents, dict):
                agents = agents.get("agents", [])

            for agent in agents:
                if agent.get("name", "").lower() == agent_name.lower():
                    agent_id = agent["id"]
                    logger.system("OK", f"resolved agent '{agent_name}' → {agent_id[:12]}")
                    return str(agent_id)

        msg = f"Agent '{agent_name}' not found on OpenFang at {self._base_url}"
        raise ValueError(msg)
