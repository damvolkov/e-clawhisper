"""OpenFang agent adapter — persistent WebSocket with background receive."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import httpx
import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from e_clawhisper.daemon.core.interfaces.agent import AgentBase
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.settings import OpenFangConfig

_RESPONSE_TYPES = frozenset({"text_delta", "response", "typing", "tool_start", "tool_result", "phase"})


class OpenFangAdapter(AgentBase):
    """Persistent WebSocket to OpenFang agent with background message dispatch.

    Background recv loop separates unsolicited messages (agents_updated,
    pings) from response messages, which are queued for send_message().
    """

    __slots__ = ("_host", "_port", "_timeout", "_base_url", "_ws", "_agent_id", "_response_queue", "_recv_task")

    def __init__(self, config: OpenFangConfig) -> None:
        self._host = config.host
        self._port = config.port
        self._timeout = config.timeout
        self._base_url = f"http://{config.host}:{config.port}"
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
        ws_url = f"ws://{self._host}:{self._port}/api/agents/{agent_id}/ws"
        self._ws = await websockets.connect(ws_url)

        connected_msg = await self._ws.recv()
        data = orjson.loads(connected_msg)
        if data.get("type") != "connected":
            logger.warning("openfang unexpected connect: %s", data, icon=LogIcon.AGENT)

        self._recv_task = asyncio.create_task(self._receive_loop())
        logger.info("openfang_connected agent_id=%s", agent_id[:12], icon=LogIcon.AGENT)

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
            logger.info("openfang_disconnected", icon=LogIcon.AGENT)

    async def is_connected(self) -> bool:
        return self._ws is not None and self._recv_task is not None

    ##### RECEIVE LOOP #####

    async def _receive_loop(self) -> None:
        """Background loop: dispatch incoming messages by type."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                data: dict[str, object] = orjson.loads(raw)
                msg_type = data.get("type", "")

                if msg_type in _RESPONSE_TYPES:
                    await self._response_queue.put(data)
                elif msg_type in ("agents_updated", "connected"):
                    logger.debug("openfang_broadcast: %s (ignored)", msg_type, icon=LogIcon.AGENT)
                else:
                    logger.debug("openfang_unknown: %s", msg_type, icon=LogIcon.AGENT)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("openfang_connection_lost — reconnect needed", icon=LogIcon.AGENT)

    ##### MESSAGING #####

    async def send_message(self, text: str) -> AsyncIterator[str]:
        """Send plain text message, yield streaming response chunks from queue."""
        if not self._ws:
            msg = "OpenFang WebSocket not connected"
            raise ConnectionError(msg)

        while not self._response_queue.empty():
            self._response_queue.get_nowait()

        await self._ws.send(text)
        logger.debug("openfang_send: %s", text[:100], icon=LogIcon.AGENT)

        full_response = ""
        while True:
            try:
                data = await asyncio.wait_for(self._response_queue.get(), timeout=self._timeout)
            except TimeoutError:
                logger.warning("openfang response timeout after %.0fs", self._timeout, icon=LogIcon.AGENT)
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
                    logger.debug("openfang_tool: %s", data.get("type"), icon=LogIcon.AGENT)

    ##### RESOLUTION #####

    async def resolve_agent_id(self, agent_name: str) -> str:
        """GET /api/agents and find agent by name."""
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            resp = await client.get("/api/agents")
            resp.raise_for_status()
            agents = resp.json()

            if isinstance(agents, dict):
                agents = agents.get("agents", [])

            for agent in agents:
                if agent.get("name", "").lower() == agent_name.lower():
                    agent_id = agent["id"]
                    logger.info("resolved agent '%s' -> %s", agent_name, agent_id[:12], icon=LogIcon.AGENT)
                    return str(agent_id)

        msg = f"Agent '{agent_name}' not found on OpenFang at {self._base_url}"
        raise ValueError(msg)
