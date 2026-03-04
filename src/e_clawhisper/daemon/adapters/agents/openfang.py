"""OpenFang agent adapter — WebSocket streaming communication."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import orjson
import websockets
from websockets.asyncio.client import ClientConnection

from e_clawhisper.daemon.core.interfaces.agent import AgentBase
from e_clawhisper.shared.logger import LogIcon, logger
from e_clawhisper.shared.settings import OpenFangConfig


class OpenFangAdapter(AgentBase):
    """Connects to OpenFang agent via WebSocket for streaming conversation.

    Protocol:
        - Endpoint: ws://{host}:{port}/api/agents/{agent_id}/ws
        - On connect: receive {"type": "connected", "agent_id": "..."}
        - Send: {"type": "message", "content": "user text"}
        - Receive: text_delta (streaming chunks), response (final)
        - Resolve agent ID: GET /api/agents → filter by name
    """

    def __init__(self, config: OpenFangConfig) -> None:
        self._host = config.host
        self._port = config.port
        self._timeout = config.timeout
        self._base_url = f"http://{config.host}:{config.port}"
        self._ws: ClientConnection | None = None
        self._agent_id: str = ""

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def connect(self, agent_id: str) -> None:
        self._agent_id = agent_id
        ws_url = f"ws://{self._host}:{self._port}/api/agents/{agent_id}/ws"
        self._ws = await websockets.connect(ws_url)

        connected_msg = await self._ws.recv()
        data = orjson.loads(connected_msg)
        if data.get("type") != "connected":
            logger.warning("openfang unexpected connect: %s", data, icon=LogIcon.AGENT)

        logger.info("openfang_connected agent_id=%s", agent_id[:12], icon=LogIcon.AGENT)

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("openfang_disconnected", icon=LogIcon.AGENT)

    async def is_connected(self) -> bool:
        return self._ws is not None

    async def send_message(self, text: str) -> AsyncIterator[str]:
        """Send message to OpenFang, yield streaming response chunks."""
        if not self._ws:
            msg = "OpenFang WebSocket not connected"
            raise ConnectionError(msg)

        payload = orjson.dumps({"type": "message", "content": text})
        await self._ws.send(payload)
        logger.debug("openfang_send: %s", text[:100], icon=LogIcon.AGENT)

        full_response = ""
        async for raw in self._ws:
            data = orjson.loads(raw)
            msg_type = data.get("type", "")

            match msg_type:
                case "text_delta":
                    chunk = data.get("content", "")
                    full_response += chunk
                    yield chunk
                case "response":
                    final = data.get("content", "")
                    if final and not full_response:
                        yield final
                    break
                case "typing":
                    continue
                case "tool_start" | "tool_result":
                    logger.debug("openfang_tool: %s", msg_type, icon=LogIcon.AGENT)
                    continue
                case _:
                    logger.debug("openfang_unknown: %s", msg_type, icon=LogIcon.AGENT)

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
