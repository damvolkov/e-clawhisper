"""OpenFang channel — HTTP POST and WebSocket communication."""

from __future__ import annotations

import httpx
import orjson

from e_clawhisper.channels.base import BaseChannel
from e_clawhisper.core.logger import LogIcon, logger


class OpenFangChannel(BaseChannel):
    """Channel bridging to OpenFang agent backend.

    Uses HTTP POST for request-response (reliable).
    WebSocket URL exposed for future streaming upgrade.
    """

    def __init__(
        self,
        *,
        base_url: str,
        agent_id: str,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def message_url(self) -> str:
        return f"/api/agents/{self._agent_id}/message"

    @property
    def ws_url(self) -> str:
        ws_base = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{ws_base}/api/agents/{self._agent_id}/ws"

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )
        logger.info(
            "openfang_connected base_url=%s agent_id=%s",
            self._base_url,
            self._agent_id,
            icon=LogIcon.CHANNEL,
        )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("openfang_disconnected", icon=LogIcon.CHANNEL)

    async def is_connected(self) -> bool:
        return self._client is not None

    async def send_message(self, message: str) -> str:
        """POST to /api/agents/<id>/message and return response text."""
        if not self._client:
            await self.connect()

        assert self._client is not None

        logger.debug("openfang_send: %s", message[:100], icon=LogIcon.CHANNEL)

        response = await self._client.post(
            self.message_url,
            content=orjson.dumps({"message": message}),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        data = orjson.loads(response.content)
        reply = data.get("response", "")

        logger.debug("openfang_recv: %s", reply[:100], icon=LogIcon.CHANNEL)
        return reply
