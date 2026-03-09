"""Generic LLM agent adapter — OpenAI-compatible SSE streaming via httpx."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import orjson

from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.settings import GenericLLMConfig


class GenericAdapter:
    """OpenAI-compatible chat completions with SSE streaming."""

    __slots__ = ("_base_url", "_model", "_timeout", "_system_prompt", "_api_key", "_client", "_messages", "_agent_id")

    def __init__(self, config: GenericLLMConfig) -> None:
        self._base_url = f"http://{config.host}:{config.port}"
        self._model = config.model
        self._timeout = config.timeout
        self._system_prompt = config.system_prompt
        self._api_key = config.api_key
        self._client: httpx.AsyncClient | None = None
        self._messages: list[dict[str, str]] = []
        self._agent_id: str = ""

    @property
    def agent_id(self) -> str:
        return self._agent_id

    ##### CONNECTION #####

    async def connect(self, agent_id: str) -> None:
        self._agent_id = agent_id
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=self._timeout, write=10.0, pool=10.0),
        )
        self._messages = [{"role": "system", "content": self._system_prompt}]
        logger.system("OK", f"generic LLM connected model={self._model}")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._messages.clear()
        logger.system("STOP", "generic LLM disconnected")

    async def is_connected(self) -> bool:
        return self._client is not None

    ##### MESSAGING #####

    async def send(self, text: str) -> AsyncIterator[str]:
        if not self._client:
            msg = "generic LLM not connected"
            raise ConnectionError(msg)

        self._messages.append({"role": "user", "content": text})

        payload = {
            "model": self._model,
            "messages": list(self._messages),
            "stream": True,
        }

        full_response = ""
        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    chunk = orjson.loads(data)
                    choices = chunk.get("choices") or [{}]
                    if content := choices[0].get("delta", {}).get("content"):
                        full_response += content
                        yield content
        finally:
            if full_response:
                self._messages.append({"role": "assistant", "content": full_response})
            else:
                self._messages.pop()

    ##### RESOLUTION #####

    async def resolve_agent_id(self, agent_name: str) -> str:
        agent_id = f"{agent_name}@{self._model}"
        logger.system("OK", f"resolved generic agent '{agent_name}' → {agent_id}")
        return agent_id
