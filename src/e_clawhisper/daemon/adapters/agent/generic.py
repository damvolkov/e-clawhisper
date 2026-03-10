"""Generic LLM agent adapter — multi-provider streaming via PydanticAI."""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from e_clawhisper.daemon.adapters.agent.base import AgentAdapter
from e_clawhisper.shared.logger import logger
from e_clawhisper.shared.settings import GenericLLMConfig, LLMProvider

##### MODEL FACTORY #####

_PROVIDER_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "gemini-2.0-flash",
    LLMProvider.OPENAI: "openai:gpt-4o-mini",
    LLMProvider.ANTHROPIC: "anthropic:claude-sonnet-4-20250514",
}


def _build_model(config: GenericLLMConfig) -> Model | str:
    """Build PydanticAI model from config — vLLM uses custom OpenAI provider."""
    match config.provider:
        case LLMProvider.VLLM:
            provider = OpenAIProvider(
                base_url=f"http://{config.host}:{config.port}/v1",
                api_key=config.api_key or "no-key",
            )
            return OpenAIChatModel(config.model, provider=provider)
        case LLMProvider.GEMINI | LLMProvider.OPENAI | LLMProvider.ANTHROPIC:
            return config.model if config.model != "default" else _PROVIDER_MODELS[config.provider]
        case _:
            msg = f"Unsupported LLM provider: {config.provider}"
            raise ValueError(msg)


##### ADAPTER #####


class GenericAdapter(AgentAdapter):
    """Multi-provider LLM adapter — Gemini, OpenAI, Anthropic, vLLM."""

    __slots__ = ("_config", "_agent", "_agent_id", "_connected")

    def __init__(self, config: GenericLLMConfig) -> None:
        self._config = config
        self._agent: Agent[None, str] | None = None
        self._agent_id: str = ""
        self._connected: bool = False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    ##### CONNECTION #####

    async def connect(self, agent_id: str) -> None:
        self._agent_id = agent_id
        model = _build_model(self._config)
        self._agent = Agent(
            model=model,
            system_prompt=self._config.system_prompt,
        )
        self._connected = True
        logger.system("OK", f"LLM connected provider={self._config.provider} model={self._config.model}")

    async def disconnect(self) -> None:
        self._agent = None
        self._connected = False
        logger.system("STOP", "LLM disconnected")

    async def is_connected(self) -> bool:
        return self._connected and self._agent is not None

    ##### MESSAGING #####

    async def send(self, text: str) -> AsyncIterator[str]:
        if not self._agent:
            msg = "LLM not connected"
            raise ConnectionError(msg)

        async with self._agent.run_stream(text) as result:
            async for delta in result.stream_text(delta=True):
                yield delta

    ##### RESOLUTION #####

    async def resolve_agent_id(self, agent_name: str) -> str:
        agent_id = f"{agent_name}@{self._config.provider}:{self._config.model}"
        logger.system("OK", f"resolved agent '{agent_name}' → {agent_id}")
        return agent_id
