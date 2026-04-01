"""Generic LLM agent adapter — multi-provider streaming via PydanticAI."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from e_heed.daemon.adapters.agent.base import AgentAdapter
from e_heed.shared.logger import logger
from e_heed.shared.settings import GenericLLMConfig, LLMProvider

##### MODEL FACTORY #####

_PROVIDER_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "gemini-2.0-flash",
    LLMProvider.OPENAI: "openai:gpt-4o-mini",
    LLMProvider.ANTHROPIC: "anthropic:claude-sonnet-4-20250514",
}

# Env var name per provider — PydanticAI reads these automatically
_API_KEY_ENV: dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "GOOGLE_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


def _resolve_api_key(config: GenericLLMConfig) -> str:
    """Resolve API key: config → env var → empty."""
    if config.api_key:
        return config.api_key
    env_name = _API_KEY_ENV.get(config.provider, "")
    return os.environ.get(env_name, "") if env_name else ""


def _build_model(config: GenericLLMConfig) -> Model | str:
    """Build PydanticAI model from config."""
    api_key = _resolve_api_key(config)
    match config.provider:
        case LLMProvider.VLLM:
            provider = OpenAIProvider(
                base_url=f"{str(config.url).rstrip('/')}/v1",
                api_key=api_key or "no-key",
            )
            return OpenAIChatModel(config.model, provider=provider)
        case LLMProvider.GEMINI:
            model_name = config.model if config.model != "default" else _PROVIDER_MODELS[LLMProvider.GEMINI]
            provider = GoogleProvider(api_key=api_key or None)
            return GoogleModel(model_name, provider=provider)
        case LLMProvider.OPENAI | LLMProvider.ANTHROPIC:
            if api_key:
                env_name = _API_KEY_ENV[config.provider]
                os.environ.setdefault(env_name, api_key)
            return config.model if config.model != "default" else _PROVIDER_MODELS[config.provider]
        case _:
            msg = f"Unsupported LLM provider: {config.provider}"
            raise ValueError(msg)


##### ADAPTER #####


class GenericAdapter(AgentAdapter):
    """Multi-provider LLM adapter — Gemini, OpenAI, Anthropic, vLLM."""

    __slots__ = ("_config", "_agent", "_agent_id", "_connected", "_message_history")

    def __init__(self, config: GenericLLMConfig) -> None:
        self._config = config
        self._agent: Agent[None, str] | None = None
        self._agent_id: str = ""
        self._connected: bool = False
        self._message_history: list[ModelMessage] = []

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
        self._message_history.clear()
        logger.system("STOP", "LLM disconnected")

    async def is_connected(self) -> bool:
        return self._connected and self._agent is not None

    ##### MESSAGING #####

    async def send(self, text: str) -> AsyncIterator[str]:
        if not self._agent:
            msg = "LLM not connected"
            raise ConnectionError(msg)

        history: Sequence[ModelMessage] | None = self._message_history or None
        async with self._agent.run_stream(text, message_history=history) as result:
            async for delta in result.stream_text(delta=True):
                yield delta
            self._message_history = result.all_messages()

    def clear_history(self) -> None:
        self._message_history.clear()

    ##### RESOLUTION #####

    async def resolve_agent_id(self, agent_name: str) -> str:
        agent_id = f"{agent_name}@{self._config.provider}:{self._config.model}"
        logger.system("OK", f"resolved agent '{agent_name}' → {agent_id}")
        return agent_id
