"""Tests for GenericAdapter — model factory, connect, disconnect, history."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai.models.google import GoogleModel

from e_clawhisper.daemon.adapters.agent.generic import GenericAdapter, _build_model
from e_clawhisper.shared.settings import GenericLLMConfig, LLMProvider


##### BUILD MODEL #####


def test_build_model_gemini_default() -> None:
    cfg = GenericLLMConfig(provider=LLMProvider.GEMINI, model="default")
    result = _build_model(cfg)
    assert isinstance(result, GoogleModel)


def test_build_model_gemini_custom() -> None:
    cfg = GenericLLMConfig(provider=LLMProvider.GEMINI, model="gemini-1.5-pro")
    result = _build_model(cfg)
    assert isinstance(result, GoogleModel)


def test_build_model_openai_default() -> None:
    cfg = GenericLLMConfig(provider=LLMProvider.OPENAI, model="default")
    result = _build_model(cfg)
    assert result == "openai:gpt-4o-mini"


def test_build_model_anthropic_default() -> None:
    cfg = GenericLLMConfig(provider=LLMProvider.ANTHROPIC, model="default")
    result = _build_model(cfg)
    assert "claude" in result


@patch("e_clawhisper.daemon.adapters.agent.generic.OpenAIProvider")
@patch("e_clawhisper.daemon.adapters.agent.generic.OpenAIChatModel")
def test_build_model_vllm(mock_chat: MagicMock, mock_provider: MagicMock) -> None:
    cfg = GenericLLMConfig(
        provider=LLMProvider.VLLM,
        model="mistral-7b",
        url="http://localhost:8000",
        api_key="test-key",
    )
    _build_model(cfg)
    mock_provider.assert_called_once()
    call_kwargs = mock_provider.call_args
    assert "/v1" in call_kwargs[1]["base_url"]
    mock_chat.assert_called_once_with("mistral-7b", provider=mock_provider.return_value)


@patch("e_clawhisper.daemon.adapters.agent.generic.OpenAIProvider")
@patch("e_clawhisper.daemon.adapters.agent.generic.OpenAIChatModel")
def test_build_model_vllm_no_key(mock_chat: MagicMock, mock_provider: MagicMock) -> None:
    cfg = GenericLLMConfig(provider=LLMProvider.VLLM, model="llama")
    _build_model(cfg)
    call_kwargs = mock_provider.call_args
    assert call_kwargs[1]["api_key"] == "no-key"


##### ADAPTER INIT #####


def test_adapter_init() -> None:
    cfg = GenericLLMConfig()
    adapter = GenericAdapter(cfg)
    assert adapter.agent_id == ""
    assert adapter._connected is False


##### CONNECT / DISCONNECT #####


@patch("e_clawhisper.daemon.adapters.agent.generic.Agent")
async def test_connect_creates_agent(mock_agent_cls: MagicMock) -> None:
    cfg = GenericLLMConfig(provider=LLMProvider.GEMINI, model="gemini-2.0-flash")
    adapter = GenericAdapter(cfg)
    await adapter.connect("test-id")

    assert adapter.agent_id == "test-id"
    assert adapter._connected is True
    assert await adapter.is_connected()
    mock_agent_cls.assert_called_once()


@patch("e_clawhisper.daemon.adapters.agent.generic.Agent")
async def test_disconnect_clears_state(mock_agent_cls: MagicMock) -> None:
    cfg = GenericLLMConfig()
    adapter = GenericAdapter(cfg)
    await adapter.connect("test-id")
    adapter._message_history.append(MagicMock())

    await adapter.disconnect()

    assert adapter._connected is False
    assert adapter._agent is None
    assert adapter._message_history == []
    assert not await adapter.is_connected()


##### SEND #####


async def test_send_raises_without_connection() -> None:
    adapter = GenericAdapter(GenericLLMConfig())
    with pytest.raises(ConnectionError, match="not connected"):
        async for _ in adapter.send("hello"):
            pass


##### CLEAR HISTORY #####


@patch("e_clawhisper.daemon.adapters.agent.generic.Agent")
async def test_clear_history(mock_agent_cls: MagicMock) -> None:
    adapter = GenericAdapter(GenericLLMConfig())
    await adapter.connect("x")
    adapter._message_history.append(MagicMock())
    adapter.clear_history()
    assert adapter._message_history == []


##### RESOLVE #####


async def test_resolve_agent_id() -> None:
    cfg = GenericLLMConfig(provider=LLMProvider.GEMINI, model="flash")
    adapter = GenericAdapter(cfg)
    result = await adapter.resolve_agent_id("test-agent")
    assert "test-agent" in result
    assert "gemini" in result


##### SEND WITH AGENT #####


@patch("e_clawhisper.daemon.adapters.agent.generic.Agent")
async def test_send_yields_streamed_text(mock_agent_cls: MagicMock) -> None:
    adapter = GenericAdapter(GenericLLMConfig())
    await adapter.connect("test-id")

    async def _stream_text(delta: bool = True):  # noqa: ANN001
        yield "hello "
        yield "world"

    mock_result = MagicMock()
    mock_result.stream_text = _stream_text
    mock_result.all_messages.return_value = [MagicMock()]

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_result)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    adapter._agent.run_stream = MagicMock(return_value=mock_ctx)

    chunks = [c async for c in adapter.send("hi")]
    assert chunks == ["hello ", "world"]
    assert len(adapter._message_history) == 1


##### UNSUPPORTED PROVIDER #####


def test_build_model_unsupported_raises() -> None:
    cfg = GenericLLMConfig(provider=LLMProvider.GEMINI, model="x")
    cfg.provider = "unsupported"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported"):
        _build_model(cfg)
