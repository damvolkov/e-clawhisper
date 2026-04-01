"""Tests for Orchestrator — init, factories, phase, conversation, shutdown."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from e_heed.daemon.adapters.agent.generic import GenericAdapter
from e_heed.daemon.adapters.agent.openfang import OpenfangAdapter
from e_heed.daemon.adapters.stt.evoice import EVoiceSTTAdapter
from e_heed.daemon.adapters.stt.whisperlive import WhisperliveAdapter
from e_heed.daemon.adapters.tts.kokoro import KokoroAdapter
from e_heed.daemon.adapters.tts.piper import PiperAdapter
from e_heed.daemon.orchestrator import Orchestrator, PipelinePhase
from e_heed.shared.operational.events import TurnComplete, TurnError
from e_heed.shared.settings import AgentBackend, AppConfig, STTBackend, TTSBackend

##### HELPERS #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def _make_orchestrator(
    mock_turn: MagicMock,
    mock_sentinel: MagicMock,
    config: AppConfig | None = None,
) -> Orchestrator:
    cfg = config or AppConfig()
    orch = Orchestrator(cfg)
    return orch


##### PHASE ENUM #####


def test_phase_enum_values() -> None:
    assert PipelinePhase.SENTINEL == "sentinel"
    assert PipelinePhase.TURN == "turn"
    assert PipelinePhase.LOOP == "loop"


##### INIT & FACTORIES #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_init_default_creates_openfang(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.agent.backend = AgentBackend.OPENFANG
    orch = Orchestrator(cfg)
    assert isinstance(orch._agent, OpenfangAdapter)


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_init_generic_creates_generic(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.agent.backend = AgentBackend.GENERIC
    orch = Orchestrator(cfg)
    assert isinstance(orch._agent, GenericAdapter)


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_init_stt_evoice_default(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    assert isinstance(orch._stt, EVoiceSTTAdapter)


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_init_stt_whisperlive(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.stt.backend = STTBackend.WHISPERLIVE
    orch = Orchestrator(cfg)
    assert isinstance(orch._stt, WhisperliveAdapter)


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_init_tts_kokoro(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.tts.backend = TTSBackend.KOKORO
    orch = Orchestrator(cfg)
    assert isinstance(orch._tts, KokoroAdapter)


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_init_tts_piper(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.tts.backend = TTSBackend.PIPER
    orch = Orchestrator(cfg)
    assert isinstance(orch._tts, PiperAdapter)


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_phase_property(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    assert orch.phase == PipelinePhase.SENTINEL


##### SHOULD LOOP #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_should_loop_enabled(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.conversation.enabled = True
    cfg.conversation.max_turns = 5
    orch = Orchestrator(cfg)
    orch._conversation_turns = 0
    assert orch._should_loop() is True


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_should_loop_disabled(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.conversation.enabled = False
    orch = Orchestrator(cfg)
    assert orch._should_loop() is False


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_should_loop_max_reached(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.conversation.enabled = True
    cfg.conversation.max_turns = 3
    orch = Orchestrator(cfg)
    orch._conversation_turns = 3
    assert orch._should_loop() is False


##### END CONVERSATION #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_end_conversation_resets_state(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.agent.backend = AgentBackend.GENERIC
    orch = Orchestrator(cfg)
    orch._conversation_turns = 5
    orch._phase = PipelinePhase.TURN

    orch._end_conversation()

    assert orch._conversation_turns == 0
    assert orch._phase == PipelinePhase.SENTINEL


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_end_conversation_clears_generic_history(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.agent.backend = AgentBackend.GENERIC
    orch = Orchestrator(cfg)
    orch._conversation_turns = 1
    orch._agent._message_history.append(MagicMock())

    orch._end_conversation()

    assert orch._agent._message_history == []


##### STOP #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_stop(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._sentinel.stop = AsyncMock()
    orch._turn.stop = AsyncMock()
    orch._running = True

    await orch.stop()

    assert orch._running is False
    orch._sentinel.stop.assert_called_once()
    orch._turn.stop.assert_called_once()


##### ENSURE AGENT #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_ensure_agent_connected(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._agent = AsyncMock()
    orch._agent.is_connected = AsyncMock(return_value=True)

    assert await orch._ensure_agent() is True


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_ensure_agent_reconnects(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._agent = AsyncMock()
    orch._agent.is_connected = AsyncMock(return_value=False)
    orch._agent.agent_id = "test-id"
    orch._agent.connect = AsyncMock()
    orch._agent.disconnect = AsyncMock()

    assert await orch._ensure_agent() is True


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_ensure_agent_reconnect_fails(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._agent = AsyncMock()
    orch._agent.is_connected = AsyncMock(return_value=False)
    orch._agent.agent_id = "test-id"
    orch._agent.connect = AsyncMock(side_effect=ConnectionError("down"))
    orch._agent.disconnect = AsyncMock()

    assert await orch._ensure_agent() is False


##### SHUTDOWN #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_shutdown_closes_all(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._audio = AsyncMock()
    orch._stt = AsyncMock()
    orch._agent = AsyncMock()

    await orch._shutdown()

    orch._audio.stop.assert_called_once()
    orch._stt.disconnect.assert_called_once()
    orch._agent.disconnect.assert_called_once()


##### RUN SENTINEL #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_sentinel_transitions_to_turn(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._sentinel.run = AsyncMock()
    orch._sentinel.last_event = MagicMock()
    orch._running = True

    await orch._run_sentinel()

    assert orch._phase == PipelinePhase.TURN


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_sentinel_stays_if_no_event(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._sentinel.run = AsyncMock()
    orch._sentinel.last_event = None
    orch._running = True

    await orch._run_sentinel()

    assert orch._phase == PipelinePhase.SENTINEL


##### RUN TURN #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_turn_complete_loops(mock_turn_cls: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.conversation.enabled = True
    orch = Orchestrator(cfg)
    orch._agent = AsyncMock()
    orch._agent.is_connected = AsyncMock(return_value=True)
    orch._audio = MagicMock()

    result = TurnComplete(turn_id="t1", transcript="hi", response="hello", duration=1.0)
    orch._turn.run = AsyncMock(return_value=result)

    await orch._run_turn()

    assert orch._phase == PipelinePhase.LOOP
    assert orch._conversation_turns == 1


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_turn_error_ends_conversation(mock_turn_cls: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._agent = AsyncMock()
    orch._agent.is_connected = AsyncMock(return_value=True)
    orch._audio = MagicMock()

    result = TurnError(turn_id="t1", reason="empty_transcript")
    orch._turn.run = AsyncMock(return_value=result)

    await orch._run_turn()

    assert orch._phase == PipelinePhase.SENTINEL
    assert orch._conversation_turns == 0


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_turn_agent_disconnected_ends(mock_turn_cls: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._agent = AsyncMock()
    orch._agent.is_connected = AsyncMock(return_value=False)
    orch._agent.connect = AsyncMock(side_effect=ConnectionError())
    orch._agent.disconnect = AsyncMock()
    orch._audio = MagicMock()

    await orch._run_turn()

    assert orch._phase == PipelinePhase.SENTINEL


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_turn_complete_no_loop(mock_turn_cls: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.conversation.enabled = False
    orch = Orchestrator(cfg)
    orch._agent = AsyncMock()
    orch._agent.is_connected = AsyncMock(return_value=True)
    orch._audio = MagicMock()

    result = TurnComplete(turn_id="t1", transcript="hi", response="hello", duration=1.0)
    orch._turn.run = AsyncMock(return_value=result)

    await orch._run_turn()

    assert orch._phase == PipelinePhase.SENTINEL


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_turn_timeout(mock_turn_cls: MagicMock, mock_sentinel: MagicMock) -> None:
    import asyncio

    cfg = AppConfig()
    cfg.turn_timeout = 0.01  # very short
    orch = Orchestrator(cfg)
    orch._agent = AsyncMock()
    orch._agent.is_connected = AsyncMock(return_value=True)
    orch._audio = MagicMock()
    orch._turn.stop = AsyncMock()

    async def _slow_turn(audio: object) -> TurnComplete:
        await asyncio.sleep(10)
        return TurnComplete(turn_id="t1", transcript="hi", response="hello", duration=1.0)

    orch._turn.run = _slow_turn

    await orch._run_turn()

    assert orch._phase == PipelinePhase.SENTINEL
    orch._turn.stop.assert_called_once()


##### CONVERSATION LOOP #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_conversation_loop_voice_detected(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._sentinel.wait_for_voice = AsyncMock(return_value=True)
    orch._audio = MagicMock()
    orch._running = True

    await orch._run_conversation_loop()

    assert orch._phase == PipelinePhase.TURN


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_conversation_loop_timeout(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._sentinel.wait_for_voice = AsyncMock(return_value=False)
    orch._audio = MagicMock()
    orch._running = True

    await orch._run_conversation_loop()

    assert orch._phase == PipelinePhase.SENTINEL


##### FACTORY ERRORS #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_create_stt_unsupported_raises(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.stt.backend = "unsupported"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported STT"):
        Orchestrator(cfg)


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_create_tts_unsupported_raises(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.tts.backend = "unsupported"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported TTS"):
        Orchestrator(cfg)


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
def test_create_agent_unsupported_raises(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    cfg = AppConfig()
    cfg.agent.backend = "unsupported"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported agent"):
        Orchestrator(cfg)


##### RUN LOOP #####


@patch("e_heed.daemon.orchestrator.SentinelPipeline")
@patch("e_heed.daemon.orchestrator.TurnPipeline")
async def test_run_loop_dispatches_phases(mock_turn: MagicMock, mock_sentinel: MagicMock) -> None:
    orch = Orchestrator(AppConfig())
    orch._running = True

    call_count = 0

    async def _fake_sentinel(self_: object) -> None:
        nonlocal call_count
        call_count += 1
        orch._running = False

    with patch.object(Orchestrator, "_run_sentinel", _fake_sentinel):
        await orch._run_loop()

    assert call_count == 1
