"""Pipeline manager — assembles and wires all pipeline components."""

from __future__ import annotations

from e_clawhisper.daemon.adapters.agents.openfang import OpenFangAdapter
from e_clawhisper.daemon.adapters.stt.whisper_live import WhisperLiveAdapter
from e_clawhisper.daemon.adapters.tts.piper import PiperAdapter
from e_clawhisper.daemon.core.interfaces.agent import AgentBase
from e_clawhisper.daemon.core.interfaces.stt import STTBase
from e_clawhisper.daemon.core.interfaces.tts import TTSBase
from e_clawhisper.daemon.core.processors.turn_manager import TurnManager
from e_clawhisper.daemon.core.processors.vad import TenVADProcessor
from e_clawhisper.daemon.core.processors.wake_word import WakeWordDetector
from e_clawhisper.shared.operational.audio_device import AudioDevice
from e_clawhisper.shared.settings import AgentBackend, AppConfig, STTBackend, TTSBackend


class PipelineManager:
    """Factory that assembles pipeline components from config."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def create_audio_device(self) -> AudioDevice:
        cfg = self._config.audio
        return AudioDevice(
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            chunk_size=cfg.chunk_size,
        )

    def create_vad(self) -> TenVADProcessor:
        return TenVADProcessor(
            config=self._config.vad,
            sample_rate=self._config.audio.sample_rate,
        )

    def create_wake_word(self) -> WakeWordDetector:
        return WakeWordDetector(wake_word=self._config.agent.name)

    def create_turn_manager(self) -> TurnManager:
        return TurnManager(conversation_timeout=30.0)

    def create_stt(self) -> STTBase:
        match self._config.stt.backend:
            case STTBackend.WHISPERLIVE:
                return WhisperLiveAdapter(config=self._config.stt.whisperlive)

    def create_tts(self) -> TTSBase:
        match self._config.tts.backend:
            case TTSBackend.PIPER:
                return PiperAdapter(config=self._config.tts.piper)

    def create_agent(self) -> AgentBase:
        match self._config.agent.backend:
            case AgentBackend.OPENFANG:
                return OpenFangAdapter(config=self._config.backends.openfang)
