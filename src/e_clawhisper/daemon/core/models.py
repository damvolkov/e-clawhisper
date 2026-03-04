"""Pydantic models for daemon inter-component communication."""

from __future__ import annotations

from pydantic import BaseModel

##### PIPELINE #####


class VADResult(BaseModel):
    """Result of processing one or more VAD frames."""

    is_speech: bool
    should_stop: bool
    probability: float


class TranscriptChunk(BaseModel):
    """A partial or final transcript from STT."""

    text: str
    is_final: bool = False


class AgentResponse(BaseModel):
    """A chunk of agent response."""

    text: str
    is_final: bool = False


##### IPC #####


class DaemonStatus(BaseModel):
    """Status reported via IPC."""

    running: bool
    pipeline_state: str
    conversation_active: bool
    agent_name: str
    agent_backend: str
