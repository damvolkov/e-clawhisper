"""Pydantic models for daemon inter-component communication."""

from __future__ import annotations

from pydantic import BaseModel


class VADResult(BaseModel):
    """Result of a single VAD frame."""

    is_speech: bool
    probability: float


class TranscriptChunk(BaseModel):
    """A partial or final transcript from STT."""

    text: str
    is_final: bool = False


class AgentResponse(BaseModel):
    """A chunk of agent response."""

    text: str
    is_final: bool = False


class DaemonStatus(BaseModel):
    """Status reported via IPC."""

    running: bool
    pipeline_state: str
    conversation_active: bool
    agent_name: str
    agent_backend: str
