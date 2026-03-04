"""Voice session: always-listening wake-word activation with AgentOS bridge."""

from __future__ import annotations

import time
from typing import AsyncIterable

from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    FunctionTool,
    ModelSettings,
    RoomInputOptions,
    UserStateChangedEvent,
    llm,
)
from livekit.plugins import silero

from e_clawhisper.adapters.stt import WhisperLiveSTT
from e_clawhisper.adapters.tts import PiperTTS
from e_clawhisper.channels.base import BaseChannel
from e_clawhisper.channels.openfang import OpenFangChannel
from e_clawhisper.core.logger import LogIcon, logger
from e_clawhisper.core.settings import settings as st


##### AGENT #####


class ClawWhisperAgent(Agent):
    """Voice agent with wake-word gating and AgentOS channel bridge.

    Flow:
        IDLE  — listens via STT, scans transcript for wake word
        ACTIVE — forwards user speech to AgentOS, speaks response via TTS
        IDLE  — reverts after conversation_timeout seconds of silence
    """

    def __init__(
        self,
        *,
        channel: BaseChannel,
        wake_word: str,
        conversation_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            instructions=f"Voice assistant activated by '{wake_word}'.",
        )
        self._channel = channel
        self._wake_word = wake_word.lower()
        self._active = False
        self._last_activity: float = 0.0
        self._conversation_timeout = conversation_timeout

    @property
    def is_active(self) -> bool:
        return self._active

    def _extract_user_message(self, chat_ctx: llm.ChatContext) -> str | None:
        """Extract the latest user message text from chat context."""
        for item in reversed(chat_ctx.items):
            if getattr(item, "role", None) == "user":
                if text := getattr(item, "text_content", None):
                    return str(text).strip()
                content = getattr(item, "content", [])
                if isinstance(content, str):
                    return content.strip()
                for part in content:
                    if text := getattr(part, "text", None):
                        return str(text).strip()
        return None

    def _check_timeout(self) -> None:
        if self._active and self._last_activity:
            elapsed = time.time() - self._last_activity
            if elapsed > self._conversation_timeout:
                self._active = False
                logger.info(
                    "conversation_timeout elapsed=%.1fs",
                    elapsed,
                    icon=LogIcon.STOP,
                )

    def _strip_wake_word(self, text: str) -> str:
        lower = text.lower()
        idx = lower.find(self._wake_word)
        if idx == -1:
            return text
        before = text[:idx].rstrip(" ,!.")
        after = text[idx + len(self._wake_word) :].lstrip(" ,!.")
        return f"{before} {after}".strip()

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        """Override: route user speech through wake-word gate to AgentOS."""
        user_msg = self._extract_user_message(chat_ctx)
        if not user_msg:
            return

        self._check_timeout()

        if not self._active:
            if self._wake_word in user_msg.lower():
                self._active = True
                self._last_activity = time.time()
                logger.info(
                    "wake_word_detected word=%s",
                    self._wake_word,
                    icon=LogIcon.WAKE,
                )

                query = self._strip_wake_word(user_msg)
                if query:
                    response = await self._channel.send_message(query)
                    yield response
                else:
                    yield "I'm listening. How can I help you?"
            else:
                logger.debug(
                    "ignoring (no wake word): %s",
                    user_msg[:60],
                    icon=LogIcon.CHAT,
                )
            return

        self._last_activity = time.time()
        logger.debug("sending to channel: %s", user_msg[:100], icon=LogIcon.CHANNEL)

        response = await self._channel.send_message(user_msg)
        yield response


##### SESSION #####


def _create_channel() -> BaseChannel:
    """Factory for the configured agent backend channel."""
    match st.CHANNEL_BACKEND:
        case "openfang":
            return OpenFangChannel(
                base_url=st.openfang_base_url,
                agent_id=st.AGENT_ID,
                timeout=st.OPENFANG_TIMEOUT,
            )
        case backend:
            msg = f"Unsupported channel backend: {backend}"
            raise ValueError(msg)


class VoiceSession:
    """Orchestrates VAD + STT + wake-word + AgentOS channel + TTS."""

    def __init__(self) -> None:
        self._session: AgentSession | None = None

    def _setup_event_handlers(self, session: AgentSession) -> None:
        @session.on("agent_state_changed")
        def on_agent_state(ev: AgentStateChangedEvent) -> None:
            logger.debug(
                "agent_state: %s -> %s",
                ev.old_state,
                ev.new_state,
                icon=LogIcon.AGENT,
            )

        @session.on("user_state_changed")
        def on_user_state(ev: UserStateChangedEvent) -> None:
            logger.debug(
                "user_state: %s -> %s",
                ev.old_state,
                ev.new_state,
                icon=LogIcon.CHAT,
            )

        @session.on("conversation_item_added")
        def on_conversation_item(ev: ConversationItemAddedEvent) -> None:
            item = ev.item
            if getattr(item, "type", None) == "message":
                content = getattr(item, "text_content", "") or ""
                logger.debug(
                    "message [%s]: %s",
                    getattr(item, "role", "unknown"),
                    str(content)[:80],
                    icon=LogIcon.CHAT,
                )

    async def entrypoint(self, ctx: agents.JobContext) -> None:
        """LiveKit agent entrypoint — connects room, starts pipeline."""
        await ctx.connect()

        logger.info("session_starting room=%s", ctx.room.name, icon=LogIcon.START)

        channel = _create_channel()
        await channel.connect()

        agent = ClawWhisperAgent(
            channel=channel,
            wake_word=st.AGENT_NAME,
            conversation_timeout=st.CONVERSATION_TIMEOUT,
        )

        self._session = AgentSession(
            stt=WhisperLiveSTT(),
            tts=PiperTTS(),
            vad=silero.VAD.load(),
        )

        self._setup_event_handlers(self._session)

        await self._session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(),
        )

        logger.info(
            "session_started room=%s channel=%s wake_word=%s",
            ctx.room.name,
            st.CHANNEL_BACKEND,
            st.AGENT_NAME,
            icon=LogIcon.START,
        )
