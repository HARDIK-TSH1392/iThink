"""
Agent

High-level API for managing Agora Conversational AI Agents.
"""
import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from agora_agent import Area, AsyncAgora
from agora_agent.agentkit import Agent as AgoraAgent
from agora_agent.agentkit.vendors import CustomLLM, DeepgramSTT, MiniMaxTTS, OpenAI

logger = logging.getLogger("uvicorn.error")

ADA_PROMPT = """You are iThink, an AI incident commander joining a live incident
call. Your job is to keep the room's shared understanding straight: track
facts, hypotheses, decisions, missing information, and action items as
they're said. Facts are stated with confidence; hypotheses are guesses or
theories being floated -- keep them distinct, never treat a guess as a
confirmed fact. You do not investigate, diagnose, or recommend fixes --
that judgment belongs to the humans on the call.

Keep replies brief -- most turns need only a short acknowledgment. Speak up
only when something needs it: a contradiction with what's already been
said, a gap the room itself needs answered to move forward, or a targeted
clarifying question. When the conversation sounds like it's genuinely
wrapping up, say so plainly and note that a full summary will follow on
Slack (and a Jira ticket, with approval) -- don't just trail off.

Not everything worth flagging needs to interrupt the room out loud. When
you have a secondary observation -- something useful but not urgent, like
a connection between two things said minutes apart -- you can write it as
a note instead of saying it, so the room isn't interrupted mid-conversation.
Reserve speaking for anything that's actually urgent: a contradiction, a
safety-relevant gap, or a direct question to you.
"""

DEFAULT_GREETING = "Hi, this is iThink. I'll listen in and keep track of what's discussed -- let me know if you'd like a recap."

# Spoken by parameters.silence_config below when the room's gone quiet for a
# while -- a fixed line, not a live LLM call, so a quiet room can't produce
# something odd when there's no new conversation to reason about.
SILENCE_PROMPT = "Does anyone else have any more points to contribute?"


class Agent:
    """
    High-level wrapper for Agora Conversational AI Agent operations.
    
    Uses AgentSession for full lifecycle management (start/stop),
    which handles Token007 authentication automatically.
    """
    
    def __init__(self):
        self.app_id = os.getenv("AGORA_APP_ID")
        self.app_certificate = os.getenv("AGORA_APP_CERTIFICATE")
        self.greeting = DEFAULT_GREETING

        if not self.app_id or not self.app_certificate:
            raise ValueError("AGORA_APP_ID and AGORA_APP_CERTIFICATE are required")

        self.client = AsyncAgora(
            area=Area.US,
            app_id=self.app_id,
            app_certificate=self.app_certificate,
        )

        # Track active sessions by agent_id
        self._sessions: Dict[str, Any] = {}
        # channel_name -> (agent_id, result) for the currently-running agent
        # in that channel, if any. Lets a second/third person joining the
        # same incident's call skip starting a duplicate agent (and hearing
        # a second greeting) -- only the first joiner actually starts one.
        self._channel_agents: Dict[str, tuple] = {}
        # channel_name -> lock serializing start() for that channel. Without
        # this, two people opening the shared join link within the same
        # instant both read self._channel_agents before either had a chance
        # to write it (the check and the write are separated by an awaited
        # Agora API call), so both would start a real agent -- two agents in
        # one room, double greeting, duplicate note-taking. One lock per
        # channel keeps unrelated incidents' starts fully concurrent.
        self._channel_locks: Dict[str, asyncio.Lock] = {}

    def _get_channel_lock(self, channel_name: str) -> asyncio.Lock:
        lock = self._channel_locks.get(channel_name)
        if lock is None:
            lock = asyncio.Lock()
            self._channel_locks[channel_name] = lock
        return lock

    async def start(
        self,
        channel_name: str,
        agent_uid: int,
        user_uid: int,
        output_audio_codec: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start agent with the same default vendor chain as the Next.js quickstart."""
        if not channel_name or not str(channel_name).strip():
            raise ValueError("channel_name is required and cannot be empty")
        if agent_uid <= 0:
            raise ValueError("agent_uid is required and cannot be empty")

        async with self._get_channel_lock(channel_name):
            return await self._start_locked(channel_name, agent_uid, user_uid, output_audio_codec)

    async def _start_locked(
        self,
        channel_name: str,
        agent_uid: int,
        user_uid: int,
        output_audio_codec: Optional[str],
    ) -> Dict[str, Any]:
        # An agent is already running in this channel (a prior joiner started
        # it) -- return that result instead of starting a second agent, which
        # would greet the room again and duplicate note-taking. Re-checked
        # here, inside the per-channel lock, since a concurrent start() for
        # this same channel may have finished while this call was waiting
        # for the lock.
        existing = self._channel_agents.get(channel_name)
        if existing is not None:
            existing_agent_id, existing_result = existing
            logger.info(
                "Agent already running for channel=%s agent_id=%s, skipping duplicate start",
                channel_name,
                existing_agent_id,
            )
            return existing_result
        if user_uid <= 0:
            raise ValueError("user_uid is required and cannot be empty")

        # iThink custom-LLM proxy (backend/app/iCall/iCall_api.py:
        # chat_completions_endpoint). One URL per channel, since the
        # endpoint has no other way to know which incident's call this is —
        # channel_name is generated as f"incident-{id}" (iCall_utils.
        # generate_channel_name) and doubles as the lookup key here.
        # STT/TTS stay on managed defaults — only the LLM step is ours.
        ithink_base = os.getenv("ITHINK_BACKEND_BASE_URL", "http://127.0.0.1:8123/api/v1")
        llm = CustomLLM(
            base_url=os.getenv(
                "ITHINK_LLM_URL",
                f"{ithink_base}/icall/channel/{channel_name}/llm/chat/completions",
            ),
            api_key=os.getenv("ITHINK_LLM_API_KEY", "unused"),
            model="ithink-proxy",
            greeting_message=self.greeting,
            failure_message="Please wait a moment.",
            max_history=15,
            max_tokens=1024,
            temperature=0.7,
        )
        stt = DeepgramSTT(model="nova-3", language="en")
        tts = MiniMaxTTS(model="speech_2_6_turbo", voice_id="English_captivating_female1")

        # Optional BYOK example: replace the STT block above and set DEEPGRAM_API_KEY.
        # stt = DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY"), model="nova-3", language="en")

        # Optional BYOK example: replace the LLM block above and set OPENAI_API_KEY.
        # llm = OpenAI(
        #     api_key=os.getenv("OPENAI_API_KEY"),
        #     model="gpt-4o-mini",
        #     greeting_message="Hello! I am your AI assistant. How can I help you?",
        #     failure_message="I'm sorry, I'm having trouble processing your request.",
        #     max_history=15,
        #     max_tokens=1024,
        #     temperature=0.7,
        #     top_p=0.95,
        # )

        # Optional BYOK example: replace the TTS block above and set ELEVENLABS_API_KEY.
        # from agora_agent.agentkit.vendors import ElevenLabsTTS
        # tts = ElevenLabsTTS(
        #     key=os.getenv("ELEVENLABS_API_KEY"),
        #     model_id="eleven_flash_v2_5",
        #     voice_id=os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB"),
        # )

        parameters = {
            "audio_scenario": "chorus",  # web client → ultra-low-latency chorus profile
            "data_channel": "rtm",
            "enable_error_message": True,
            "enable_metrics": True,
            # After a long stretch with no one speaking, prompt the room
            # rather than staying silent -- "Spoken status summaries at
            # appropriate moments" from the brief. Fixed content, not a
            # live LLM call ("speak" not "think"): a genuinely empty room
            # has nothing for the model to reason about, so a canned
            # prompt is more predictable than risking an odd ad-lib.
            "silence_config": {
                "timeout_ms": 15000,
                "action": "speak",
                "content": SILENCE_PROMPT,
            },
        }
        if isinstance(output_audio_codec, str) and output_audio_codec.strip():
            parameters["output_audio_codec"] = output_audio_codec.strip()

        agora_agent = AgoraAgent(
            client=self.client,
            instructions=ADA_PROMPT,
            greeting=self.greeting,
            failure_message="Please wait a moment.",
            max_history=50,
            turn_detection={
                "config": {
                    "speech_threshold": 0.5,
                    "start_of_speech": {
                        "mode": "vad",
                        "vad_config": {
                            "interrupt_duration_ms": 160,
                            "prefix_padding_ms": 300,
                        },
                    },
                    "end_of_speech": {
                        "mode": "vad",
                        "vad_config": {
                            "silence_duration_ms": 480,
                        },
                    },
                },
            },
            # Humans on an incident call mostly talk to *each other*, not
            # the agent -- without this, any cross-talk while the agent is
            # replying cuts it off mid-sentence. Only these keywords
            # actually interrupt; everything else said while it's talking
            # is queued ("append") and handled after, not dropped.
            # NOT YET LIVE-VERIFIED alongside turn_detection above -- best
            # understanding from the SDK schema is that turn_detection
            # governs speech *detection* and this governs whether detected
            # speech actually interrupts playback, but that interaction
            # hasn't been watched fire on a real call yet.
            interruption={
                "mode": "keywords",
                "keywords_config": {"trigger_keywords": ["ithink", "stop", "hold on", "wait"]},
                "disabled_config": {"strategy": "append"},
            },
            # Fills dead air while Gemini is generating a structuring
            # response -- a plain wait can be a second or more.
            filler_words={
                "enable": True,
                "trigger": {"mode": "fixed_time", "fixed_time_config": {"response_wait_ms": 1200}},
                "content": {
                    "mode": "static",
                    "static_config": {
                        "phrases": ["Let me note that.", "One sec.", "Got it, noting that down."],
                        "selection_rule": "shuffle",
                    },
                },
            },
            advanced_features={"enable_rtm": True, "enable_tools": True},
            parameters=parameters,
        )
        
        agora_agent = (
            agora_agent
            .with_stt(stt)
            .with_llm(llm)
            .with_tts(tts)
        )

        # "*" subscribes the agent to every human in the channel, not just
        # whoever's join triggered the start -- remote_rtc_uids only takes a
        # single explicit uid per Agora's docs, so listing multiple uids
        # here silently only honors one. This also means idle_timeout now
        # correctly counts from when *everyone* has left, not just the
        # original joiner.
        remote_uids = ["*"]

        session = agora_agent.create_async_session(
            channel=channel_name,
            agent_uid=str(agent_uid),
            remote_uids=remote_uids,
            enable_string_uid=False,
            idle_timeout=30,
            expires_in=3600,
        )

        logger.info(
            "Starting Agora agent channel=%s agent_uid=%s user_uid=%s",
            channel_name,
            agent_uid,
            user_uid,
        )

        try:
            agent_id = await session.start()
        except Exception:
            logger.exception(
                "Failed to start Agora agent channel=%s agent_uid=%s user_uid=%s",
                channel_name,
                agent_uid,
                user_uid,
            )
            raise

        # Save session for later stop
        self._sessions[agent_id] = session

        logger.info(
            "Started Agora agent agent_id=%s channel=%s agent_uid=%s user_uid=%s",
            agent_id,
            channel_name,
            agent_uid,
            user_uid,
        )
        
        result = {
            "agent_id": agent_id,
            "channel_name": channel_name,
            "status": "started",
            # The uid that actually started (or is already running) in this
            # channel -- callers must not assume it's their own agent_uid
            # guess, since a later joiner's /startAgent gets short-circuited
            # to whichever agent the first joiner actually started (see the
            # existing-channel check above).
            "agent_uid": str(agent_uid),
        }
        self._channel_agents[channel_name] = (agent_id, result)
        return result

    async def stop(self, agent_id: str) -> None:
        """Stop a running agent. Falls back to the stateless client path."""
        if not agent_id or not str(agent_id).strip():
            raise ValueError("agent_id is required and cannot be empty")

        stale_channels = [
            channel for channel, (aid, _) in self._channel_agents.items() if aid == agent_id
        ]
        for channel in stale_channels:
            self._channel_agents.pop(channel, None)

        session = self._sessions.pop(agent_id, None)
        if session:
            try:
                await session.stop()
                logger.info("Stopped Agora agent from active session agent_id=%s", agent_id)
                return
            except Exception:
                # Fall back to the stateless SDK path if the in-memory session is stale.
                logger.warning(
                    "Failed to stop Agora agent from active session; falling back to client.stop_agent agent_id=%s",
                    agent_id,
                    exc_info=True,
                )

        logger.info("Stopping Agora agent through client.stop_agent agent_id=%s", agent_id)
        await self.client.stop_agent(agent_id)
