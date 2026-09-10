"""
Agent

High-level API for managing Agora Conversational AI Agents.
"""
import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from agora_agent import Area, AsyncAgora
from agora_agent.agentkit import Agent as AgoraAgent
from agora_agent.agentkit.vendors import CustomLLM, DeepgramSTT, MiniMaxTTS, OpenAI

logger = logging.getLogger("uvicorn.error")

ADA_PROMPT = """You are Watcher, an AI incident commander joining a live incident
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

DEFAULT_GREETING = "Hi, this is Watcher. I'll listen in and keep track of what's discussed -- let me know if you'd like a recap."

# Appended to the conversation and handed to the LLM by parameters.
# silence_config below (action="think") when the room's gone quiet for a
# while. Deliberately an inert control string, never something a real
# participant would say -- iCall_api.chat_completions_endpoint (backend/
# app/iCall/iCall_api.py) recognizes it as the last message and swaps in
# its own deterministic, transcript-aware nudge (or stays silent if only
# one person is on the call) instead of treating it as real speech to
# extract facts from. Keep this in sync with SILENCE_TRIGGER_MARKER there.
SILENCE_TRIGGER_MARKER = "[[ithink-silence-check]]"


async def _fetch_keyterms(ithink_base: str, channel_name: str) -> Optional[str]:
    """
    Deepgram keyterm-prompting string for this call (see
    iCall_utils.build_keyterms) -- boosts recognition of words STT has no
    reason to get right on its own (the incident's service name, this
    agent's own name, incident-call jargon). Confirmed live this session:
    "auth-api" came back as "OT API" with no boosting at all.

    Best-effort, same reasoning as _fetch_late_joiner_catchup in server.py:
    a slow/unreachable backend should degrade to no boosting, never block
    the agent from starting.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{ithink_base}/icall/channel/{channel_name}/keyterms")
            response.raise_for_status()
            return response.json().get("data", {}).get("keyterm")
    except Exception:
        logger.warning("Failed to fetch keyterms for channel=%s", channel_name, exc_info=True)
        return None


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

        # Area picks the regional domain pool for Agora's own REST control
        # plane (agent join/start/stop) -- separate from which STT/TTS/LLM
        # vendors are reachable (confirmed against the installed SDK's
        # region.py: AP and US both resolve to the same "global" vendor
        # list, only CN is special-cased). This hackathon runs in India;
        # AP routes our join/start/stop calls to Agora's Asia-Pacific
        # domains instead of US ones. Does not touch the backend's own
        # Gemini round trip (that's a separate, unrelated hop) -- this is
        # specifically the "how fast does the agent join" latency, not
        # per-turn Thinking Engine latency.
        self.client = AsyncAgora(
            area=Area.AP,
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

        # Native MCP tool-calling -- Agora's platform calls these MCP
        # servers directly and forwards real OpenAI-style `tools`/
        # `tool_choice` to our custom LLM endpoint (confirmed live: only
        # works with advanced_features.enable_tools=True below, silently
        # ignored otherwise). GitHub's is the real, official remote MCP
        # server; iLogs' is our own thin MCP wrapper around this backend's
        # GET /ilogs/ (see backend/ilogs_mcp_service/) -- both need a
        # publicly reachable endpoint since Agora's cloud calls them, not
        # this local process.
        mcp_servers = []
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            mcp_servers.append({
                "name": "github",
                "endpoint": "https://api.githubcopilot.com/mcp/",
                "headers": {"Authorization": f"Bearer {github_token}"},
            })
        ilogs_mcp_url = os.getenv("ILOGS_MCP_URL")
        if ilogs_mcp_url:
            mcp_servers.append({"name": "ilogs", "endpoint": ilogs_mcp_url})

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
            mcp_servers=mcp_servers or None,
        )
        # en-IN is a real, separately-documented Deepgram nova-3 language
        # code (confirmed against Deepgram's own docs, not just "en" with
        # an accent guess) -- tunes the acoustic model for Indian-accented
        # English instead of defaulting toward US English. Per-call keyterm
        # fetch already boosts the agent's own name via build_keyterms.
        #
        # keyterm/smart_format/punctuation were reverted earlier this
        # session after incident-33/34/35 each produced real, non-silence
        # turns but zero usable transcript content -- correlated, never
        # actually root-caused. Reinstated alongside the en-IN locale
        # change on the team's decision to re-test properly rather than
        # assume the old correlation still holds with the locale now
        # correct -- but it just reproduced live again (incident-43,
        # 2026-09-06: 8 real, non-silence turns, latest_user_message=''
        # every single time, confirmed via direct log inspection, not a
        # guess). Reverting keyterm/smart_format/punctuation again, keeping
        # en-IN (never implicated in either occurrence -- both times the
        # empty-transcript symptom tracked keyterm/smart_format/
        # punctuation being on, not the locale). If this combination is
        # ever revisited, re-test it in isolation (one flag at a time)
        # rather than reinstating all three together again.
        stt = DeepgramSTT(
            model="nova-3",
            language="en-IN",
        )
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
            # appropriate moments" from the brief. action="think" appends
            # `content` to the conversation and routes it through the LLM
            # (our custom LLM proxy) instead of speaking it verbatim, so
            # what actually gets said can depend on the call so far --
            # whether only one person is present, and how much has already
            # been figured out -- rather than always repeating the same
            # canned line. See SILENCE_TRIGGER_MARKER above.
            "silence_config": {
                # 15s was firing mid-investigation, while someone was still
                # reading logs/dashboards rather than actually done talking
                # -- bumped to 30s so the check-in matches a room that's
                # genuinely gone quiet, not just mid-pause. Safe to lengthen
                # now that get_live_participant_count already suppresses
                # this entirely for a lone participant.
                "timeout_ms": 30000,
                "action": "think",
                "content": SILENCE_TRIGGER_MARKER,
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
                    # 0.5 is the SDK's own mid-range default. Flagged early
                    # this session as an open question (does a quieter
                    # speaker register as "speaking" at all) and never
                    # actually revisited until now. The SDK's own docs are
                    # explicit: lower values make it easier to detect
                    # speech, higher values ignore weak sounds. Lowered
                    # deliberately -- a missed quiet speaker (never
                    # transcribed at all) is a worse failure than a little
                    # extra background noise picked up, same asymmetry as
                    # the direct-address wake-word decision earlier this
                    # session.
                    "speech_threshold": 0.3,
                    "start_of_speech": {
                        "mode": "vad",
                        "vad_config": {
                            # Reported live: toggling a participant's own
                            # mic off/on while the agent is talking cuts
                            # the agent's voice off mid-sentence. Likely
                            # mechanism, not yet confirmed audibly (can't
                            # hear it from here): re-enabling a mic track
                            # is a known source of a brief hardware
                            # pop/noise burst as the capture pipeline
                            # restarts -- combined with speech_threshold
                            # already lowered to 0.3 (deliberately more
                            # sensitive to quiet sound, see above) and a
                            # 160ms interrupt window, that transient burst
                            # plausibly reads as "start of speech" and
                            # correctly-per-its-own-logic barges in on the
                            # agent, which sounds identical to a broken/cut
                            # voice. Bumped to 350ms -- long enough that a
                            # short click/pop shouldn't cross it, still
                            # short enough that a genuine interruption
                            # (which naturally sustains) barges in quickly.
                            # Needs live re-testing to confirm; this is a
                            # reasoned adjustment, not a verified fix.
                            "interrupt_duration_ms": 350,
                            "prefix_padding_ms": 300,
                        },
                    },
                    "end_of_speech": {
                        # Fixed 800ms VAD silence was still the root cause
                        # behind several distinct symptoms this session (the
                        # duplicate Bob/rollback-owner replies, the duplicate
                        # Watcher/status replies, and the "Outage is not in
                        # the US" / "in Africa" false-contradiction) -- one
                        # continuous sentence with a natural mid-thought
                        # pause gets cut into two independent backend turns
                        # whenever that pause happens to exceed a fixed
                        # silence window. Confirmed via the installed SDK's
                        # own type file (agora_agent's
                        # start_agents_request_properties_turn_detection_config_end_of_speech_mode.py)
                        # that end_of_speech.mode independently supports
                        # "semantic" for our cascaded/custom-LLM pipeline
                        # (this codebase is not mllm.enable=true, so the
                        # "semantic has no effect under mllm" caveat found
                        # during research doesn't apply here). Semantic mode
                        # judges whether an utterance is grammatically/
                        # semantically complete rather than only measuring
                        # silence, which is what actually distinguishes "the
                        # outage is not in the US" (complete thought, safe to
                        # end the turn) from a genuine mid-word pause.
                        "mode": "semantic",
                        "semantic_config": {
                            # Shorter than the old fixed 800ms: semantic mode
                            # judging completeness (not just silence length)
                            # is what makes a shorter base window safe here
                            # -- 300-600ms is the range research on
                            # combined semantic+acoustic VAD turn-taking
                            # points to for reducing false end-of-turn
                            # triggers, and 400ms sits in the middle of it.
                            "silence_duration_ms": 400,
                            # Bounded, not -1/unbounded -- caps the worst
                            # case (an ambiguous utterance the semantic layer
                            # can't confidently resolve) at 2s rather than
                            # risking an indefinite hang waiting for a
                            # determination that never firms up. After this
                            # timeout the SDK falls back to ending the turn
                            # based on current state, same failure mode as
                            # the old fixed-silence behavior, just as a
                            # bounded fallback instead of the default case.
                            "max_wait_ms": 2000,
                            # Catches "hold on" / "just a moment" as an
                            # explicit pause rather than turn-end -- directly
                            # relevant to an incident call, where someone
                            # mid-command frequently says exactly that while
                            # checking a terminal or dashboard.
                            "pause_state_enabled": True,
                        },
                        # Kept as a documented fallback reference, not
                        # deleted -- inert while mode is "semantic" (the SDK
                        # only reads vad_config when mode is "vad"), but
                        # switching mode back to "vad" is then a one-line
                        # revert if live testing shows semantic mode
                        # underperforms (e.g. the "English and Chinese only"
                        # language-support claim found during research,
                        # unverified against official Agora docs, turning out
                        # to be real against this pipeline's en-IN STT
                        # locale).
                        "vad_config": {
                            "silence_duration_ms": 800,
                        },
                    },
                },
            },
            # Real, live-observed bug in the previous config: `enable` was
            # never set, and `disabled_config` (per the SDK's own schema)
            # only applies when `enable` is false -- it does NOT govern
            # "what happens for non-keyword speech" under keywords mode,
            # which is what this was written to assume. With `enable`
            # unset, interruption plausibly never activated at all,
            # matching what was actually observed live: the agent kept
            # talking through cross-talk regardless of what was said.
            # start_of_speech is also just a better fit than keywords for
            # an incident call -- people interject by talking, not by
            # saying "stop"/"wait"/"ithink" first. keywords_config and
            # disabled_config are dropped since neither applies once
            # enable=true and mode=start_of_speech (the schema scopes both
            # to modes/states this config no longer uses).
            interruption={
                "enable": True,
                "mode": "start_of_speech",
            },
            # Removed filler_words entirely -- it's an Agora engine feature
            # that speaks a canned phrase from a static list on a fixed
            # timer, completely independent of chat_completions_endpoint's
            # gate: it fires whenever Gemini's real round trip exceeds
            # response_wait_ms, which for live structuring calls is often
            # every single turn. Confirmed live: the transcript showed
            # "Let me note that." / "Got it, noting that down." / "One sec."
            # firing constantly, indistinguishable to a listener from the
            # agent "speaking every time" -- the exact generic-acknowledgment
            # pattern the problem-statement audit (see the commit removing
            # spoken_reply's own "brief natural acknowledgment" guidance)
            # already decided to eliminate, just at the wrong layer: that
            # fix only touched the LLM's own reply content, not this
            # separate, content-blind engine feature. There's no
            # requirement in the brief for latency-masking chatter, and it
            # actively defeats should_speak_aloud's whole point (stay
            # silent unless there's a real reason) -- dead air while
            # Gemini thinks is fine, humans keep talking through it.
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
