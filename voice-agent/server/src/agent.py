"""
Agent

High-level API for managing Agora Conversational AI Agents.
"""
import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, Optional

import httpx

from agora_agent import Area, AsyncAgora
from agora_agent.agentkit import Agent as AgoraAgent
from agora_agent.agentkit.token import generate_convo_ai_token
from agora_agent.agentkit.vendors import (
    AnamAvatar, CustomLLM, DeepgramSTT, Gemini, MiniMaxTTS, OpenAI, SarvamSTT, SarvamTTS,
)

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

# Tier 1 (English + Hindi): one continuous Deepgram nova-3 session,
# language="multi" -- Deepgram's own real-time code-switching mode,
# confirmed to cover this exact language pair. No entry in TIER2_LANGUAGES
# means "multi" (or any language_code we don't recognize -- fail toward
# the proven path, not a silent Sarvam attempt on an unsupported code).
#
# Tier 2 (everything else): Sarvam, model="saaras:v3" -- the ONLY model
# string confirmed to produce real transcripts through Agora's Sarvam
# integration (live-tested twice: "saaras:v3-realtime" breaks
# transcription entirely, empty transcripts despite the pipeline staying
# alive). No keyterm-equivalent boosting exists on this path (confirmed
# 3x live: unboosted 1/4, bare-word prompt 0/4, natural-sentence prompt
# 0/4 on wake-word recognition) and no partial/interim transcripts exist
# either (confirmed via pilot-bot-v1's own production code hitting the
# identical wall on direct API access) -- accepted, real limitations of
# this tier, not something more parameter-tuning fixes.
#
# turn_detection_language falls back to "en-IN" for four of these nine
# languages -- confirmed absent from the installed SDK's own
# TURN_DETECTION_LANGUAGE_VALUES whitelist (mr-IN, pa-IN, ml-IN, or-IN are
# not in the 32-entry list; ta-IN/te-IN/kn-IN/bn-IN/gu-IN/hi-IN are).
# Passing an unsupported code raises ValueError at agent-start time, so
# this fallback isn't optional -- it's a real, accepted mismatch (Agora's
# own semantic end-of-speech layer judges completeness in en-IN for these
# four languages' speech), not a bug worth chasing.
TIER2_LANGUAGES: Dict[str, Dict[str, str]] = {
    "ta-IN": {"sarvam_speaker": "priya", "turn_detection_language": "ta-IN"},
    "te-IN": {"sarvam_speaker": "priya", "turn_detection_language": "te-IN"},
    "kn-IN": {"sarvam_speaker": "priya", "turn_detection_language": "kn-IN"},
    "bn-IN": {"sarvam_speaker": "priya", "turn_detection_language": "bn-IN"},
    "gu-IN": {"sarvam_speaker": "priya", "turn_detection_language": "gu-IN"},
    "mr-IN": {"sarvam_speaker": "priya", "turn_detection_language": "en-IN"},
    "pa-IN": {"sarvam_speaker": "priya", "turn_detection_language": "en-IN"},
    "ml-IN": {"sarvam_speaker": "priya", "turn_detection_language": "en-IN"},
    "or-IN": {"sarvam_speaker": "priya", "turn_detection_language": "en-IN"},
}

LANGUAGE_DISPLAY_NAMES: Dict[str, str] = {
    "multi": "English/Hindi",
    "ta-IN": "Tamil", "te-IN": "Telugu", "kn-IN": "Kannada", "bn-IN": "Bengali",
    "mr-IN": "Marathi", "gu-IN": "Gujarati", "pa-IN": "Punjabi",
    "ml-IN": "Malayalam", "or-IN": "Odia",
}


async def _fetch_voice_config(ithink_base: str, channel_name: str) -> Dict[str, Any]:
    """
    Voice-pipeline config for this call: Deepgram keyterm-prompting string
    (see iCall_utils.build_keyterms) plus language_code, the durable
    Tier-1/Tier-2 selection read at agent-start time. Replaces
    _fetch_keyterms -- same endpoint, grown one field, still one round
    trip per agent start.

    Best-effort, same reasoning as _fetch_late_joiner_catchup in server.py:
    a slow/unreachable backend should degrade to Tier-1 defaults, never
    block the agent from starting.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{ithink_base}/icall/channel/{channel_name}/keyterms")
            response.raise_for_status()
            data = response.json().get("data", {})
            return {"keyterm": data.get("keyterm"), "language_code": data.get("language_code") or "multi"}
    except Exception:
        logger.warning("Failed to fetch voice config for channel=%s", channel_name, exc_info=True)
        return {"keyterm": None, "language_code": "multi"}


async def _fetch_delegate_info(ithink_base: str, channel_name: str) -> Optional[Dict[str, Any]]:
    """
    Delegate-mode notes for this call (see iCall_api's GET .../delegate) --
    set when the resolved approver approved but couldn't personally join
    (iOrchestrate's approve_delegate/modal flow). Same best-effort
    degrade-safe shape as _fetch_keyterms: a slow/unreachable backend, or
    simply no delegate notes for this call (the common case), both just
    mean the normal greeting/no-avatar path runs.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{ithink_base}/icall/channel/{channel_name}/delegate")
            response.raise_for_status()
            data = response.json().get("data", {})
            return data if data.get("delegate_notes") else None
    except Exception:
        logger.warning("Failed to fetch delegate info for channel=%s", channel_name, exc_info=True)
        return None


class Agent:
    """
    High-level wrapper for Agora Conversational AI Agent operations.

    Uses AgentSession for full lifecycle management (start/stop),
    which handles Token007 authentication automatically.
    """

    # How long the delegate avatar agent waits before joining/greeting,
    # so it speaks after Watcher's own short default greeting instead of
    # over it (see _start_delegate_avatar_agent) -- a rough heuristic, not
    # a measured value, since Agora has no native cross-agent turn signal.
    # DEFAULT_GREETING is ~21 words -- at a natural TTS pace that's ~8-9s
    # of actual speech, plus 1-2s of connection/TTS-startup latency before
    # the first audio even begins. 6s was confirmed live (2026-09-12) to
    # be too short -- the avatar started talking over Watcher's own
    # greeting. Bumped with real margin rather than a small nudge, since
    # a slightly-late avatar reads fine in a demo but talking over Watcher
    # reads as broken.
    DELEGATE_AVATAR_START_DELAY_SECONDS = 11

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
        # channel_name -> (agent_id, result, agent_uid, user_uid) for the
        # currently-running agent in that channel, if any. Lets a
        # second/third person joining the same incident's call skip
        # starting a duplicate agent (and hearing a second greeting) --
        # only the first joiner actually starts one. agent_uid/user_uid
        # are kept here (not just agent_id/result) so switch_language can
        # restart with the same identity across a language handoff.
        self._channel_agents: Dict[str, tuple] = {}
        # channel_name -> lock serializing start() for that channel. Without
        # this, two people opening the shared join link within the same
        # instant both read self._channel_agents before either had a chance
        # to write it (the check and the write are separated by an awaited
        # Agora API call), so both would start a real agent -- two agents in
        # one room, double greeting, duplicate note-taking. One lock per
        # channel keeps unrelated incidents' starts fully concurrent.
        self._channel_locks: Dict[str, asyncio.Lock] = {}
        # channel_name -> the delegate-avatar agent's own agent_id, when
        # one is running for that channel (see _start_delegate_avatar_agent).
        # Tracked separately from _channel_agents (Watcher's own) since the
        # two are independent agents with independent lifecycles that
        # nonetheless need to be torn down together (see stop()) -- the
        # web client only ever learns Watcher's agent_id, never this one.
        self._channel_delegate_agents: Dict[str, str] = {}

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
        language_code: Optional[str] = None,
        greeting_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start agent with the same default vendor chain as the Next.js quickstart."""
        if not channel_name or not str(channel_name).strip():
            raise ValueError("channel_name is required and cannot be empty")
        if agent_uid <= 0:
            raise ValueError("agent_uid is required and cannot be empty")

        async with self._get_channel_lock(channel_name):
            return await self._start_locked(
                channel_name, agent_uid, user_uid, output_audio_codec, language_code, greeting_override,
            )

    async def _start_locked(
        self,
        channel_name: str,
        agent_uid: int,
        user_uid: int,
        output_audio_codec: Optional[str],
        language_code: Optional[str] = None,
        greeting_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        # An agent is already running in this channel (a prior joiner started
        # it) -- return that result instead of starting a second agent, which
        # would greet the room again and duplicate note-taking. Re-checked
        # here, inside the per-channel lock, since a concurrent start() for
        # this same channel may have finished while this call was waiting
        # for the lock. Not applicable to a switch_language restart -- that
        # path calls stop() first, which evicts this entry before start()
        # runs again for the same channel.
        existing = self._channel_agents.get(channel_name)
        if existing is not None:
            existing_agent_id, existing_result = existing[0], existing[1]
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

        # Delegate mode: the resolved approver approved but couldn't join,
        # and left notes on what's been done / what to cover (see
        # iOrchestrate_api's approve_delegate/modal flow). Represented by a
        # SEPARATE second agent (see _start_delegate_avatar_agent below) --
        # its own voice, its own face, actually standing in for the absent
        # lead in the room -- not a change to Watcher's own persona.
        # Watcher's greeting only gets Watcher-authored (not carrying the
        # delegate's update itself) when that second agent can't actually
        # start. That requires BOTH an avatar vendor AND GEMINI_API_KEY
        # (the second agent's own native LLM, checked again inside
        # _start_delegate_avatar_agent) -- avatar-only would silently drop
        # the delegate's notes on the floor: Watcher assumes someone else
        # is carrying the message, the second agent bails on missing
        # GEMINI_API_KEY, nobody says it.
        delegate_info = await _fetch_delegate_info(ithink_base, channel_name)
        # Anam is a first-class Agora vendor (unlike GenericAvatar), so
        # there's no api_base_url to configure -- Agora's own backend
        # already knows how to reach it.
        anam_api_key = os.getenv("ANAM_API_KEY")
        anam_avatar_id = os.getenv("ANAM_AVATAR_ID")
        avatar_configured = bool(anam_api_key and anam_avatar_id)
        delegate_agent_configured = avatar_configured and bool(os.getenv("GEMINI_API_KEY"))

        greeting = self.greeting
        if delegate_info and not delegate_agent_configured:
            approver_name = delegate_info.get("approver_name") or "the resolved approver"
            greeting = (
                f"Hi, this is Watcher. {approver_name} couldn't join today, so I'm standing in "
                f"for them. Here's their update: {delegate_info['delegate_notes']} "
                f"Now, let's go around -- who's working on what?"
            )

        # Native MCP tool-calling -- Agora's platform calls these MCP
        # servers directly and forwards real OpenAI-style `tools`/
        # `tool_choice` to our custom LLM endpoint (confirmed live: only
        # works with advanced_features.enable_tools=True below, silently
        # ignored otherwise). GitHub's is the real, official remote MCP
        # server; iLogs' is our own thin MCP wrapper around this backend's
        # GET /ilogs/ (see backend/ilogs_mcp_service/) -- both need a
        # publicly reachable endpoint since Agora's cloud calls them, not
        # this local process.
        # allowed_tools/timeout_ms confirmed against Agora's actual REST API
        # schema (Docs-Source's join.mdx Parameter definitions) -- entries
        # are otherwise Dict[str, Any] in the SDK, completely unvalidated,
        # so a typo'd key here would silently do nothing rather than error.
        # Without allowed_tools, GitHub's real remote MCP server exposes its
        # FULL tool list to the live LLM -- confirmed via tools/list --
        # including create_pull_request, merge_pull_request, delete_file,
        # push_files, create_repository, etc. Watcher only ever needs to
        # answer questions during a live incident call, never take GitHub
        # actions, so scoped to read-only investigation tools only.
        mcp_servers = []
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            mcp_servers.append({
                "name": "github",
                "endpoint": "https://api.githubcopilot.com/mcp/",
                "headers": {"Authorization": f"Bearer {github_token}"},
                "allowed_tools": [
                    "list_commits", "get_commit", "list_pull_requests",
                    "pull_request_read", "list_issues", "issue_read",
                    "get_file_contents", "search_commits", "search_code",
                ],
                "timeout_ms": 8000,
            })
        ilogs_mcp_url = os.getenv("ILOGS_MCP_URL")
        if ilogs_mcp_url:
            ilogs_server: Dict[str, Any] = {
                "name": "ilogs",
                "endpoint": ilogs_mcp_url,
                "allowed_tools": [
                    "get_recent_logs", "get_incident_status", "list_action_items",
                    "search_similar_incidents",
                ],
                "timeout_ms": 8000,
            }
            # Closes the "anyone with the tunnel URL can read our incident
            # logs" gap -- the tool itself rejects a missing/wrong secret
            # (see ilogs_mcp_service/server.py). Optional: matches that
            # service's own "unset means no check" dev-local default.
            ilogs_shared_secret = os.getenv("ILOGS_SHARED_SECRET")
            if ilogs_shared_secret:
                ilogs_server["headers"] = {"X-Ilogs-Shared-Secret": ilogs_shared_secret}
            mcp_servers.append(ilogs_server)

        # filler_words was removed entirely (see the interruption block below)
        # because its only mode at the time -- a static phrase list, fired
        # unconditionally on every LLM round-trip over response_wait_ms --
        # fought should_speak_aloud's own gate. The SDK also supports a
        # "generated" content mode: a *separate*, parallel LLM call (fed only
        # the last user message, explicitly instructed not to answer it) that
        # still falls back to the static list on failure/timeout/empty --
        # mechanically different from what was tested and removed, so this
        # re-enables it only in generated mode, and only when a real key for
        # that separate call is actually configured (GEMINI_API_KEY unset ->
        # filler_words stays off entirely, same as today -- never falls back
        # to the already-proven-broken static-only behavior).
        # response_wait_ms=2500 reuses the exact threshold this codebase
        # already validated (see commit history: 1200ms fired on nearly every
        # turn, 2500ms only fires when a turn is genuinely slow).
        filler_words = None
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            filler_words = {
                "enable": True,
                "trigger": {
                    "mode": "fixed_time",
                    "fixed_time_config": {"response_wait_ms": 2500},
                },
                "content": {
                    "mode": "generated",
                    # Required even in generated mode -- the SDK's own
                    # fallback tier when the generated call isn't ready,
                    # fails, or returns empty text.
                    "static_config": {
                        "phrases": ["One moment.", "Still with you.", "Just a second."],
                        "selection_rule": "round_robin",
                    },
                    "generated_config": {
                        "llm_provider": {
                            # Google's OpenAI-compatible endpoint for Gemini
                            # (ai.google.dev/gemini-api/docs/openai) -- a
                            # genuinely separate call from the main iThink
                            # custom-LLM proxy, per the SDK's own docstring
                            # ("runs in parallel with the main business LLM").
                            #
                            # Both base_url and url set: confirmed live, this
                            # installed SDK's own type hints (Fern-generated,
                            # pydantic model) say the field is base_url, but
                            # Agora's actual REST API rejects the request with
                            # InvalidFieldValue demanding
                            # properties.filler_words.content.generated_config
                            # .llm_provider.url specifically -- the SDK's
                            # client-side types are stale relative to the live
                            # API here. Sending both costs nothing (extra
                            # fields are allowed) and survives either being
                            # the one actually read server-side.
                            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                            "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                            "api_key": gemini_api_key,
                            "params": {"model": "gemini-flash-lite-latest"},
                        },
                        "prompt": (
                            "Generate a brief, conversational filler phrase "
                            "acknowledging you heard the user while you finish "
                            "thinking. Do not answer their question or "
                            "restate what they said."
                        ),
                    },
                },
            }

        llm = CustomLLM(
            base_url=os.getenv(
                "ITHINK_LLM_URL",
                f"{ithink_base}/icall/channel/{channel_name}/llm/chat/completions",
            ),
            api_key=os.getenv("ITHINK_LLM_API_KEY", "unused"),
            model="ithink-proxy",
            # greeting_override is set by switch_language after a handoff
            # ("I'm back -- now listening in Tamil.") -- reuses the
            # existing, already-proven greeting-on-join code path rather
            # than introducing a separate agent_think()/speak() call for
            # this specific notice. Falls back to the (possibly
            # delegate-enriched, see `greeting` above) normal greeting
            # rather than self.greeting directly, so a delegate's notes
            # still get spoken on a language-handoff restart too.
            greeting_message=greeting_override or greeting,
            failure_message="Please wait a moment.",
            max_history=15,
            max_tokens=1024,
            temperature=0.7,
            mcp_servers=mcp_servers or None,
        )
        # Two-tier multilingual STT/TTS, keyed off the call's durable
        # language_code (backend/app/iCall/iCall_model.py). Full history
        # of how keyterm's actual working format was found (a %20-encoded
        # string, not a literal space, matching Agora's own documented
        # example byte-for-byte -- literal spaces and JSON-array/
        # additional_params forms all reproduced a real, live,
        # zero-content-transcript failure across incidents 33/34/35/43/
        # 50/51) lives in git history on fix/watcher-wake-word-mishearings
        # and the keyterm-format investigation branch -- not repeated here
        # now that the branching logic itself needs the space.
        voice_config = await _fetch_voice_config(ithink_base, channel_name)
        resolved_language = language_code or voice_config["language_code"]
        tier2_config = TIER2_LANGUAGES.get(resolved_language)

        if tier2_config is None:
            # Tier 1: English + Hindi, one continuous Deepgram session via
            # language="multi" -- Deepgram's own real-time code-switching
            # mode, covering both languages with zero vendor-switch logic.
            # keyterm boosting is proven live (3/3 wake-word accuracy) only
            # in this exact encoding -- %20, not a literal space or comma,
            # matching Agora's own documented example for this field.
            keyterm = voice_config["keyterm"]
            stt = DeepgramSTT(
                model="nova-3",
                language="multi",
                keyterm=keyterm.replace(" ", "%20") if keyterm else None,
            )
            # language_boost="hi" removed: confirmed live (real TTS-layer
            # error, code 2013, "invalid params: language_boost") that
            # MiniMax rejects a plain ISO code here -- whatever format it
            # actually wants (a full language name, an "auto" value, or
            # something else) is unconfirmed. Second corrected assumption
            # in a row from the Doc 2 feature sweep (after sal_mode) --
            # not re-attempted without checking MiniMax's own real
            # accepted-value list first.
            tts = MiniMaxTTS(
                model="speech_2_6_turbo",
                voice_id="English_captivating_female1",
            )
            turn_detection_language = "en-IN"
        else:
            # Tier 2: Sarvam, model="saaras:v3" is the ONLY value confirmed
            # to produce real transcripts through Agora's Sarvam
            # integration -- "saaras:v3-realtime" breaks transcription
            # entirely (confirmed twice live), likely because Agora's
            # integration connects to Sarvam's legacy (non-"-realtime")
            # endpoint regardless of the model string given. No keyterm
            # equivalent exists on this path at all (confirmed 3x live) --
            # accepted, not something more parameter-tuning fixes.
            sarvam_key = os.getenv("SARVAM_API_KEY")
            stt = SarvamSTT(api_key=sarvam_key, language=resolved_language, model="saaras:v3")
            tts = SarvamTTS(
                key=sarvam_key,
                target_language_code=resolved_language,
                speaker=tier2_config["sarvam_speaker"],
            )
            turn_detection_language = tier2_config["turn_detection_language"]

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
            # "chorus" ("real-time chorus scenario... requires ultra-low
            # latency" per Agora's own SDK docstring) was chosen for raw
            # speed, but Agora's audio best-practices doc names this exact
            # scenario as the documented cause of needing to speak loudly
            # for the agent to pick up speech, with "aiserver" -- "optimized
            # for interactions between the user and the conversational AI
            # agent in terms of latency and network resilience" -- as the
            # fix. Confirmed live (incident-46, 2026-09-09): had to speak
            # unusually loudly for turns to register at all. Not yet
            # re-tested live after this change; if aiserver doesn't resolve
            # it, the next lever is speech_threshold below, not this one.
            "audio_scenario": "aiserver",
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
            # Without this, stop() can cut the agent off mid-sentence --
            # nothing today guarantees it finishes speaking before leaving
            # the channel. graceful_enabled makes stop() wait for the agent
            # to reach IDLE (done speaking) before actually exiting, capped
            # at graceful_timeout_seconds so a stuck/looping agent can't hang
            # a real stop() call indefinitely. Also what keeps a language-
            # switch handoff's stop() from cutting the agent off mid-word
            # right as the switch fires, not just ordinary /stopAgent calls.
            "farewell_config": {
                "graceful_enabled": True,
                "graceful_timeout_seconds": 8,
            },
        }
        if isinstance(output_audio_codec, str) and output_audio_codec.strip():
            parameters["output_audio_codec"] = output_audio_codec.strip()

        agora_agent = AgoraAgent(
            client=self.client,
            instructions=ADA_PROMPT,
            greeting=greeting,
            failure_message="Please wait a moment.",
            # max_history lives on CustomLLM below (max_history=15), not here --
            # Agent.__init__'s own max_history/instructions/greeting/failure_message
            # are documented-deprecated in favor of configuring the LLM/MLLM vendor
            # directly. A stray max_history=50 here was dead config: it never
            # governed anything, and its different value (50 vs 15) made it look
            # like an intentional, larger history window that didn't actually exist.
            turn_detection={
                # Separate from the STT's own `language` above -- this is
                # what Agora's own turn-detection/semantic-completeness
                # layer uses to judge whether a sentence is actually done
                # (see end_of_speech.mode="semantic" below), and it was
                # never set here, silently defaulting to "en-US" (see
                # agora_agent.agentkit.agent.DEFAULT_TURN_DETECTION_LANGUAGE
                # in the installed SDK) while every other locale-aware
                # setting in this file is en-IN. "en-IN" is a real,
                # validated value for this field too (agentkit/agent.py's
                # own TURN_DETECTION_LANGUAGE_VALUES whitelist). Computed
                # above per-tier: en-IN for Tier 1 and for the four Tier-2
                # languages absent from that whitelist (mr-IN, pa-IN,
                # ml-IN, or-IN), the language's own code for the other
                # five Tier-2 languages that are present in it.
                "language": turn_detection_language,
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
                            #
                            # The SDK's vad_config actually has two separate
                            # interrupt thresholds: interrupt_duration_ms
                            # (barge-in while the agent is NOT talking) and
                            # speaking_interrupt_duration_ms (barge-in while
                            # the agent IS talking -- the exact mic-pop
                            # scenario the comment above describes). Only the
                            # first was ever set, so the mic-pop case was
                            # actually governed by whatever the SDK's own
                            # default is for the second field, not 350ms.
                            # Set explicitly to the same value for now --
                            # same reasoning applies to both, and there's no
                            # live evidence yet that they should differ.
                            "interrupt_duration_ms": 350,
                            # Separate from interrupt_duration_ms above --
                            # this one specifically gates how long a voice
                            # has to sustain to interrupt the agent while
                            # it is ALREADY TALKING, distinct from ordinary
                            # turn-taking when it's silent (confirmed in
                            # the installed SDK's own vad_config type,
                            # matches Agora's own documented example
                            # pairing 160ms/320ms for the same two fields).
                            # Real feedback from a mentor session (Hardik,
                            # 2026-09-10): backchannels ("umm," "okay,"
                            # "right") were being read as real
                            # interruptions, cutting the agent off mid-
                            # sentence for what was just the room listening,
                            # not taking the floor. Backchannels are
                            # typically well under 500ms; a genuine
                            # interruption is sustained speech. Higher than
                            # interrupt_duration_ms deliberately -- ordinary
                            # turn-taking (agent silent) should stay
                            # responsive; only the mid-speech case gets the
                            # extra tolerance. Keyword-only interruption
                            # (interruption.mode="keywords") was considered
                            # and rejected for this: already tried once on
                            # this project and reverted after real
                            # interjections that didn't use a listed
                            # trigger word got talked over -- worse than
                            # the problem being fixed. Not live-tested yet;
                            # this is a reasoned adjustment same as
                            # interrupt_duration_ms above, not a verified
                            # fix.
                            "speaking_interrupt_duration_ms": 650,
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
            # Selective Attention Locking left OFF (see advanced_features
            # below): "recognition" mode requires a pre-registered
            # voiceprint sample per speaker (sal.sample_urls, a 10-15s
            # 16kHz mono PCM file) -- confirmed live, Agora rejects the
            # whole start_agent call with InvalidModuleParameter when
            # sal_mode is "recognition" and sample_urls is empty, which it
            # always is here since there's no per-responder voiceprint
            # enrollment step in this system. "locking" mode doesn't need
            # sample_urls but isn't a real alternative either -- it latches
            # onto ONE speaker and blocks ~95% of other human voices, which
            # would suppress the other legitimate responders on exactly the
            # multi-responder incident bridge this was meant to help.
            filler_words=filler_words,
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
            # SAL disabled: confirmed live (real 400 from Agora, not a
            # silent failure) that sal_mode="recognition" REQUIRES
            # sample_urls (pre-registered voice samples per speaker) to be
            # non-empty -- contradicting the doc's framing that recognition
            # mode "doesn't require picking one speaker." We have no way to
            # pre-register incident-call participants' voices before they
            # join a live call, so this mode isn't usable here without data
            # we can't realistically supply. Corrected assumption, not
            # re-attempted without a real sample-collection flow.
            advanced_features={"enable_rtm": True, "enable_tools": True},
            # Opt-in, not default: disables Agora's automatic cross-region
            # failover in exchange for a hard India-region guarantee.
            # Real trade-off given this hackathon's explicit India focus
            # (lower baseline latency, no silent failover elsewhere) --
            # not enabled unconditionally since the failover behavior
            # itself has never caused a problem worth giving up.
            geofence={"area": "INDIA"} if os.getenv("WATCHER_GEOFENCE_INDIA") else None,
            parameters=parameters,
        )
        
        agora_agent = (
            agora_agent
            .with_stt(stt)
            .with_llm(llm)
            .with_tts(tts)
        )

        if tier2_config is not None:
            # SarvamTTSOptions has no `model`/`additional_params` field at
            # all (unlike every sibling vendor wrapper), so Agora's backend
            # falls back to its own hardcoded default -- confirmed live to
            # be the now-deprecated "bulbul:v2" (real HTTP 400 from
            # Sarvam's own API: "Model 'bulbul:v2' has been deprecated.
            # Please use 'bulbul:v3' instead."). The RAW type underneath
            # the wrapper (SarvamTtsParams) uses extra="allow", and
            # Agent._tts is stored as a plain mutable dict after
            # with_tts() -- same workaround Doc 2 documented for
            # CustomLLM's missing `tools` kwarg. Untested live yet; if
            # Agora's backend doesn't read this key at all, this line is a
            # no-op, not a regression -- worst case, same broken bulbul:v2
            # behavior as before this change.
            agora_agent._tts["params"]["model"] = "bulbul:v3"

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
            "language_code": resolved_language,
        }
        # Tracks agent_uid/user_uid alongside (agent_id, result) too, not
        # just the pair -- switch_language needs both to restart the agent
        # under the SAME identity after a language handoff (see its own
        # unpacking below).
        self._channel_agents[channel_name] = (agent_id, result, agent_uid, user_uid)

        # Delegate avatar: a SECOND, independent agent -- its own voice,
        # its own face -- joining alongside Watcher, not a change to
        # Watcher itself. Launched as a background task, NOT awaited here:
        # this method's caller is the actual /startAgent HTTP request, and
        # the delegate avatar's own start includes a deliberate delay (see
        # _start_delegate_avatar_agent) so it speaks after Watcher's own
        # greeting instead of talking over it -- awaiting that inline would
        # hold up the whole call from starting for that same delay, which
        # is worse than the caller getting a response the moment Watcher
        # itself is confirmed up. Never allowed to fail the whole call
        # either way: if this raises, Watcher has already joined and the
        # call proceeds without a visual stand-in, same degrade-safe
        # discipline as every other optional piece in this file.
        if delegate_info and delegate_agent_configured and channel_name not in self._channel_delegate_agents:
            asyncio.create_task(
                self._start_delegate_avatar_agent_safely(channel_name, delegate_info, user_uid)
            )

        return result

    async def _start_delegate_avatar_agent_safely(
        self, channel_name: str, delegate_info: Dict[str, Any], user_uid: int
    ) -> None:
        """Fire-and-forget wrapper: create_task swallows exceptions silently
        unless something awaits the task or checks its result, so this is
        the one place that actually logs a failure here."""
        try:
            await self._start_delegate_avatar_agent(channel_name, delegate_info, user_uid)
        except Exception:
            logger.exception(
                "Failed to start delegate avatar agent for channel=%s -- "
                "Watcher itself is still up and the call proceeds without it",
                channel_name,
            )

    async def _start_delegate_avatar_agent(
        self, channel_name: str, delegate_info: Dict[str, Any], user_uid: int
    ) -> None:
        """
        A SECOND, independent Agora agent representing the delegating
        approver -- its own voice, its own avatar, its own conversational
        loop -- joining the room alongside Watcher, not a change to
        Watcher's own persona or session. Watcher keeps doing its own job
        (structuring/extraction/redaction/speak-gate) for the whole room,
        including whatever this agent says; this agent's only job is to
        open with the delegate's update and actively ask the room for
        status, in their own conversational back-and-forth -- deliberately
        NOT given iCall's structuring pipeline, since duplicating that
        here would mean two agents both trying to extract/gate the same
        conversation.

        Uses Agora's NATIVE Gemini LLM vendor (not a custom-LLM proxy to
        our own backend) precisely because this agent doesn't need
        schema-constrained structured output or a deterministic
        speak-gate -- ordinary conversational replies are exactly what
        native mode is for, and building a second backend endpoint to
        replicate that would be pure overhead.

        Uses its own agent_uid (a fresh random one, same convention
        server.py already uses for Watcher's) -- reusing Watcher's own
        uid here was an earlier, wrong iteration of this feature: an
        avatar attached to Watcher's own session makes the avatar look
        like Watcher, not an independent stand-in for the absent lead.

        Uses Anam (ANAM_API_KEY/ANAM_AVATAR_ID), a first-class Agora
        avatar vendor -- not GenericAvatar, which this was originally
        built against before real avatar credentials existed anywhere in
        this project. Anam needs no api_base_url (Agora's backend already
        knows how to reach it) and no agora_uid on the avatar config
        itself (unlike GenericAvatar, which requires one); the RTC join
        details are handled entirely on Agora's side for a named vendor.

        Still not fully verified end-to-end: the native Gemini
        system_messages shape here (Content-style role/parts) is a
        best-effort match to Gemini's own API shape, and the TTS voice_id
        ("English_magnetic_voiced_man") is unverified against the real
        MiniMax catalog since no BYOK key exists here either.

        Delayed START_DELAY_SECONDS before actually starting: confirmed
        live, with no delay this agent's own greeting starts almost the
        same instant Watcher's does, so they talk over each other instead
        of the avatar speaking after Watcher's intro. Agora has no native
        primitive for sequencing two independent agents' turns (RTM
        pub/sub is the general mechanism suggested for agent coordination,
        but there's no ready-made "wait for the other agent to finish"
        signal) -- this is a fixed-delay heuristic sized to roughly how
        long Watcher's own short default greeting takes to speak, not a
        measured or guaranteed handoff.
        """
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY not set -- cannot start the delegate avatar agent's native LLM for channel=%s",
                channel_name,
            )
            return

        await asyncio.sleep(self.DELEGATE_AVATAR_START_DELAY_SECONDS)

        approver_name = delegate_info.get("approver_name") or "the resolved approver"
        delegate_notes = delegate_info["delegate_notes"]

        greeting = (
            f"Hi, I'm standing in for {approver_name} today, who couldn't join. "
            f"Here's their update: {delegate_notes} Now, who's working on what?"
        )
        system_prompt = (
            f"You are a stand-in representative for {approver_name}, who approved this "
            f"incident but couldn't personally join the call. Their own update on what "
            f"they've done and what they want covered: \"{delegate_notes}\"\n\n"
            "Open the conversation with that update, in your own words but faithful to "
            "what they said. Then actively ask the other participants for status -- what "
            "each of them is working on, what's blocking them, what's still unclear. "
            "Respond naturally when addressed, as if relaying on their behalf. You are "
            "NOT responsible for tracking facts, decisions, or action items -- a separate "
            "system on this call already does that; your only job is representing "
            f"{approver_name} and keeping the conversation moving. Keep turns brief and "
            "conversational, not a formal report."
        )

        avatar_agent_uid = random.randint(10000000, 99999999)

        stt = DeepgramSTT(model="nova-3", language="en-IN")
        # url overridden to Gemini's non-streaming generateContent endpoint
        # (dropping streamGenerateContent?alt=sse, which this vendor class
        # hardcodes by default). Root cause per Agora's own optimize-latency
        # doc: streaming forwards each sentence to TTS/avatar rendering as
        # soon as it's ready, which is exactly the mute/unmute-every-chunk
        # pattern confirmed live in the browser console (audio+video both
        # cut out and back in together every ~2-3s, matching a per-sentence
        # publish cycle). Non-streaming waits for Gemini's complete response
        # before handing it to TTS/avatar, trading a bit of initial latency
        # (acceptable here -- this agent already waits
        # DELEGATE_AVATAR_START_DELAY_SECONDS before even starting) for one
        # continuous render instead of many.
        #
        # NOT independently verified against a live listener as of this
        # change (no human available to confirm smoothness) -- confirmed
        # only that Gemini's plain generateContent endpoint itself returns a
        # normal, complete response with our real key and this model.
        # style="gemini" is hardcoded by this vendor's to_config()
        # regardless of URL; whether Agora's backend parses a non-streaming
        # body the same way is the one part genuinely unverified. Bounded
        # risk: this agent is fully independent of Watcher's own session
        # (wrapped in try/except in the caller), so a bad interaction here
        # cannot affect Watcher -- worst case is the delegate avatar failing
        # to start at all, which Agora's own agent-status API (GET
        # /v2/projects/{appid}/agents/{agent_id}) can confirm or rule out
        # after the fact via its stop reason.
        llm = Gemini(
            api_key=gemini_api_key,
            model="gemini-flash-lite-latest",
            url=(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-flash-lite-latest:generateContent?key=" + gemini_api_key
            ),
            system_messages=[{"role": "user", "parts": [{"text": system_prompt}]}],
            greeting_message=greeting,
        )
        # sample_rate=24000 pinned explicitly, unlike Watcher's own
        # MiniMaxTTS (agent.py, no avatar attached): Anam's own docs
        # default to expecting 24000Hz and let it be configured (16000 /
        # 24000 / 48000). Watcher's audio publishes straight to the RTC
        # channel with no intermediate consumer, so MiniMax's own
        # (unspecified, provider-default) output rate never has to match
        # anything else. Here, Anam sits between our TTS output and the
        # final published track, re-rendering it into synced video -- a
        # mismatch between what MiniMax actually outputs and what Anam
        # assumes it's receiving would produce exactly the symptom seen
        # live: tracks report healthy/playing at the transport level, but
        # the perceptual audio is silent. Pinning both sides to the same
        # explicit value removes the guesswork.
        tts = MiniMaxTTS(model="speech_2_6_turbo", voice_id="English_magnetic_voiced_man", sample_rate=24000)

        delegate_agent = (
            AgoraAgent(
                client=self.client,
                instructions=system_prompt,
                greeting=greeting,
                failure_message="One moment.",
                # Disabled, unlike Watcher's own start_of_speech
                # interruption -- confirmed live symptom: the avatar's
                # participant tile kept flipping muted/unmuted with no
                # audible speech ever completing, consistent with it
                # self-interrupting on echo of its own voice picked up by
                # the human's mic (this agent listens only to that one
                # uid, per remote_uids, so anything reflected back through
                # their mic reads as them starting to speak). Watcher
                # doesn't show the same symptom, but its greeting is much
                # shorter and only spoken once, so the same echo path may
                # just be less likely to land mid-sentence there.
                interruption={"enable": False},
                advanced_features={"enable_rtm": True},
            )
            .with_stt(stt)
            .with_llm(llm)
            .with_tts(tts)
            .with_avatar(AnamAvatar(
                api_key=os.getenv("ANAM_API_KEY"),
                avatar_id=os.getenv("ANAM_AVATAR_ID"),
                # Confirmed against Agora's own live Anam docs (the user
                # pasted the actual doc page content): agora_uid and
                # agora_token are BOTH required fields for Anam, same as
                # GenericAvatar -- but this installed SDK version's
                # AnamAvatarOptions Python model only exposes api_key and
                # avatar_id, missing both. additional_params is the escape
                # hatch (merged into the request dict verbatim by
                # to_config()) to send what the live API actually needs
                # despite the SDK's own bindings being behind it. Without
                # this, avatar.enable=true was accepted by Agora's API
                # (no 400 error) but its audio never played on the
                # client -- consistent with the avatar's RTC identity
                # never being correctly established.
                additional_params={
                    "agora_uid": str(avatar_agent_uid),
                    "agora_token": generate_convo_ai_token(
                        app_id=self.app_id,
                        app_certificate=self.app_certificate,
                        channel_name=channel_name,
                        uid=avatar_agent_uid,
                        token_expire=3600,
                    ),
                    # Matches the MiniMaxTTS sample_rate=24000 set above --
                    # stated explicitly on both sides rather than relying
                    # on Anam's documented default (24000) happening to
                    # match whatever MiniMax would have used unset.
                    "sample_rate": 24000,
                    # Confirmed live (2026-09-12): the avatar's voice broke
                    # up repeatedly. Left unset, this defaults to "high"
                    # (per Agora's own Anam integration doc) -- real-time
                    # H264 video plus synced audio at high quality is a
                    # meaningfully heavier bandwidth/rendering load than
                    # audio alone, and this network has been the recurring
                    # root cause behind other flakiness all session (tunnel
                    # drops, etc). Stepped down explicitly rather than
                    # leaving it to the heavier default.
                    "quality": "medium",
                },
            ))
        )

        session = delegate_agent.create_async_session(
            channel=channel_name,
            agent_uid=str(avatar_agent_uid),
            # Confirmed live: Agora rejects remote_uids=["*"] outright
            # ("subscribing to all remote RTC UIDs is not allowed") on any
            # session with an avatar enabled -- unlike Watcher's own
            # wildcard subscription above, which has no avatar and is
            # unaffected. Only the joiner who actually triggered this
            # /startAgent call is a known uid at this point; anyone who
            # joins the room later won't be heard by this agent specifically
            # (Watcher's own wildcard subscription still hears everyone,
            # since only this second agent carries the avatar).
            remote_uids=[str(user_uid)],
            enable_string_uid=False,
            idle_timeout=30,
            expires_in=3600,
        )

        logger.info(
            "Starting delegate avatar agent channel=%s agent_uid=%s", channel_name, avatar_agent_uid
        )
        agent_id = await session.start()
        self._sessions[agent_id] = session
        self._channel_delegate_agents[channel_name] = agent_id
        logger.info("Started delegate avatar agent agent_id=%s channel=%s", agent_id, channel_name)

    async def stop(self, agent_id: str) -> None:
        """
        Stop a running agent. Falls back to the stateless client path.

        The web client only ever learns Watcher's own agent_id (see
        result["agent_id"] in _start_locked) -- it has no way to ask for
        the delegate avatar agent specifically. So stopping Watcher for a
        channel also stops that channel's delegate avatar agent, if one
        is running, keeping "End Conversation" a single action from the
        caller's side even though two independent agents may be in the
        room.
        """
        if not agent_id or not str(agent_id).strip():
            raise ValueError("agent_id is required and cannot be empty")

        stale_channels = [
            channel for channel, entry in self._channel_agents.items() if entry[0] == agent_id
        ]
        for channel in stale_channels:
            self._channel_agents.pop(channel, None)
            delegate_agent_id = self._channel_delegate_agents.pop(channel, None)
            if delegate_agent_id:
                try:
                    await self.stop(delegate_agent_id)
                except Exception:
                    logger.exception(
                        "Failed to stop delegate avatar agent agent_id=%s for channel=%s",
                        delegate_agent_id,
                        channel,
                    )

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

    async def _get_status(self, agent_id: str) -> Optional[str]:
        """
        Polls the real agent status (IDLE/STARTING/RUNNING/STOPPING/
        STOPPED/FAILED) -- used by switch_language to confirm a stopped
        agent has actually released its channel identity before restarting
        with the same agent_uid. Replicates stop_agent's own token-building
        (pool_client.py) since client.agents.get() needs the same
        app-credentials auth header stop_agent already builds for itself,
        and this method isn't exposed as a top-level client convenience the
        way stop_agent is. Returns None on any failure -- caller treats
        that as "unknown, proceed anyway" rather than blocking forever.
        """
        request_options = None
        if self.client.auth_mode == "app-credentials":
            token = generate_convo_ai_token(
                app_id=self.app_id, app_certificate=self.app_certificate, channel_name="status", uid=0,
            )
            request_options = {"additional_headers": {"Authorization": f"agora token={token}"}}
        try:
            response = await self.client.agents.get(self.app_id, agent_id, request_options=request_options)
            return response.status
        except Exception:
            logger.warning("Failed to query agent status agent_id=%s", agent_id, exc_info=True)
            return None

    async def switch_language(self, channel_name: str, target_language: str) -> Dict[str, Any]:
        """
        Handoff to a different STT/TTS tier mid-call -- the only way to
        change vendor, confirmed exhaustively from the installed SDK's own
        UpdateAgentsRequestProperties (token/llm/mllm only, no asr/tts
        field exists at all). Stop the current agent, wait for it to
        actually release the channel (the ERR_REPEAT_JOIN_REQUEST risk
        this project hit once already, incident-55), then start a fresh
        one with the SAME agent_uid/user_uid so the room's "Watcher"
        identity doesn't change across the switch.

        Raises if no agent is currently tracked for this channel -- there's
        nothing to hand off from.
        """
        existing = self._channel_agents.get(channel_name)
        if existing is None:
            raise ValueError(f"No agent running for channel={channel_name} to switch from")
        old_agent_id, _, agent_uid, user_uid = existing

        logger.info(
            "Switching language channel=%s target=%s old_agent_id=%s",
            channel_name, target_language, old_agent_id,
        )
        await self.stop(old_agent_id)

        # Bounded poll, not indefinite -- proceed regardless after the
        # timeout rather than risk hanging the switch forever on a status
        # query that never resolves. This is the safeguard, not a
        # guarantee: the actual ERR_REPEAT_JOIN_REQUEST risk is reduced,
        # not eliminated, by waiting.
        for _ in range(20):  # ~10s at 500ms per poll
            status = await self._get_status(old_agent_id)
            if status in ("STOPPED", "IDLE") or status is None:
                break
            await asyncio.sleep(0.5)

        display_name = LANGUAGE_DISPLAY_NAMES.get(target_language, target_language)
        ithink_base = os.getenv("ITHINK_BACKEND_BASE_URL", "http://127.0.0.1:8123/api/v1")
        greeting_english = f"I'm back -- now listening in {display_name}."
        # Only this post-handoff greeting gets translated, deliberately --
        # it's spoken by the tier being ENTERED, which is the one actually
        # built for the target language. The pre-handoff acknowledgment
        # (build_language_switch_ack, in iCall_utils.py) stays in English
        # on purpose: it's spoken by the tier being LEFT, which may not
        # render the target language correctly at all (that's often
        # exactly why the call is leaving it). Best-effort -- falls back
        # to the English line on any failure, never blocks the handoff.
        greeting_override = greeting_english
        if target_language != "multi":
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(
                        f"{ithink_base}/icall/translate-line",
                        json={"text": greeting_english, "language_code": target_language},
                    )
                    response.raise_for_status()
                    greeting_override = response.json().get("data", {}).get("translated") or greeting_english
            except Exception:
                logger.warning(
                    "Failed to translate post-handoff greeting, using English channel=%s target=%s",
                    channel_name, target_language, exc_info=True,
                )

        result = await self.start(
            channel_name, agent_uid=agent_uid, user_uid=user_uid,
            language_code=target_language,
            greeting_override=greeting_override,
        )

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.patch(
                    f"{ithink_base}/icall/channel/{channel_name}/language",
                    json={"code": target_language},
                )
        except Exception:
            logger.warning(
                "Failed to confirm language switch to backend channel=%s target=%s",
                channel_name, target_language, exc_info=True,
            )
        return result
