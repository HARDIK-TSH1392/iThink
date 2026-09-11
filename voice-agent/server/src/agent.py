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
from agora_agent.agentkit.token import generate_convo_ai_token
from agora_agent.agentkit.vendors import (
    CustomLLM, DeepgramSTT, MiniMaxTTS, OpenAI, SarvamSTT, SarvamTTS,
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
            # greeting_override is set by switch_language after a handoff
            # ("I'm back -- now listening in Tamil.") -- reuses the
            # existing, already-proven greeting-on-join code path rather
            # than introducing a separate agent_think()/speak() call for
            # this specific notice.
            greeting_message=greeting_override or self.greeting,
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
            # Makes stop() wait for the agent to finish its current
            # sentence (reach IDLE) before actually leaving, up to the
            # timeout -- without this, a language-switch handoff's stop()
            # could cut the agent off mid-word right as the switch fires.
            # Also benefits ordinary /stopAgent calls, not just handoffs.
            "farewell_config": {"graceful_enabled": True, "graceful_timeout_seconds": 5},
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
        self._channel_agents[channel_name] = (agent_id, result, agent_uid, user_uid)
        return result

    async def stop(self, agent_id: str) -> None:
        """Stop a running agent. Falls back to the stateless client path."""
        if not agent_id or not str(agent_id).strip():
            raise ValueError("agent_id is required and cannot be empty")

        stale_channels = [
            channel for channel, entry in self._channel_agents.items() if entry[0] == agent_id
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
