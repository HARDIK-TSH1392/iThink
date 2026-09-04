from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):

    # For local development
    database_url: str = "sqlite+aiosqlite:///./ithink_dev.db"

    gemini_api_key: str = ""

    # From Agora Console -> Project -> notification config. Empty until a
    # webhook is actually registered there (needs a public HTTPS URL, so
    # this stays unset for local dev). See iCall_utils.verify_agora_signature.
    agora_webhook_secret: str = ""

    # Slack Incoming Webhook URL, for the minimal notify-on-approval
    # orchestration action. Empty until one is configured.
    slack_webhook_url: str = ""

    # Slack app's Signing Secret (Basic Information -> App Credentials),
    # used to verify interactive button clicks are genuinely from Slack.
    slack_signing_secret: str = ""

    # Bot User OAuth Token (xoxb-...), used to DM a specific resolved
    # approver rather than posting the approval-request to the channel.
    slack_bot_token: str = ""

    # Jira Cloud REST API v3 -- second approval gate before ticket creation.
    jira_site_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = ""

    # Base URL of the voice-agent web client (voice-agent/web) that a
    # responder actually clicks to join the call. Defaults to localhost --
    # only reachable from this machine -- swap for a real public URL (once
    # a stable tunnel/deployment exists) without touching any calling code.
    voice_agent_web_base_url: str = "http://localhost:3000"

    # The isolated github_mcp_service (see backend/github_mcp_service/) --
    # kept out of process from this backend because the official mcp SDK's
    # SSE support needs a starlette version that conflicts with FastAPI's
    # pin here. Called over plain HTTP, same as Slack/Jira.
    github_mcp_service_url: str = "http://localhost:8003"

    # voice-agent/server's own FastAPI service (agent lifecycle + the
    # _channel_names join registry) -- called from the silence-trigger path
    # in iCall_api.chat_completions_endpoint to check live participant count
    # before nudging an empty room. Reverse direction of the usual call
    # (voice-agent calls this backend, not the other way around), but same
    # "separate local service, plain HTTP" pattern.
    voice_agent_server_url: str = "http://localhost:8002"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()