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

    # Base URL of the voice-agent web client (voice-agent/web), used to build
    # the join link put in the approved-incident Slack message: {this}/?channel={name}.
    # Only correct for whoever's machine is actually running that client --
    # sharing the resulting link across machines needs that client to be
    # reachable from wherever the recipient is, same class of problem as the
    # Agora/Slack webhook public-URL requirement, not solved by this setting.
    voice_client_base_url: str = "http://localhost:3000"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()