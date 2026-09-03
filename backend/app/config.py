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

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()