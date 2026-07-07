"""Argus configuration loader. Reads secrets from .env, never hardcodes."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (A:\Hermes\Agents\argus\.env).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration. Frozen so it can be hashed/cached."""

    freellmapi_base_url: str
    freellmapi_api_key: str
    telegram_bot_token: str
    telegram_allowed_user_id: int

    # Optional knobs (with sane defaults).
    reports_root: Path
    checkpoint_db: Path
    langgraph_dir: Path
    request_timeout_seconds: float
    max_revision_rounds: int

    @classmethod
    def load(cls) -> "Settings":
        base = os.environ.get("FREELLMAPI_BASE_URL", "http://127.0.0.1:3001/v1")
        key = os.environ.get("FREELLMAPI_API_KEY", "")
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        allowed = int(os.environ.get("TELEGRAM_ALLOWED_USER_ID", "0") or "0")

        if not key or not key.startswith("freellmapi-"):
            raise RuntimeError(
                "FREELLMAPI_API_KEY missing or malformed in .env "
                "(expected 'freellmapi-<key>')."
            )
        if not token or ":" not in token:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN missing or malformed in .env "
                "(expected '<id>:<secret>' from @BotFather)."
            )

        reports_root = Path(
            os.environ.get("ARGUS_REPORTS_ROOT", r"A:\Hermes\Downloads\reports")
        )
        checkpoint_db = Path(
            os.environ.get("ARGUS_CHECKPOINT_DB", str(_PROJECT_ROOT / "argus_checkpoints.sqlite"))
        )
        langgraph_dir = Path(
            os.environ.get("ARGUS_LANGGRAPH_DIR", str(_PROJECT_ROOT / ".langgraph_checkpoints"))
        )
        langgraph_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            freellmapi_base_url=base.rstrip("/"),
            freellmapi_api_key=key,
            telegram_bot_token=token,
            telegram_allowed_user_id=allowed,
            reports_root=reports_root,
            checkpoint_db=checkpoint_db,
            langgraph_dir=langgraph_dir,
            request_timeout_seconds=float(os.environ.get("ARGUS_TIMEOUT", "60")),
            max_revision_rounds=int(os.environ.get("ARGUS_MAX_REVISIONS", "3")),
        )


# Convenience singleton (lazy).
_cached: Settings | None = None


def get_settings() -> Settings:
    global _cached
    if _cached is None:
        _cached = Settings.load()
    return _cached