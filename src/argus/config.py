"""Argus configuration loader. Reads secrets from .env, never hardcodes."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (three levels above this file).
# override=False: variables already set in the process environment win.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _path_env(name: str, default: str | Path) -> Path:
    """Read a path from the environment, expanding a leading ``~``.

    ``Path("~/DS_Vault")`` is a *literal* directory called "~", so on
    macOS/Linux — where writing ``~/...`` in .env is the natural thing to
    do — an unexpanded value silently creates a "~" folder inside the
    repo instead of pointing at the home directory.
    """
    raw = os.environ.get(name) or str(default)
    return Path(raw).expanduser()


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
    library_db: Path
    vault_root: Path
    media_root: Path
    transcripts_root: Path
    history_root: Path
    ffmpeg_path: str | None         # unread: media.resolve_ffmpeg reads ARGUS_FFMPEG itself
    request_timeout_seconds: float
    max_revision_rounds: int        # unread: panel.route_after_panel reads ARGUS_MAX_REVISIONS itself

    @classmethod
    def load(cls) -> "Settings":
        base = os.environ.get("FREELLMAPI_BASE_URL", "http://127.0.0.1:3001/v1")
        key = os.environ.get("FREELLMAPI_API_KEY", "")
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        # 0 (unset) is accepted here: the bot still starts, and
        # bot._allowed() then rejects every user.
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

        # DS vault — the single home for everything Argus produces.
        # Directories are created lazily by the writers (media downloader,
        # report builder, mirror), NOT here: Settings.load() must stay
        # filesystem-neutral so hermetic tests / CI (no A:\ drive) work.
        vault_root = _path_env(
            "ARGUS_VAULT_ROOT",
            r"A:\DS_Vault\DS Main Vault\DS_AUA_Toolkit\Assets\Argus",
        )
        media_root = vault_root / "media"
        transcripts_root = vault_root / "transcripts"
        history_root = vault_root / "research history"

        # Reports live in the vault research-history folder by default
        # (v2 decision). ARGUS_REPORTS_ROOT still overrides for tests.
        reports_root = _path_env("ARGUS_REPORTS_ROOT", history_root)
        checkpoint_db = _path_env(
            "ARGUS_CHECKPOINT_DB", _PROJECT_ROOT / "argus_checkpoints.sqlite")
        library_db = _path_env(
            "ARGUS_LIBRARY_DB", _PROJECT_ROOT / "argus_library.sqlite")

        return cls(
            freellmapi_base_url=base.rstrip("/"),
            freellmapi_api_key=key,
            telegram_bot_token=token,
            telegram_allowed_user_id=allowed,
            reports_root=reports_root,
            checkpoint_db=checkpoint_db,
            library_db=library_db,
            vault_root=vault_root,
            media_root=media_root,
            transcripts_root=transcripts_root,
            history_root=history_root,
            ffmpeg_path=os.environ.get("ARGUS_FFMPEG") or None,
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