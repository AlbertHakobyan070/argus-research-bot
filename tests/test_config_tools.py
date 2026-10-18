"""Argus config + tool wrapper tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from argus import config
from argus.tools import (
    snatch_url, crawl_url, normalize_to_markdown,
)


def test_settings_load_from_env():
    s = config.get_settings()
    assert s.freellmapi_api_key.startswith("freellmapi-")
    assert ":" in s.telegram_bot_token
    assert s.freellmapi_base_url.startswith("http")


def test_normalize_handles_missing(monkeypatch):
    """normalize_to_markdown against an unreachable host should not crash."""
    r = normalize_to_markdown("https://no.such.host.invalid", timeout=10)
    assert r.ok is False
    assert r.error

def test_path_settings_expand_tilde(monkeypatch, tmp_path):
    """On macOS/Linux the natural .env value is ~/DS_Vault/... and an
    unexpanded Path("~/x") is a literal folder named "~" — which would
    silently create one inside the repo instead of using $HOME."""
    import argus.config as config_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ARGUS_VAULT_ROOT", "~/DS_Vault/Argus")
    monkeypatch.setenv("FREELLMAPI_API_KEY", "freellmapi-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test")

    s = config_mod.Settings.load()
    assert "~" not in str(s.vault_root), s.vault_root
    assert s.vault_root == tmp_path / "DS_Vault" / "Argus"
    # derived roots inherit the expansion
    assert s.media_root == tmp_path / "DS_Vault" / "Argus" / "media"


def test_ffmpeg_env_expands_tilde(monkeypatch, tmp_path):
    from argus.media import resolve_ffmpeg

    fake = tmp_path / "bin" / "ffmpeg"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("ARGUS_FFMPEG", "~/bin/ffmpeg")
    assert resolve_ffmpeg() == str(fake)
