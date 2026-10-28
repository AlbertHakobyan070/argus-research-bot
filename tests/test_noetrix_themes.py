"""The imported Noetrix themes (static/themes-noetrix.css + .json): each
defines every token a kit theme does and has a swatch, and the appearance
API accepts them by mode."""
from __future__ import annotations

import json
import re

import pytest
from aiohttp.test_utils import TestClient, TestServer

from argus.webapp import api, server
from webapp_fakes import FakeRuns, auth, make_app

STATIC = server.STATIC_DIR
LISTING = json.loads((STATIC / "themes-noetrix.json").read_text(encoding="utf-8"))
CSS = (STATIC / "themes-noetrix.css").read_text(encoding="utf-8")


def _block(css: str, theme: str) -> set[str]:
    m = re.search(r':root\[data-theme="' + theme + r'"\][^{]*\{([^}]*)\}', css)
    assert m, f"no block for {theme}"
    return set(re.findall(r"--([\w-]+):", m.group(1)))


@pytest.mark.parametrize("theme", [t["id"] for t in LISTING])
def test_every_imported_theme_is_complete_and_has_a_swatch(theme):
    kit = (STATIC / "vendor" / "arsenal" / "tokens" / "tokens.css").read_text(encoding="utf-8")
    assert _block(kit, "graphite") <= _block(CSS, theme)
    assert f'.tsw[data-theme-id="{theme}"]' in CSS


@pytest.fixture
async def client(tmp_path):
    async with TestClient(TestServer(make_app(FakeRuns(), reports_root=tmp_path))) as c:
        yield c


async def test_the_api_accepts_imported_themes_by_mode(client):
    light = next(t["id"] for t in LISTING if t["mode"] == "light")
    dark = next(t["id"] for t in LISTING if t["mode"] == "dark")
    r = await client.put("/api/appearance", json={"light": light, "dark": dark}, headers=auth())
    saved = (await r.json())["appearance"]
    assert r.status == 200 and (saved["light"], saved["dark"]) == (light, dark)
    assert (await client.put("/api/appearance", json={"light": dark}, headers=auth())).status == 400
    assert light in api.LIGHT_THEMES and dark in api.DARK_THEMES
