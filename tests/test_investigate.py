"""Investigate (spec docs/superpowers/specs/2026-11-27-investigate-design.md): the finder's extraction,
grounding and GitHub checks, the bookmarks table, the attach and bookmark flows in media_service, the chat
(mx:inv, rp:*, /repos, bk:*) and the console API. No network: GitHub, the web and the model are fakes."""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp.test_utils import TestClient, TestServer

from argus import investigate as finder
from argus import media_service as ms
from argus.bot.common import _tg_len
from argus.library import Library
from webapp_fakes import FakeBot, FakeRuns, auth, make_app, media_settings


@pytest.fixture
async def lib(tmp_path):
    library = Library(tmp_path / "lib.sqlite")
    await library.open()
    yield library
    await library.close()


@pytest.fixture
def roots(tmp_path):
    s = media_settings(tmp_path / "vault")
    for r in (s.media_root, s.transcripts_root, s.history_root, s.reports_root):
        Path(r).mkdir(parents=True)
    return s


@pytest.fixture(autouse=True)
def _clean():
    ms._INVESTIGATING.clear()
    yield
    ms._INVESTIGATING.clear()


def _file(root: Path, rel: str, data: bytes = b"x" * 1000) -> Path:
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


async def _add(lib, path: Path, kind: str = "media", **kw) -> int:
    return await lib.add_asset(kind=kind, path=str(path), size_bytes=path.stat().st_size,
                               title=kw.pop("title", path.stem), **kw)


def _found(name: str, **kw) -> dict:
    return {"kind": "repo", "key": name.lower(), "name": name, "url": f"https://github.com/{name}", "about": "A kit",
            "said": "agent kit", "stars": 5, "archived": False, "fork": False, "renamed_from": None,
            "from": "description", "quote": f"github.com/{name}", "at_s": None, "how": "link", "sure": True,
            "on": True, **kw}


def _fake_llm(monkeypatch, mentions: list[dict], picks: dict | None = None) -> list:
    """The model: ``mentions`` for the reading call; for a pick call, ``picks[name]`` ("" or absent: none)."""
    seen = []

    async def snapshot():
        return SimpleNamespace(profile="test")

    def complete(role, messages, *, engines, validate, **kw):
        seen.append((role, messages[1].content))
        if messages[0].content == finder.PICK:
            name = re.search(r"^Name: (.*)$", messages[1].content, re.M).group(1)
            answer = {"pick": (picks or {}).get(name, "")}
        else:
            answer = {"mentions": mentions}
        return SimpleNamespace(parsed=validate(json.dumps(answer)), records=[SimpleNamespace(served_model="m1")])

    monkeypatch.setattr(finder.registry, "active_snapshot", snapshot)
    monkeypatch.setattr(finder.llm, "complete", complete)
    return seen


async def _settle():
    """Let every background job (and its follow-up) finish."""
    while ms._TASKS:
        await asyncio.gather(*list(ms._TASKS))


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def test_github_links_take_repos_and_skip_gists_and_site_pages():
    text = ("Code: https://github.com/acme/agent-kit/tree/main/docs, also github.com/Acme/Agent-Kit.git. "
            "Gist https://gist.github.com/alice/0123abcd, https://github.com/orgs/acme/people, "
            "docs at https://acme.github.io/agent-kit/ and raw.githubusercontent.com/bob/tool/main/x.sh")
    assert [(o, r) for o, r, _ in finder.github_links(text)] == [
        ("acme", "agent-kit"), ("Acme", "Agent-Kit"), ("bob", "tool"), ("acme", "agent-kit")]


def test_bare_slugs_only_where_they_read_as_repos():
    assert [(o, r) for o, r, _ in finder.bare_slugs("1jehuang/jcode", whole=True)] == [("1jehuang", "jcode")]
    desc = ("OtterMind/Chat2DB\nStar it on GitHub: nicobailon/visual-explainer\n"
            "see docs/readme.md and src/main\nrepo and/or 24/7 km/h")
    assert [(o, r) for o, r, _ in finder.bare_slugs(desc)] == [("OtterMind", "Chat2DB"),
                                                                ("nicobailon", "visual-explainer")]


def test_material_reads_both_titles_the_uploaders_comments_and_a_clean_transcript():
    info = {"title": "Video by marc", "description": "d", "tags": ["GitHub", ""], "uploader_id": "u1",
            "chapters": [{"title": "Intro"}, {"x": 1}],
            "comments": [{"author": "marc", "author_id": "u1", "text": "Repo: github.com/acme/kit"},
                         {"author": "fan", "author_id": "u2", "text": "github.com/spam/spam"}]}
    m = finder.material(title="https://github.com/acme/kit", info=info,
                        transcript="# T\n# source: https://x\n# backend: b (en)\n\n[0:05] hello\nWEBVTT\n"
                                   "00:01.000 --> 00:02.000\n1\nbye")
    assert m.title == "https://github.com/acme/kit\nVideo by marc"
    assert m.comments == ["Repo: github.com/acme/kit"] and m.tags == ["GitHub"] and m.chapters == ["Intro"]
    assert m.transcript == "[0:05] hello\nbye"


def test_deterministic_finds_say_where_and_when():
    m = finder.Material(title="x", description="Get it: https://t.co/abc",
                        transcript="[1:02] go to github.com/acme/kit now",
                        expanded={"https://t.co/abc": "https://github.com/bob/tool"})
    assert [(c["owner"], c["repo"], c["from"], c["how"], c["at_s"], c["sure"]) for c in finder.deterministic(m)] == [
        ("bob", "tool", "description", "short link", None, True), ("acme", "kit", "transcript", "link", 62, True)]


def test_transcript_chunks_cut_at_lines_and_say_when_the_rest_was_not_read(monkeypatch):
    monkeypatch.setattr(finder, "CHUNK", 10)
    monkeypatch.setattr(finder, "MAX_CHUNKS", 2)
    assert finder.transcript_chunks("aaaa\nbbbb\ncccc\n" + "x" * 25) == (["aaaa\nbbbb\n", "cccc\n"], True)
    assert finder.transcript_chunks("short") == (["short"], False)


def test_readme_lead_skips_badges_headings_html_and_quotes():
    md = ("# Kit\n[![ci](b.svg)](x) [![pypi](p.svg)](y)\n<p align='center'><img src='l.png'></p>\n\n> note\n\n"
          "A **fast** [agent](https://x) kit\nfor Python.\n\n## Install")
    assert finder.readme_lead(md) == "A fast agent kit for Python."


def test_parse_mentions_cleans_and_refuses_a_bad_shape():
    with pytest.raises(ValueError):
        finder.parse_mentions('{"items": []}')
    out = finder.parse_mentions(json.dumps({"mentions": [
        {"name": " LangGraph ", "kind": "repo", "owner": "lang chain", "repo": "x", "url": "ftp://x",
         "quote": " we  use it ", "about": "graphs"}, {"name": ""}, {"name": "Cursor", "kind": "app"}]}))
    assert out[0] == {"name": "LangGraph", "kind": "repo", "owner": "", "repo": "", "url": "", "quote": "we use it",
                      "about": "graphs"}
    assert len(out) == 2 and out[1]["name"] == "Cursor" and out[1]["kind"] == "tool"


def test_ground_keeps_the_materials_own_words_and_drops_inventions():
    m = finder.Material(description="Tools I use", transcript="[0:59] and then we use\n[1:00] lang graph for the agents")
    assert finder.ground(m, {"name": "LangGraph", "quote": "we use lang graph for the agents"}) == {
        "from": "transcript", "quote": "we use lang graph for the agents", "at_s": 59}
    named = finder.ground(m, {"name": "LangGraph", "quote": "LangGraph is great"})   # the quote is not there; the name is
    assert named["from"] == "transcript" and "lang graph" in named["quote"] and named["at_s"] == 60
    assert finder.ground(m, {"name": "AutoGen", "quote": "we love autogen a lot"}) is None


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def _github(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(finder.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_gh_repo_follows_renames_and_sorts_githubs_answers(monkeypatch):
    seen = []

    def handler(req):
        seen.append(req.headers.get("authorization"))
        p = req.url.path
        if p == "/repos/old/name":
            return httpx.Response(301, headers={"location": "https://api.github.com/repositories/7"})
        if p == "/repositories/7":
            return httpx.Response(200, json={"full_name": "new/name", "html_url": "https://github.com/new/name",
                                             "description": " A  kit ", "stargazers_count": 1200, "private": False})
        if p == "/repos/gone/x":
            return httpx.Response(404, json={"message": "Not Found"})
        if p == "/repos/secret/x":
            return httpx.Response(200, json={"full_name": "secret/x", "private": True})
        if p == "/repos/limit/x":
            return httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000000"},
                                  json={"message": "API rate limit exceeded"})
        return httpx.Response(500)

    _github(monkeypatch, handler)
    monkeypatch.setenv("GITHUB_TOKEN", "t0k")
    assert finder.gh_repo("old", "name") == {"name": "new/name", "url": "https://github.com/new/name", "about": "A kit",
                                             "stars": 1200, "archived": False, "fork": False}
    assert seen[0] == "Bearer t0k"
    assert finder.gh_repo("gone", "x") is None and finder.gh_repo("secret", "x") is None
    with pytest.raises(finder.GitHubLimit) as e:
        finder.gh_repo("limit", "x")
    assert e.value.reset == 1700000000
    with pytest.raises(finder.GitHubError):
        finder.gh_repo("broken", "x")


def test_about_from_github_falls_back_to_the_readme(monkeypatch):
    monkeypatch.setattr(finder, "gh_repo", lambda o, r: {"name": "acme/kit", "about": "", "url": "", "stars": 1,
                                                          "archived": False, "fork": False})
    monkeypatch.setattr(finder, "gh_readme", lambda o, r: "# Kit\n\nKit builds agents.\n")
    assert finder.about_from_github("acme", "kit") == "Kit builds agents."
    monkeypatch.setattr(finder, "gh_readme", lambda o, r: "")
    with pytest.raises(ValueError, match="no description"):
        finder.about_from_github("acme", "kit")


async def test_find_checks_everything_and_reports_what_it_dropped(monkeypatch):
    m = finder.Material(title="Agents talk",
                        description="Code: github.com/acme/kit\nOld: github.com/old/name\nDead: github.com/gone/x",
                        transcript="[0:10] we use crew AI for the agents\n[0:20] and the Cursor editor\n"
                                   "[0:30] also PentAGI is cool")
    repo = {"acme/kit": "acme/kit", "old/name": "new/name", "crewaiinc/crewai": "crewAIInc/crewAI"}

    def gh(o, r):
        name = repo.get(f"{o}/{r}".lower())
        return name and {"name": name, "url": f"https://github.com/{name}", "about": "about", "stars": 10,
                         "archived": False, "fork": False}

    monkeypatch.setattr(finder, "gh_repo", gh)
    monkeypatch.setattr(finder, "gh_search", lambda name: [
        {"full_name": "x/crewai-tools", "name": "crewai-tools", "stars": 9000},
        {"full_name": "crewAIInc/crewAI", "name": "crewAI", "stars": 61000}] if name.lower() == "crewai" else [])
    monkeypatch.setattr(finder, "ddgs_hits", lambda q, max_results=8: (
        ([{"url": "https://www.cursor.com/", "snippet": ""}], "") if q.startswith("Cursor") else ([], "")))
    monkeypatch.setattr(finder, "open_page", lambda url: ("https://cursor.com/", "Cursor - The AI Code Editor"))
    seen = _fake_llm(monkeypatch, [
        {"name": "Kit", "kind": "repo", "owner": "acme", "repo": "kit", "quote": "Code: github.com/acme/kit",
         "about": "agent kit"},
        {"name": "CrewAI", "kind": "repo", "quote": "we use crew AI for the agents", "about": "multi-agent framework"},
        {"name": "Cursor", "kind": "tool", "quote": "and the Cursor editor", "about": "AI editor"},
        {"name": "PentAGI", "kind": "repo", "quote": "also PentAGI is cool", "about": "pentesting agent"},
        {"name": "AutoGen", "kind": "repo", "quote": "autogen changed my life", "about": "invented"}],
        picks={"CrewAI": "crewAIInc/crewAI"})
    out = await finder.find(m)
    assert seen[0][0] == "media.investigate" and "github.com/acme/kit" in seen[0][1] and "part 1 of 1" in seen[0][1]
    pick = next(body for _, body in seen if body.startswith("Name: CrewAI"))
    assert "- x/crewai-tools (★9000)" in pick and "- crewAIInc/crewAI (★61000)" in pick
    assert "multi-agent framework" in pick
    assert [(f["name"], f["sure"], f["on"], f["how"]) for f in out["found"]] == [
        ("acme/kit", True, True, "link"), ("new/name", True, True, "link"),
        ("crewAIInc/crewAI", False, False, "search"), ("Cursor", False, False, "search")]
    kit, renamed, crew, cursor = out["found"]
    assert kit["said"] == "agent kit" and renamed["renamed_from"] == "old/name"
    assert crew["from"] == "transcript" and crew["at_s"] == 10 and crew["quote"] == "we use crew AI for the agents"
    assert cursor["kind"] == "tool" and cursor["key"] == "https://cursor.com" and cursor["about"].startswith("Cursor")
    assert out["skipped"]["not_found"] == ["gone/x"] and out["skipped"]["unresolved"] == ["PentAGI"]
    assert out["skipped"]["unchecked"] == [] and out["model"] == "m1"
    assert "AutoGen" not in json.dumps(out)                    # the model made it up: never shown


async def test_a_repo_that_only_shares_the_name_is_not_taken(monkeypatch):
    """Live, "Stryker" (mutation testing) had matched a CPU-Z exploit tool by exact name and stars."""
    m = finder.Material(transcript="[0:05] Stryker plants small bugs in your code\n[0:09] and hookify writes the rules")
    monkeypatch.setattr(finder, "gh_repo", lambda o, r: {"name": f"{o}/{r}", "url": f"https://github.com/{o}/{r}",
                                                         "about": "a", "stars": 1, "archived": False, "fork": False})
    monkeypatch.setattr(finder, "gh_search", lambda name: {
        "Stryker": [{"full_name": "hfiref0x/Stryker", "name": "Stryker", "stars": 113, "about": "CPU-Z CVE exploit"},
                    {"full_name": "stryker-mutator/stryker-js", "name": "stryker-js", "stars": 2600,
                     "about": "Mutation testing for JavaScript"}],
        "hookify": [{"full_name": "kevinpiac/hookify", "name": "hookify", "stars": 24, "about": "objc hooks"}]}[name])
    web = []
    monkeypatch.setattr(finder, "ddgs_hits", lambda q, max_results=8: web.append(q) or (
        [{"url": "https://github.com/anthropics/claude-code/tree/main/plugins/hookify", "snippet": "the plugin"},
         {"url": "https://github.com/other/thing", "snippet": "no name in the URL"}], ""))
    seen = _fake_llm(monkeypatch, [
        {"name": "Stryker", "kind": "repo", "quote": "Stryker plants small bugs in your code", "about": "mutation testing"},
        {"name": "hookify", "kind": "repo", "quote": "and hookify writes the rules", "about": "Claude Code plugin"}],
        picks={"Stryker": "stryker-mutator/stryker-js"})
    out = await finder.find(m)
    assert [(f["name"], f["sure"], f["on"], f["how"]) for f in out["found"]] == [
        ("stryker-mutator/stryker-js", False, False, "search")]
    assert out["skipped"]["unresolved"] == ["hookify"] and web == ["hookify github"]
    hookify = [body for _, body in seen if body.startswith("Name: hookify")]
    assert len(hookify) == 2 and "kevinpiac/hookify (★24): objc hooks" in hookify[0]
    assert "- anthropics/claude-code: the plugin" in hookify[1] and "other/thing" not in hookify[1]


async def test_an_owner_the_model_supplies_starts_unticked(monkeypatch):
    m = finder.Material(transcript="[0:01] grab Langflow AI slash Langflow today\n[0:04] and never trust Claude Code")
    monkeypatch.setattr(finder, "gh_repo", lambda o, r: {"name": f"{o}/{r}", "url": f"https://github.com/{o}/{r}",
                                                         "about": "a", "stars": 1, "archived": False, "fork": False})
    _fake_llm(monkeypatch, [
        {"name": "Langflow", "kind": "repo", "owner": "langflow-ai", "repo": "langflow",
         "quote": "grab Langflow AI slash Langflow today", "about": "agent canvas"},
        {"name": "Claude Code", "kind": "repo", "owner": "anthropics", "repo": "claude-code",
         "quote": "and never trust Claude Code", "about": "coding agent"}])
    out = await finder.find(m)
    assert [(f["name"], f["sure"], f["on"], f["how"]) for f in out["found"]] == [
        ("langflow-ai/langflow", True, True, "said"), ("anthropics/claude-code", False, False, "said")]


async def test_github_limit_stops_the_checks_and_counts_the_rest(monkeypatch):
    calls = []

    def gh(o, r):
        calls.append(r)
        if r == "two":
            raise finder.GitHubLimit(0)
        return {"name": f"{o}/{r}", "url": "u", "about": "", "stars": 1, "archived": False, "fork": False}

    monkeypatch.setattr(finder, "gh_repo", gh)
    _fake_llm(monkeypatch, [])
    out = await finder.find(finder.Material(description="github.com/a/one\ngithub.com/b/two\ngithub.com/c/three"))
    assert [f["name"] for f in out["found"]] == ["a/one"] and calls == ["one", "two"]
    assert out["skipped"]["unchecked"] == ["b/two", "c/three"] and "rate limit" in out["skipped"]["stopped"]


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------

async def test_bookmarks_upsert_keep_your_note_and_count_their_items(lib, roots):
    a = await _add(lib, _file(roots.media_root, "x/a.mp4"), title="A talk")
    b1 = await lib.add_bookmark(kind="repo", key="acme/kit", name="acme/kit", url="https://github.com/acme/kit",
                                note="from the video", stars=5, asset_id=a)
    b2 = await lib.add_bookmark(kind="repo", key="acme/kit", name="Acme/Kit", url="https://github.com/Acme/Kit",
                                note="other")
    assert b2["bookmark_id"] == b1["bookmark_id"] and b2["note"] == "from the video" and b2["stars"] == 5
    assert b2["name"] == "Acme/Kit" and b2["items"] == 0
    await lib.update_asset_meta(a, {"repos": [{"kind": "repo", "key": "acme/kit", "name": "acme/kit", "url": "u"}]})
    rows, total = await lib.list_bookmarks(q="video")
    assert total == 1 and rows[0]["items"] == 1
    assert [i["asset_id"] for i in await lib.bookmark_items("repo", "acme/kit")] == [a]
    assert await lib.bookmarked_keys(["acme/kit", "x/y"]) == {"acme/kit"}
    with pytest.raises(ValueError):
        await lib.add_bookmark(kind="site", key="k", name="n", url="u")
    with pytest.raises(ValueError):
        await lib.update_bookmark(b1["bookmark_id"], key="x")
    await lib.delete_assets([a])
    assert (await lib.get_bookmark(b1["bookmark_id"]))["asset_id"] is None       # the item went, the bookmark stays
    assert await lib.delete_bookmark(b1["bookmark_id"]) and await lib.get_bookmark(b1["bookmark_id"]) is None


async def test_an_existing_library_gains_the_bookmarks_table_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.sqlite"
    first = Library(path)
    await first.open()
    aid = await first.add_asset(kind="media", path=str(tmp_path / "v.mp4"), title="Old")
    await first.close()
    db = sqlite3.connect(path)
    db.execute("DROP TABLE bookmarks")
    db.commit()
    db.close()
    again = Library(path)
    await again.open()
    try:
        assert (await again.get_asset(aid))["title"] == "Old"
        assert (await again.list_bookmarks()) == ([], 0)
    finally:
        await again.close()


# ---------------------------------------------------------------------------
# media_service: investigate, ticks, attach, bookmarks
# ---------------------------------------------------------------------------

async def test_investigate_transcribes_first_and_keeps_a_pending_result(lib, roots, monkeypatch):
    info = _file(roots.media_root, "x/talk.info.json", json.dumps({"description": "Code: github.com/acme/kit"}).encode())
    aid = await _add(lib, _file(roots.media_root, "x/talk.mp4"), title="Talk", platform="youtube",
                     meta={"info_json": str(info)})
    seen = {}

    async def transcribe(lib_, s, asset_id, job, **kw):
        seen["kw"] = kw
        t = _file(roots.transcripts_root, "youtube/talk.txt", b"# Talk\n\nwe use crew AI")
        return await lib_.add_asset(kind="transcript", path=str(t), size_bytes=10, title="Talk",
                                    meta={"media_asset_id": asset_id})

    async def find(m, *, stage=None):
        seen["m"] = m
        if stage is not None:
            await stage("checking the candidates")
        return {"found": [_found("acme/kit")], "model": "m1", "skipped": {}, "read": {"transcript": bool(m.transcript)}}

    monkeypatch.setattr(ms, "transcribe", transcribe)
    monkeypatch.setattr(ms.finder, "find", find)
    stages = []

    async def stage(text):
        stages.append(text)

    job = ms.Job(id="j", kind="investigate", title="t")
    await ms.investigate(lib, roots, aid, job, stage=stage)
    assert seen["kw"] == {"timestamps": False}                                   # YouTube: captions first
    assert seen["m"].transcript == "we use crew AI" and "github.com/acme/kit" in seen["m"].description
    inv = (await lib.get_asset(aid))["meta"]["investigate"]
    assert inv["rev"] == 1 and inv["state"] == "pending" and inv["found"][0]["key"] == "acme/kit" and inv["note"] == ""
    assert stages[0].startswith("transcribing") and stages[-1] == "checking the candidates"
    assert job.message == "1 repo(s), 0 tool(s) or site(s)"
    seen.clear()
    await ms.investigate(lib, roots, aid, job)
    assert "kw" not in seen and (await lib.get_asset(aid))["meta"]["investigate"]["rev"] == 2


async def test_a_failed_transcript_is_said_and_the_rest_still_read(lib, roots, monkeypatch):
    aid = await _add(lib, _file(roots.media_root, "x/reel.mp4"), title="Reel", platform="instagram")

    async def transcribe(*a, **kw):
        raise ms.MediaError("speech-to-text is not installed")

    async def find(m, *, stage=None):
        return {"found": [], "model": "m1", "skipped": {}, "read": {"transcript": False}}

    monkeypatch.setattr(ms, "transcribe", transcribe)
    monkeypatch.setattr(ms.finder, "find", find)
    await ms.investigate(lib, roots, aid, ms.Job(id="j", kind="investigate", title="t"))
    inv = (await lib.get_asset(aid))["meta"]["investigate"]
    assert inv["state"] == "done" and inv["note"] == "No transcript: speech-to-text is not installed"


async def test_one_investigation_at_a_time_per_item(lib, roots, monkeypatch):
    gate = asyncio.Event()

    async def slow(lib_, s, asset_id, job, *, stage=None):
        await gate.wait()
        return asset_id

    monkeypatch.setattr(ms, "investigate", slow)
    aid = await _add(lib, _file(roots.media_root, "x/a.mp4"))
    job = ms.investigate_job(lib, roots, aid, "a")
    with pytest.raises(ms.MediaError, match="already"):
        ms.investigate_job(lib, roots, aid, "a")
    gate.set()
    await _settle()
    assert job.state == "done"
    assert ms.investigate_job(lib, roots, aid, "a") is not job                   # free again
    await _settle()


async def test_ticks_attach_and_bookmark_follow_the_latest_list(lib, roots):
    aid = await _add(lib, _file(roots.media_root, "x/a.mp4"))
    await lib.update_asset_meta(aid, {"investigate": {"rev": 3, "state": "pending", "found": [
        _found("acme/kit"), _found("bob/tool", sure=False, on=False)]}})
    with pytest.raises(ms.MediaError, match="out of date"):
        await ms.toggle_found(lib, aid, 2, 0)
    await ms.toggle_found(lib, aid, 3, 1)
    await ms.toggle_found(lib, aid, 3, 0)
    row, chosen = await ms.attach_chosen(lib, aid, 3, bookmark=True)
    assert [c["key"] for c in chosen] == ["bob/tool"]
    assert [r["key"] for r in row["meta"]["repos"]] == ["bob/tool"] and row["meta"]["investigate"]["state"] == "done"
    b = await lib.get_bookmark_by_key("repo", "bob/tool")
    assert b["note"] == "agent kit" and b["asset_id"] == aid and b["stars"] == 5
    row = await ms.set_repos(lib, aid, ["acme/kit"])                             # the console sets the exact list
    assert [r["key"] for r in row["meta"]["repos"]] == ["acme/kit"]
    with pytest.raises(ValueError, match="not found"):
        await ms.set_repos(lib, aid, ["nobody/x"])
    with pytest.raises(ValueError, match="attached"):
        await ms.set_repos(lib, aid, ["acme/kit"], bookmark=["bob/tool"])
    await ms.toggle_found(lib, aid, 3, 1)                                        # untick the last ticked one
    with pytest.raises(ms.MediaError, match="nothing is ticked"):
        await ms.attach_chosen(lib, aid, 3)


async def test_bookmark_about_fills_from_the_project_and_says_why_not(lib, monkeypatch):
    b = await lib.add_bookmark(kind="repo", key="acme/kit", name="acme/kit", url="https://github.com/acme/kit")
    monkeypatch.setattr(ms.finder, "about_from_github", lambda o, r: f"{o}/{r}: a kit")
    assert (await ms.bookmark_about(lib, b["bookmark_id"]))["about"] == "acme/kit: a kit"

    def limited(o, r):
        raise ms.finder.GitHubLimit(0)

    monkeypatch.setattr(ms.finder, "about_from_github", limited)
    with pytest.raises(ms.MediaError, match="rate limit"):
        await ms.bookmark_about(lib, b["bookmark_id"])
    tool = await lib.add_bookmark(kind="tool", key="https://cursor.com", name="Cursor", url="https://cursor.com")
    monkeypatch.setattr(ms.finder, "open_page", lambda url: (url, "Cursor - The AI Code Editor"))
    assert (await ms.bookmark_about(lib, tool["bookmark_id"]))["about"] == "Cursor - The AI Code Editor"
    with pytest.raises(ms.MediaError, match="no bookmark"):
        await ms.bookmark_about(lib, 999)
    assert (await ms.set_bookmark_note(lib, b["bookmark_id"], "  my   note "))["note"] == "my note"
    with pytest.raises(ValueError):
        await ms.set_bookmark_note(lib, b["bookmark_id"], 5)


# ---------------------------------------------------------------------------
# the chat
# ---------------------------------------------------------------------------

def _app(lib, public="https://console.example"):
    app = MagicMock()
    app.bot_data = {"library": lib, "webapp_public_url": public}
    app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=77))
    app.bot.edit_message_text = AsyncMock()
    return app


def _cb(app, data):
    q = MagicMock()
    q.data, q.answer = data, AsyncMock()
    q.message.chat.id, q.message.message_id = 42, 9
    update = MagicMock()
    update.callback_query, update.effective_user.id = q, 42
    ctx = MagicMock()
    ctx.application = app
    return update, ctx


def _datas(markup) -> list[str]:
    return [b.callback_data for row in markup.inline_keyboard for b in row if b.callback_data]


@pytest.fixture
def dispatch(monkeypatch, roots):
    from argus.bot import app as bot_app
    from argus.bot import repos
    monkeypatch.setattr(bot_app, "_allowed", lambda _s, _uid: True)
    monkeypatch.setattr(bot_app, "get_settings", lambda: roots)
    monkeypatch.setattr(repos, "_allowed", lambda _s, _uid: True)
    monkeypatch.setattr(repos, "get_settings", lambda: roots)
    return bot_app.on_callback


async def test_the_investigate_button_follows_the_job_and_shows_ticks(lib, roots, monkeypatch, dispatch):
    from argus.bot import media as bot_media
    aid = await _add(lib, _file(roots.media_root, "x/a.mp4"), title="Agents talk")
    await _add(lib, _file(roots.transcripts_root, "x/a.txt", b"hello"), kind="transcript", meta={"media_asset_id": aid})
    assert f"mx:inv:{aid}" in _datas(bot_media._item_keyboard(_app(lib), await lib.get_asset(aid)))

    async def find(m, *, stage=None):
        await stage("asking the model what the video names")
        return {"found": [_found("acme/kit"), _found("bob/tool", sure=False, on=False,
                                                     **{"from": "transcript", "quote": "use bob tool", "at_s": 72})],
                "model": "m1", "skipped": {"not_found": ["gone/x"]}, "read": {"transcript": True}}

    monkeypatch.setattr(ms.finder, "find", find)
    app = _app(lib)
    await dispatch(*_cb(app, f"mx:inv:{aid}"))
    await _settle()
    assert app.bot.send_message.await_args.kwargs["text"].startswith("🔎 Investigating Agents talk")
    edits = [c.kwargs for c in app.bot.edit_message_text.await_args_list]
    assert "asking the model" in edits[0]["text"]
    text, markup = edits[-1]["text"], edits[-1]["reply_markup"]
    assert '☑ <a href="https://github.com/acme/kit">acme/kit</a>' in text and "☐ <a" in text
    assert "said: “use bob tool” (1:12)" in text and "gone/x" in text and "looked up" in text
    assert _datas(markup) == [f"rp:t:{aid}.1.0", f"rp:t:{aid}.1.1", f"rp:ok:{aid}.1", f"rp:bm:{aid}.1"]


async def test_ticks_and_attach_through_the_dispatcher(lib, roots, dispatch):
    aid = await _add(lib, _file(roots.media_root, "x/a.mp4"), title="Agents talk")
    await lib.update_asset_meta(aid, {"investigate": {"rev": 2, "state": "pending", "found": [
        _found("acme/kit"), _found("bob/tool", on=False)]}})
    app = _app(lib)
    await dispatch(*_cb(app, f"rp:t:{aid}.2.1"))
    assert (await lib.get_asset(aid))["meta"]["investigate"]["found"][1]["on"] is True
    labels = [b.text for row in app.bot.edit_message_text.await_args.kwargs["reply_markup"].inline_keyboard
              for b in row]
    assert "☑ 2 tool" in labels
    await dispatch(*_cb(app, f"rp:t:{aid}.1.0"))
    assert "out of date" in app.bot.send_message.await_args.kwargs["text"]
    await dispatch(*_cb(app, f"rp:bm:{aid}.2"))
    meta = (await lib.get_asset(aid))["meta"]
    assert [r["key"] for r in meta["repos"]] == ["acme/kit", "bob/tool"]
    assert "bookmarked" in app.bot.send_message.await_args.kwargs["text"]
    final = app.bot.edit_message_text.await_args.kwargs
    assert "📎 Attached: acme/kit, bob/tool" in final["text"] and _datas(final["reply_markup"]) == []
    assert (await lib.list_bookmarks())[1] == 2


async def test_repos_lists_opens_fills_and_removes(lib, roots, monkeypatch, dispatch):
    from argus.bot import repos
    b = await lib.add_bookmark(kind="repo", key="acme/kit", name="acme/kit", url="https://github.com/acme/kit",
                               note="agent kit", stars=1500)
    app = _app(lib)
    update = MagicMock()
    update.effective_user.id = 42
    update.message.reply_text = AsyncMock()
    ctx = MagicMock()
    ctx.application = app
    await repos.repos_cmd(update, ctx)
    text, kw = update.message.reply_text.await_args.args[0], update.message.reply_text.await_args.kwargs
    assert "acme/kit" in text and "★1.5k" in text and "agent kit" in text
    bid = b["bookmark_id"]
    assert f"bk:o:{bid}.0" in _datas(kw["reply_markup"])
    await dispatch(*_cb(app, f"bk:o:{bid}.0"))
    assert "tap 📥 to fill it" in app.bot.edit_message_text.await_args.kwargs["text"]
    monkeypatch.setattr(ms.finder, "about_from_github", lambda o, r: "Build agents")
    await dispatch(*_cb(app, f"bk:gh:{bid}.0"))
    assert "<b>From GitHub:</b> Build agents" in app.bot.edit_message_text.await_args.kwargs["text"]
    await dispatch(*_cb(app, f"bk:rm:{bid}.0"))
    assert "Remove this bookmark?" in app.bot.edit_message_text.await_args.kwargs["text"]
    assert await lib.get_bookmark(bid) is not None
    await dispatch(*_cb(app, f"bk:rmy:{bid}.0"))
    assert await lib.get_bookmark(bid) is None
    assert "No bookmarks yet" in app.bot.edit_message_text.await_args.kwargs["text"]


def test_result_view_fits_telegram_escapes_and_keeps_callbacks_short():
    from argus.bot import repos
    found = [_found(f"owner{i}/{'r' * 90}{i}", about="<script>" + "a" * 300, quote="q" * 300) for i in range(16)]
    a = {"asset_id": 123456, "title": "T" * 200, "path": "x.mp4", "meta": {"investigate": {
        "rev": 12, "state": "pending", "found": found, "skipped": {}, "read": {"transcript": True}}}}
    text, markup = repos.result_view(a, "https://console.example")
    assert _tg_len(text) <= 4096 and "<script>" not in text
    assert all(len(d.encode()) <= 64 for d in _datas(markup))
    a["meta"]["investigate"].update(found=found[:1], skipped={"errors": ["ddgs: timed out <b>"]})
    text, _ = repos.result_view(a, None)
    assert "Lookups that failed: ddgs: timed out &lt;b&gt;" in text       # never dropped silently


# ---------------------------------------------------------------------------
# the console API
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(lib, roots):
    app = make_app(FakeRuns(), reports_root=roots.reports_root, library=lib, bot=FakeBot(), settings=roots,
                   bot_data={"webapp_ok": True, "webapp_public_url": "https://console.example"})
    await ms.save_settings(lib, {"auto_tag": False})
    async with TestClient(TestServer(app)) as c:
        yield c


async def test_api_investigate_attach_and_bookmarks(client, lib, roots, monkeypatch):
    aid = await _add(lib, _file(roots.media_root, "x/a.mp4"), title="Talk")
    gate = asyncio.Event()

    async def fake(lib_, s, asset_id, job, *, stage=None):
        await gate.wait()
        await lib_.update_asset_meta(asset_id, {"investigate": {"rev": 1, "state": "pending", "found": [
            _found("acme/kit"), _found("bob/tool", on=False)], "skipped": {}, "read": {}}})
        return asset_id

    monkeypatch.setattr(ms, "investigate", fake)
    r = await client.post(f"/api/media/{aid}/investigate", headers=auth())
    assert r.status == 200 and (await r.json())["job"]["kind"] == "investigate"
    r = await client.post(f"/api/media/{aid}/investigate", headers=auth())
    assert r.status == 409                                                       # one at a time
    gate.set()
    await _settle()
    body = await (await client.get(f"/api/media/{aid}", headers=auth())).json()
    assert body["item"]["investigate"]["state"] == "pending" and body["item"]["repos"] == [] and body["bookmarked"] == []
    r = await client.put(f"/api/media/{aid}/repos", json={"repos": ["acme/kit", "bob/tool"], "bookmark": ["bob/tool"]},
                         headers=auth())
    item = (await r.json())["item"]
    assert [x["key"] for x in item["repos"]] == ["acme/kit", "bob/tool"] and item["investigate"]["state"] == "done"
    assert (await client.put(f"/api/media/{aid}/repos", json={"repos": ["no/such"]}, headers=auth())).status == 400
    r = await client.post(f"/api/media/{aid}/bookmark", json={"key": "acme/kit"}, headers=auth())
    b = (await r.json())["bookmark"]
    assert b["videos"] == [{"id": aid, "title": "Talk", "type": "video"}]
    listed = await (await client.get("/api/bookmarks?q=acme", headers=auth())).json()
    assert listed["total"] == 1 and listed["bookmarks"][0]["items"] == 1
    body = await (await client.get(f"/api/media/{aid}", headers=auth())).json()
    assert body["bookmarked"] == ["acme/kit", "bob/tool"]
    bid = b["bookmark_id"]
    r = await client.put(f"/api/bookmarks/{bid}", json={"note": "keep"}, headers=auth())
    assert (await r.json())["bookmark"]["note"] == "keep"
    assert (await client.put(f"/api/bookmarks/{bid}", json={"note": 5}, headers=auth())).status == 400
    monkeypatch.setattr(ms.finder, "about_from_github", lambda o, r: "Build agents")
    r = await client.post(f"/api/bookmarks/{bid}/about", headers=auth())
    assert (await r.json())["bookmark"]["about"] == "Build agents"

    def gone(o, r):
        raise ValueError("acme/kit is not a public repo on GitHub (any more)")

    monkeypatch.setattr(ms.finder, "about_from_github", gone)
    r = await client.post(f"/api/bookmarks/{bid}/about", headers=auth())
    assert r.status == 502 and "not a public repo" in (await r.json())["error"]["message"]
    assert (await client.delete(f"/api/bookmarks/{bid}", headers=auth())).status == 200
    assert (await client.delete(f"/api/bookmarks/{bid}", headers=auth())).status == 404
