"""Anchors: the links the user typed into the question are read before the
brief (graph/anchors.py), excerpted into the brief's prompt, pinned as sources
the scout keeps and tags with every sub-question, and listed at the plan gate.

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_anchors.py -q
"""
from __future__ import annotations

import json

from argus.bot.research import _format_plan
from argus.engines.fetchers import FetchResult
from argus.graph import anchors as anchors_mod
from argus.graph import scout as scout_mod
from argus.graph.anchors import extract_links
from argus.graph.brief import brief_node

from helpers import fake_llm, make_run_config

REPO = "https://github.com/acme/widget"
README = "# Widget\n\nWidget is a Telegram research bot that keeps its reports in a local vault. " * 8
BRIEF = {
    "sub_questions": [{"q": "What does the Widget bot do for its users?", "kind": "repo"},
                      {"q": "Which paid research tools compete with Widget?", "kind": "web"},
                      {"q": "What would hosting Widget cost per month?", "kind": "web"}],
    "must_have_keywords": ["widget", "research bot"], "summary": "Assess Widget.",
    "success_criteria": ["names its features"], "entities": ["Widget"],
}


DOC = "Widget's architecture: a LangGraph pipeline with a brief, a scout, research waves and a panel. " * 6
RELEASES = [{"tag_name": "v0.3.0", "name": "Media library", "published_at": "2026-10-20T10:00:00Z",
             "body": "Adds a media library with tags. " * 20}]


def _stub_reading(monkeypatch, tmp_path, *, ok=True, docs=("architecture.md", "pricing-notes.md", "faq.md"),
                  releases=RELEASES):
    """``fetchers.fetch``, the GitHub API and raw files on fakes; returns the list of URLs read."""
    fetched: list[str] = []
    page = tmp_path / "readme.md"
    page.write_text(README, encoding="utf-8")

    def fetch(source, **kw):
        fetched.append(source["url"])
        assert kw["pinned"] is True
        return (FetchResult(ok=True, markdown_path=str(page), title="", chars=len(README), fetcher="native")
                if ok else FetchResult(ok=False, error="native: HTTP 404"))

    def api(path):
        if path.endswith("/contents/docs"):
            return [{"type": "file", "name": n, "path": f"docs/{n}"} for n in docs]
        if "/releases" in path:
            return list(releases)
        raise AssertionError(f"unexpected GitHub API call {path}")

    def raw(url, cfg):
        fetched.append(url)
        return DOC
    monkeypatch.setattr(anchors_mod.fetchers, "fetch", fetch)
    monkeypatch.setattr(anchors_mod, "_github_api", api)
    monkeypatch.setattr(anchors_mod, "_raw", raw)
    monkeypatch.setattr(anchors_mod, "_github_meta", lambda owner, repo: (
        f"GitHub repository {owner}/{repo} — a research bot\nLanguage: Python · Stars: 3", "a research bot",
        {}, ["README.md", "docs/", "src/"], ""))
    return fetched


def _state(request, **extra):
    return {"user_request": request, "run_config": make_run_config("short"), **extra}


def test_extract_links_trims_punctuation_dedupes_and_caps():
    text = (f"Assess this project ({REPO}). Also {REPO}/ and https://example.com/a, "
            "https://example.org/b; https://example.net/c — and ftp://x.y/z")
    assert extract_links(text) == [REPO, "https://example.com/a", "https://example.org/b"]
    assert extract_links("no links here") == []


def test_brief_reads_the_link_and_pins_it(monkeypatch, tmp_path):
    fetched = _stub_reading(monkeypatch, tmp_path)
    seen = fake_llm(monkeypatch, lambda system, human: json.dumps(BRIEF))
    out = brief_node(_state(f"Research the monetisation potential of {REPO}"))
    assert fetched[0] == REPO
    human = seen[0][1]
    assert "CONTEXT FROM THE USER'S LINKS" in human and "GitHub repository acme/widget" in human
    assert "Telegram research bot" in human                       # the README excerpt reaches the brief
    [a] = out["anchors"]
    assert (a["ok"], a["kind"], a["title"], a["about"]) == (True, "repo", "acme/widget", "a research bot")
    src = out["sources"][0]
    assert src["pinned"] and src["source"] == "anchor" and src["url"] == REPO
    assert src["text"].startswith("GitHub repository acme/widget") and "local vault" in src["text"]
    assert all(s["pinned"] and s["anchor_of"] == REPO for s in out["sources"])
    assert out["plan"]["anchors"][0]["url"] == REPO and "context" not in out["plan"]["anchors"][0]
    assert "search_spend" in out and "1 of your 1 link(s) read" in out["messages"][0]["content"]


def test_a_link_the_intake_rewrite_dropped_is_still_read(monkeypatch, tmp_path):
    fetched = _stub_reading(monkeypatch, tmp_path)
    fake_llm(monkeypatch, lambda system, human: json.dumps(BRIEF))
    out = brief_node(_state("Analyze the business potential of the project from the provided GitHub URL.",
                            messages=[{"role": "user", "content": f"Research the monetisation potential of {REPO}"}]))
    assert fetched[0] == REPO and out["anchors"][0]["url"] == REPO


def test_a_redraft_keeps_links_already_read(monkeypatch, tmp_path):
    fetched = _stub_reading(monkeypatch, tmp_path)
    seen = fake_llm(monkeypatch, lambda system, human: json.dumps(BRIEF))
    first = brief_node(_state(f"Assess {REPO}"))
    reads = len(fetched)
    again = brief_node(_state(f"Assess {REPO}", plan_feedback="focus on hosting costs", brief=first["brief"],
                              anchors=first["anchors"], sources=first["sources"]))
    assert len(fetched) == reads                                  # nothing read a second time
    assert "sources" not in again                                 # the pinned copy from the first draft stands
    assert again["anchors"] == first["anchors"] and "Telegram research bot" in seen[-1][1]


def test_a_link_that_cannot_be_read_is_shown_not_pinned(monkeypatch, tmp_path):
    _stub_reading(monkeypatch, tmp_path, ok=False, docs=(), releases=())
    monkeypatch.setattr(anchors_mod, "_github_meta", lambda owner, repo: ("", "", {}, [], "GitHub API: HTTP 404"))
    seen = fake_llm(monkeypatch, lambda system, human: json.dumps(BRIEF))
    out = brief_node(_state(f"Assess {REPO}"))
    [a] = out["anchors"]
    assert not a["ok"] and "404" in a["error"]
    assert "sources" not in out and "CONTEXT FROM THE USER'S LINKS" not in seen[0][1]
    assert any("not read" in e for e in out["errors"]) and any("GitHub API" in e for e in out["errors"])


def test_scout_keeps_the_anchor_first_and_tags_every_sub_question(monkeypatch, tmp_path):
    _stub_reading(monkeypatch, tmp_path)
    fake_llm(monkeypatch, lambda system, human: json.dumps(BRIEF))
    state = _state(f"Assess {REPO}")
    state.update(brief_node(state))
    monkeypatch.setattr(scout_mod, "plan_queries", lambda brief, **kw: (
        [{"query": "widget bot", "provider": "ddgs", "sub_qs": [0]}], [], []))
    hit = {"url": REPO, "title": "acme/widget (search copy)", "sub_qs": [0], "source": "github-search", "text": ""}
    other = {"url": "https://example.com/rival", "title": "Rival", "sub_qs": [1], "source": "ddgs", "text": ""}
    monkeypatch.setattr(scout_mod, "run_wave", lambda queries, **kw: ([hit, other], []))
    out = scout_mod.scout_node(state)
    anchored = [s for s in out["sources"] if s["source"] == "anchor"]
    assert out["sources"][0]["url"] == REPO and len(anchored) > 1
    assert all(s["sub_qs"] == [0, 1, 2] for s in anchored)
    assert [s["url"] for s in out["sources"] if s["source"] != "anchor"] == ["https://example.com/rival"]  # the
    # search copy of the repo lost the dedupe to the pinned anchor
    assert out["plan"]["anchors"][0]["title"] == "acme/widget"


def _read(request, rc=None):
    from argus.engines.ledger import SpendLedger
    from argus.graph.run_config import RunConfig
    rc = rc or RunConfig.model_validate(make_run_config("short"))
    return anchors_mod.read_anchors(request, rc=rc, ledger=SpendLedger.for_run(rc, None))


def test_a_repo_root_reads_the_parts_that_fit_the_question(monkeypatch, tmp_path):
    _stub_reading(monkeypatch, tmp_path, docs=("faq.md", "architecture.md", "pricing-notes.md", "changelog-old.md"))
    [a], sources, errors = _read(f"What does the architecture of {REPO} look like, and its pricing?")
    whys = [(p["title"], p["why"]) for p in a["pages"]]
    assert len(a["pages"]) == anchors_mod.EXTRA_MAX and errors == []
    assert whys[:2] in ([("docs/architecture.md", "documentation"), ("docs/pricing-notes.md", "documentation")],
                        [("docs/pricing-notes.md", "documentation"), ("docs/architecture.md", "documentation")])
    assert {s["url"] for s in sources} >= {REPO, f"{REPO}/blob/HEAD/docs/architecture.md"}
    assert all(s["anchor_of"] == REPO and s["pinned"] for s in sources)


def test_release_words_rank_the_releases_first(monkeypatch, tmp_path):
    _stub_reading(monkeypatch, tmp_path)
    [a], sources, _ = _read(f"What changed in the latest releases of {REPO}?")
    assert a["pages"][0]["why"] == "latest releases"
    rel = next(s for s in sources if s["url"] == f"{REPO}/releases")
    assert "v0.3.0" in rel["text"] and "Media library" in rel["text"]


def test_only_the_readme_or_a_deep_link_is_read_alone(monkeypatch, tmp_path):
    _stub_reading(monkeypatch, tmp_path)
    [a], sources, _ = _read(f"Summarise only the README of {REPO}")
    assert a["pages"] == [] and len(sources) == 1
    [b], sources, _ = _read("Explain https://example.com/docs/streaming for me")
    assert b["pages"] == [] and len(sources) == 1                  # a deep link: the user pointed at that page


def test_a_site_root_follows_its_own_representative_links(monkeypatch, tmp_path):
    _stub_reading(monkeypatch, tmp_path)
    html = b"""<html><body><a href="/docs/">Docs</a> <a href="/pricing">Pricing</a> <a href="/login">Log in</a>
      <a href="https://other.org/docs">Elsewhere</a> <a href="/careers">Careers</a> <a href="/logo.png">x</a>
      <a href="/blog/hello">Hello</a> <a href="#top">Top</a></body></html>"""
    monkeypatch.setattr(anchors_mod.fetchers, "_get", lambda client, url, cfg: (url, html, "text/html; charset=utf-8"))
    [a], sources, _ = _read("How much does https://widget.example/ cost to use?")
    picked = [p["url"] for p in a["pages"]]
    assert picked[:2] == ["https://widget.example/docs/", "https://widget.example/pricing"] or \
        picked[:2] == ["https://widget.example/pricing", "https://widget.example/docs/"]
    assert not any(x in u for u in picked for x in ("login", "careers", "other.org", "logo.png"))


def test_the_plan_card_lists_the_links_apart_from_live_search():
    anchor = {"url": REPO, "title": "acme/widget", "kind": "repo", "ok": True, "chars": 5120,
              "about": "a research bot", "error": "",
              "pages": [{"url": f"{REPO}/releases", "title": "Releases", "why": "latest releases", "ok": True,
                         "chars": 900, "error": ""}]}
    bad = {"url": "https://example.com/gone", "title": "example.com/gone", "kind": "web", "ok": False,
           "chars": 0, "about": "", "error": "native: HTTP 404"}
    card = _format_plan({"plan": {"summary": "s", "anchors": [anchor, bad]},
                         "sources": [{"url": REPO, "title": "acme/widget", "source": "anchor"},
                                     {"url": "https://example.com/rival", "title": "Rival", "source": "ddgs"}]})
    assert "*Read from your links:*" in card
    assert f"[acme/widget]({REPO}) — 5,120 chars — a research bot" in card
    assert "not read: native: HTTP 404" in card
    assert f"↳ also read: [Releases]({REPO}/releases)" in card
    assert "*Found sources (1, live search):*" in card              # the anchor is not counted twice
