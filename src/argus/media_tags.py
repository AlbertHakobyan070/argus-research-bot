"""Tag and group suggestions for the media library (spec 2026-10-26).

``suggest`` asks the ``media.tag`` role (cheap tier, JSON) for 3–8 tags, one
group, a short gist and the links worth keeping, from what is known about an
item: its title, the downloader's description and the start of a transcript.
It is told the tags and groups already confirmed, so the vocabulary stays
small. ``extract_links`` pulls URLs and GitHub repos deterministically, so a
repo named in a description is kept even when the model drops it.

Nothing here confirms anything: the result goes to ``meta.suggest`` and the
item waits for review (``meta.review = "pending"``).
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from . import llm
from .engines import registry
from .graph.jsonx import parse_json_obj
from .media import extract_urls

MAX_TAGS = 8
MAX_LINKS = 25
_TAG_OK = re.compile(r"[^a-z0-9+#.-]+")
# github.com/<owner>/<repo>, with or without a scheme (a transcript says it aloud)
_GITHUB = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9-]{1,39})/([A-Za-z0-9._-]{1,100})", re.I)
_NOT_REPOS = frozenset({"orgs", "topics", "sponsors", "features", "marketplace", "settings", "about"})

SYSTEM = """You file a personal library of saved videos about software engineering, AI agents, LLMs and similar
topics so their owner can find them again months later, when they no longer remember what is inside.

Answer with ONE JSON object and nothing else:
{"tags": [3 to 8 short lowercase tags], "group": "one broad group", "summary": "2-3 sentences",
 "links": [{"url": "...", "label": "what it is"}]}

Rules:
- Tags name what the video is ABOUT and what a person would search for: tools, libraries, repos, techniques,
  concepts (e.g. "langgraph", "rag", "prompt-caching", "claude-code", "system-design"). Lowercase, hyphens for
  spaces, no '#'. Prefer an EXISTING tag when one fits; invent a new one only when none does.
- group is one broad shelf (e.g. "AI agents", "LLM engineering", "Dev tooling", "Career"). Reuse an EXISTING group
  when one fits.
- summary says what a viewer learns or gets from it, concretely (the repo, the workflow, the advice).
- links: only URLs that appear in the material below (repos, tools, docs). Never invent one. [] when none.
- Work only from the material given; if it is thin (a title only), keep the tags few and general."""


def normalize_tag(raw: Any) -> str:
    """A tag as the library stores it: lowercase, hyphens, at most 32 chars; "" when nothing is left."""
    t = _TAG_OK.sub("-", str(raw or "").strip().lower().lstrip("#")).strip("-.")
    return t[:32].rstrip("-.")


def extract_links(*texts: str) -> list[dict]:
    """URLs and GitHub repos in ``texts``: repos first (normalized to
    ``https://github.com/owner/repo``), then other links; deduped, capped."""
    repos: dict[str, dict] = {}
    others: dict[str, dict] = {}
    for text in texts:
        for owner, repo in _GITHUB.findall(text or ""):
            repo = re.sub(r"\.git$", "", repo.rstrip(".,;:)"))
            if owner.lower() in _NOT_REPOS or not repo:
                continue
            url = f"https://github.com/{owner}/{repo}"
            repos.setdefault(url.lower(), {"url": url, "kind": "repo", "label": f"{owner}/{repo}"})
        for url in extract_urls(text or ""):
            if not _GITHUB.match(url):
                others.setdefault(url.lower(), {"url": url, "kind": "link", "label": ""})
    return [*repos.values(), *others.values()][:MAX_LINKS]


def _material(title: str, description: str, transcript: str, url: str) -> str:
    parts = [f"Title: {title or '(none)'}"]
    if url:
        parts.append(f"Source: {url}")
    if description.strip():
        parts.append(f"Description:\n{description.strip()[:3000]}")
    if transcript.strip():
        parts.append(f"Transcript (start):\n{transcript.strip()[:6000]}")
    return "\n\n".join(parts)


def _parse(text: str) -> dict:
    obj = parse_json_obj(text, require_any=("tags",))
    if not isinstance(obj.get("tags"), list):
        raise ValueError("tags is not a list")
    tags = list(dict.fromkeys(t for t in (normalize_tag(x) for x in obj["tags"]) if t))[:MAX_TAGS]
    if not tags:
        raise ValueError("no usable tags")
    links = [{"url": str(link.get("url")), "label": str(link.get("label") or "")[:120]}
             for link in obj.get("links") or [] if isinstance(link, dict) and str(link.get("url") or "").startswith("http")]
    return {"tags": tags, "group": str(obj.get("group") or "").strip()[:48],
            "summary": str(obj.get("summary") or "").strip()[:800], "links": links}


async def suggest(*, title: str, description: str = "", transcript: str = "", url: str = "",
                  vocabulary: list[str] = (), groups: list[str] = ()) -> dict:
    """The model's suggestion plus the deterministic links: ``{tags, group,
    summary, links, model}``. Raises ``llm.EngineError`` (and its quota
    subclass) when no engine answers usefully."""
    known = (f"EXISTING tags (most used first): {json.dumps(list(vocabulary)[:80])}\n"
             f"EXISTING groups: {json.dumps(list(groups)[:30])}\n\n")
    engines = await registry.active_snapshot()
    messages = [SystemMessage(content=SYSTEM),
                HumanMessage(content=known + _material(title, description, transcript, url))]
    res = await asyncio.to_thread(llm.complete, "media.tag", messages, engines=engines, temperature=0.2,
                                  max_tokens=700, validate=_parse)
    out = dict(res.parsed)
    seen = set()
    links = []
    for link in [*extract_links(description, transcript), *out["links"]]:
        key = link["url"].rstrip("/").lower()
        if key not in seen:
            seen.add(key)
            links.append({"url": link["url"], "label": link.get("label") or "", "kind": link.get("kind", "link")})
    out["links"] = links[:MAX_LINKS]
    out["model"] = res.records[-1].served_model if res.records else ""
    return out
