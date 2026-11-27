"""Investigate: the GitHub repos — and the tools and sites — a saved video
mentions or implies, every one checked before it is shown
(spec docs/superpowers/specs/2026-11-27-investigate-design.md).

``find(material)``:

1. Short links (t.co, bit.ly …) in the title, description and the uploader's
   comments are followed to where they land.
2. Deterministic: github.com/owner/repo (not gist.github.com),
   owner.github.io/repo, raw.githubusercontent.com/owner/repo, and a bare
   ``owner/repo`` that is the title, a line of its own or on a line that
   says GitHub / repo.
3. The ``media.investigate`` role names the projects the material mentions,
   each with its words; a mention whose words (or name) the material does not
   contain is dropped — the evidence shown is always the material's own text.
   An owner/repo the model gives is sure only when the words say both.
4. A repo named without owner/repo: the model picks, by description, among
   GitHub's repos whose name contains it (a namesake is not it), then among
   the repos a web search for "<name> github" links; else "no match". A tool
   or site: its link in the material, or a web search whose domain carries
   its name.
5. Every repo through ``GET /repos/{owner}/{repo}``: a redirect is followed
   (a renamed repo shows under its new name), 404 means gone or not public
   (dropped). Every site is fetched through the SSRF guard (2xx after
   redirects; its <title> read).

GitHub allows 60 calls an hour and 10 searches a minute without
``GITHUB_TOKEN`` (5000 and 30 with one). Hitting a limit stops the checks;
what was not checked is counted in the result, never shown unchecked.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from html import unescape
from pathlib import Path

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from . import llm
from .engines import fetchers, registry
from .engines.urls import canonical_url, domain, registrable_domain
from .graph.jsonx import parse_json_obj
from .graph.run_config import FetchConfig
from .graph.search_providers import ddgs_hits
from .media import extract_urls

logger = logging.getLogger("argus.investigate")

MAX_CHECKS = 15          # GET /repos per investigation
MAX_SEARCHES = 6         # GitHub searches per investigation (10 a minute without a token)
MAX_WEB = 6              # web searches per investigation
MAX_SITES = 6            # tool / site pages fetched
MAX_SHORT = 10           # short links followed
MAX_FOUND = 16
CHUNK = 12000            # transcript characters per model call
MAX_CHUNKS = 4           # ponytail: a longer transcript is read in part (the result says so)
_API = "https://api.github.com"

_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_REPO = r"[A-Za-z0-9._-]{1,100}"
_GH_URL = re.compile(rf"(?<![\w.@-])(?:https?://)?(?:www\.)?github\.com/({_OWNER})/({_REPO})", re.I)
_GH_PAGES = re.compile(rf"(?<![\w.-])(?:https?://)?({_OWNER})\.github\.io/({_REPO})", re.I)
_GH_RAW = re.compile(rf"(?<![\w.-])(?:https?://)?raw\.githubusercontent\.com/({_OWNER})/({_REPO})", re.I)
_SLUG = re.compile(rf"(?<![\w./:@#-])({_OWNER})/([A-Za-z0-9][A-Za-z0-9._-]{{0,99}})(?![\w/])")
_TS = re.compile(r"\[(\d+):(\d\d)(?::(\d\d))?\]")
_NOT_OWNERS = frozenset({
    "about", "apps", "codespaces", "collections", "contact", "customer-stories", "dashboard", "enterprise",
    "events", "explore", "features", "issues", "join", "login", "marketplace", "new", "notifications", "orgs",
    "pricing", "pulls", "readme", "resources", "search", "security", "settings", "signup", "site", "solutions",
    "sponsors", "team", "topics", "trending", "users"})
_NOT_SLUGS = frozenset({"and/or", "either/or", "w/o", "n/a", "i/o", "tcp/ip", "km/h", "on/off", "yes/no",
                        "true/false", "input/output", "read/write", "he/she", "his/her", "client/server",
                        "pros/cons", "b/c", "c/o", "a/b"})
_SLUG_HINT = re.compile(r"github|\brepo|⭐|\bstars?\b", re.I)
SHORT_HOSTS = frozenset({"t.co", "bit.ly", "tinyurl.com", "buff.ly", "ow.ly", "rb.gy", "is.gd", "shorturl.at",
                         "cutt.ly", "lnkd.in", "goo.gl", "tiny.cc", "rebrand.ly", "s.id", "dub.sh", "t.ly"})
# Never a tool or site the video presents: where it is posted, the creator's social and money links.
_NOT_SITES = frozenset({"youtube.com", "youtu.be", "instagram.com", "x.com", "twitter.com", "t.me", "tiktok.com",
                        "facebook.com", "fb.com", "linkedin.com", "patreon.com", "discord.gg", "discord.com",
                        "buymeacoffee.com", "ko-fi.com", "amzn.to", "amazon.com", "reddit.com", "threads.net",
                        "linktr.ee", "beacons.ai", "gumroad.com", "substack.com", "github.com", "github.io",
                        "githubusercontent.com", "telegram.me", "whatsapp.com", "wa.me", "bsky.app"})
_FETCH = FetchConfig(max_bytes=1_500_000, timeout_s=15)

SYSTEM = """You read what is known about one saved video — its title, description, tags, chapters, the uploader's
comments and a transcript — and list the software projects it mentions or clearly refers to. The transcript is
speech-to-text: names are often misspelled or split ("lang graph" for LangGraph, "crew AI" for CrewAI), and a
GitHub repo may be spoken as "<owner> slash <repo>".

Answer with ONE JSON object and nothing else:
{"mentions": [{"name": "...", "kind": "repo", "owner": "", "repo": "", "url": "", "quote": "...", "about": "..."}]}

- name: the project's usual name, spelled correctly ("LangGraph", "browser-use").
- kind: "repo" for open-source code (libraries, frameworks, CLIs, MCP servers, apps whose code is on GitHub);
  "tool" for software or apps that are not open source, or when you do not know; "site" for websites and web
  services.
- owner and repo: only when the material gives them (a GitHub link, or spoken as "owner slash repo"); else "".
- url: only a link that appears in the material and belongs to this project; else "".
- quote: the words from the material where it is mentioned, copied exactly (5 to 25 words, not paraphrased).
- about: what the video says it is or does, at most 12 words.
- List every project the video names or shows, most important first, at most 20. Leave out generic ideas (RAG,
  agents, MCP itself), programming languages, the platform the video is posted on, and the creator's own social,
  community, sponsor and affiliate links.
- Never invent a project or a link. {"mentions": []} when there is none."""

PICK = """A video names a software project. Below are its name, the video's words and what the video says it is,
then candidate GitHub repositories. Pick the one that IS that project, judging by name, description and what the
video says; a repo that only shares the name is not it. Answer with ONE JSON object and nothing else:
{"pick": "<owner/repo copied from the list>"}, or {"pick": ""} when none of them clearly is it."""


class GitHubLimit(Exception):
    """GitHub refused for its rate limit; ``reset`` is the epoch second it lifts (0: unknown)."""

    def __init__(self, reset: int = 0):
        super().__init__("GitHub's rate limit")
        self.reset = reset


class GitHubError(Exception):
    """GitHub answered something other than a repo, a 404 or its rate limit."""


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------

@dataclass
class Material:
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    chapters: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)       # the uploader's own
    transcript: str = ""
    expanded: dict[str, str] = field(default_factory=dict)  # short link → where it lands

    def sections(self) -> list[tuple[str, str]]:
        """(where, text) in the order evidence is looked for."""
        return [("title", self.title), ("description", self.description), ("comments", "\n".join(self.comments)),
                ("tags", ", ".join(self.tags)), ("chapters", "\n".join(self.chapters)),
                ("transcript", self.transcript)]


def read_info(path: str | Path | None) -> dict:
    """The yt-dlp info.json at ``path``; {} when there is none."""
    if not path or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    return data if isinstance(data, dict) else {}


def info_path(asset: dict) -> Path | None:
    """The item's info.json: ``meta.info_json``, else the sidecar beside the file."""
    meta = asset.get("meta") or {}
    if meta.get("info_json") and Path(meta["info_json"]).is_file():
        return Path(meta["info_json"])
    p = Path(asset["path"])
    if p.with_suffix(".info.json").is_file():
        return p.with_suffix(".info.json")
    return next(iter(sorted(p.parent.glob(f"{p.stem}*.info.json"))), None) if p.parent.is_dir() else None


def clean_transcript(text: str) -> str:
    """A transcript file's text: the ``# title / # source / # backend`` header,
    WEBVTT and cue numbers and timing lines dropped; ``[m:ss]`` stamps kept."""
    out: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if (not out and s.startswith("#")) or s == "WEBVTT" or s.isdigit() or "-->" in s:
            continue
        out.append(line)
    return "\n".join(out).strip()


def material(*, title: str, info: dict, transcript: str = "") -> Material:
    """What the finder reads: the library title (the person may have renamed
    it — even to the repo's URL) and the downloader's own title, description,
    tags and chapters, the uploader's own comments, the transcript."""
    titles = [t for t in dict.fromkeys([(title or "").strip(), str(info.get("title") or "").strip()]) if t]
    who = {str(info.get(k)) for k in ("uploader_id", "channel_id", "uploader", "channel") if info.get(k)}
    comments = [str(c.get("text")) for c in info.get("comments") or []
                if isinstance(c, dict) and c.get("text") and ({str(c.get("author_id")), str(c.get("author"))} & who)]
    return Material(title="\n".join(titles), description=str(info.get("description") or ""),
                    tags=[str(t) for t in info.get("tags") or [] if t],
                    chapters=[str(c.get("title")) for c in info.get("chapters") or []
                              if isinstance(c, dict) and c.get("title")],
                    comments=comments[:20], transcript=clean_transcript(transcript))


# ---------------------------------------------------------------------------
# deterministic links
# ---------------------------------------------------------------------------

def _repo_name(repo: str) -> str:
    return re.sub(r"\.git$", "", repo.rstrip(".,;:)!?'\""), flags=re.I)


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def github_links(text: str) -> list[tuple[str, str, int]]:
    """(owner, repo, position) of every GitHub repo link in ``text``."""
    out = []
    for rx in (_GH_URL, _GH_RAW, _GH_PAGES):
        for m in rx.finditer(text or ""):
            owner, repo = m.group(1), _repo_name(m.group(2))
            if rx is _GH_URL and owner.lower() in _NOT_OWNERS:
                continue
            if repo and repo not in (".", ".."):
                out.append((owner, repo, m.start()))
    return out


def bare_slugs(text: str, *, whole: bool = False) -> list[tuple[str, str, int]]:
    """``owner/repo`` written without github.com: only the whole ``text``
    (``whole``, the title), a line of its own, or a line that says GitHub,
    repo or stars — anything else is too often a path or a fraction."""
    out = []
    for line_m in re.finditer(r"[^\n]+", text or ""):
        line = line_m.group(0)
        bare = re.sub(r"https?://\S+", " ", line)
        for m in _SLUG.finditer(bare):
            owner, repo = m.group(1), _repo_name(m.group(2))
            slug = f"{owner}/{repo}"
            if (slug.lower() in _NOT_SLUGS or owner.lower() in _NOT_OWNERS or not re.search(r"[A-Za-z]", owner)
                    or not re.search(r"[A-Za-z]", repo) or len(repo) < 2):
                continue
            alone = re.sub(r"[\W_]+", "", bare.replace(m.group(0), "")) == ""
            if whole or alone or _SLUG_HINT.search(line):
                out.append((owner, repo, line_m.start() + m.start()))
    return out


def _snippet(text: str, pos: int, span: int = 0, width: int = 70) -> str:
    """The words around ``pos`` on its line, ≤ about 2·width characters, stamps dropped."""
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos + span)
    line_end = len(text) if line_end < 0 else line_end
    start, end = max(line_start, pos - width), min(line_end, pos + span + width)
    s = " ".join(_TS.sub(" ", text[start:end]).split())
    return ("…" if start > line_start else "") + s + ("…" if end < line_end else "")


def _stamp_before(text: str, pos: int) -> int | None:
    """Seconds of the last ``[m:ss]`` (or ``[h:mm:ss]``) stamp at or before ``pos``; None without one."""
    last = None
    for m in _TS.finditer(text, 0, pos + 1):
        a, b, c = m.groups()
        last = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    return last


def deterministic(m: Material) -> list[dict]:
    """Repo candidates the material names outright, deduped, in order of
    appearance: ``{owner, repo, from, quote, at_s, how, sure: True}``."""
    out: dict[str, dict] = {}

    def add(owner: str, repo: str, where: str, text: str, pos: int, how: str) -> None:
        key = f"{owner}/{repo}".lower()
        if key not in out:
            out[key] = {"owner": owner, "repo": repo, "from": where,
                        "quote": _snippet(text, pos, len(owner) + len(repo) + 1),
                        "at_s": _stamp_before(text, pos) if where == "transcript" else None, "how": how, "sure": True}

    for where, text in m.sections():
        for owner, repo, pos in github_links(text):
            add(owner, repo, where, text, pos, "link")
        if where in ("title", "description", "comments"):
            for short, final in m.expanded.items():
                if short in text:
                    for owner, repo, _ in github_links(final):
                        add(owner, repo, where, text, text.index(short), "short link")
            for owner, repo, pos in bare_slugs(text, whole=where == "title"):
                add(owner, repo, where, text, pos, "slug")
    return list(out.values())


# ---------------------------------------------------------------------------
# the model: what the video names
# ---------------------------------------------------------------------------

def _header(m: Material) -> str:
    desc = m.description
    for short, final in m.expanded.items():
        desc = desc.replace(short, f"{short} (→ {final})")
    parts = [f"Title: {m.title or '(none)'}"]
    if desc.strip():
        parts.append(f"Description:\n{desc.strip()[:4000]}")
    if m.tags:
        parts.append(f"Tags: {', '.join(m.tags)[:600]}")
    if m.chapters:
        parts.append("Chapters:\n" + "\n".join(m.chapters)[:1500])
    if m.comments:
        parts.append("The uploader's comments:\n" + "\n".join(m.comments)[:1500])
    return "\n\n".join(parts)


def transcript_chunks(text: str) -> tuple[list[str], bool]:
    """The transcript in pieces of at most ``CHUNK`` characters, cut at line
    ends (a longer line is cut where it must); at most ``MAX_CHUNKS`` pieces
    (True: the rest was not read)."""
    lines: list[str] = []
    for line in (text or "").splitlines(keepends=True):
        lines += [line[i:i + CHUNK] for i in range(0, len(line), CHUNK)] or [line]
    chunks, cur = [], ""
    for line in lines:
        if cur and len(cur) + len(line) > CHUNK:
            chunks.append(cur)
            cur = ""
        cur += line
    if cur.strip():
        chunks.append(cur)
    return chunks[:MAX_CHUNKS], len(chunks) > MAX_CHUNKS


_KINDS = ("repo", "tool", "site")


def parse_mentions(text: str) -> list[dict]:
    """The model's answer as clean mentions; ``ValueError`` when it is not
    ``{"mentions": [...]}`` (the facade then retries or moves on)."""
    obj = parse_json_obj(text, require_any=("mentions",))
    if not isinstance(obj.get("mentions"), list):
        raise ValueError("mentions is not a list")
    out = []
    for x in obj["mentions"][:20]:
        if not isinstance(x, dict) or not str(x.get("name") or "").strip():
            continue
        owner, repo = str(x.get("owner") or "").strip(), _repo_name(str(x.get("repo") or "").strip())
        if not (re.fullmatch(_OWNER, owner) and re.fullmatch(_REPO, repo)):
            owner = repo = ""
        url = str(x.get("url") or "").strip()
        out.append({"name": str(x["name"]).strip()[:80], "kind": x.get("kind") if x.get("kind") in _KINDS else "tool",
                    "owner": owner, "repo": repo, "url": url if url.startswith("http") else "",
                    "quote": " ".join(str(x.get("quote") or "").split())[:300],
                    "about": " ".join(str(x.get("about") or "").split())[:120]})
    return out


def parse_pick(text: str, names: list[str]) -> str:
    """The model's pick, lowercased, from ``names`` (lowercased full names),
    or "" for none; ``ValueError`` for anything else."""
    obj = parse_json_obj(text, require_any=("pick",))
    pick = str(obj.get("pick") or "").strip().lower()
    if pick and pick not in names:
        raise ValueError(f"{pick!r} is not one of the candidates")
    return pick


def _quote_rx(quote: str) -> re.Pattern | None:
    words = re.findall(r"\w+", quote.lower())
    if len(words) < 3:
        return None
    return re.compile(r"(?:\W|\[\d+:\d\d(?::\d\d)?\])+".join(map(re.escape, words)), re.I)


def _name_rx(name: str) -> re.Pattern | None:
    sq = _squash(name)
    return re.compile(r"[\s._-]*".join(map(re.escape, sq)), re.I) if len(sq) >= 4 else None


def ground(m: Material, mention: dict) -> dict | None:
    """Where the material says it: ``{from, quote, at_s}`` — the material's
    own words (its quote, else the words around its name), or None when the
    material contains neither (the model made it up)."""
    for rx, whole_quote in ((_quote_rx(mention["quote"]), True), (_name_rx(mention["name"]), False)):
        if rx is None:
            continue
        for where, text in m.sections():
            hit = rx.search(text)
            if hit:
                quote = (" ".join(_TS.sub(" ", hit.group(0)).split()) if whole_quote
                         else _snippet(text, hit.start(), hit.end() - hit.start()))
                return {"from": where, "quote": quote[:300],
                        "at_s": _stamp_before(text, hit.start()) if where == "transcript" else None}
    return None


def _in_material(m: Material, url: str) -> bool:
    key = url.rstrip("/").lower()
    return (any(key in text.lower() for _, text in m.sections())
            or key in {v.rstrip("/").lower() for v in m.expanded.values()})


async def mentions(m: Material) -> tuple[list[dict], str, bool]:
    """(grounded mentions, served model, transcript cut short?). Raises
    ``llm.EngineError`` when no engine answers."""
    engines = await registry.active_snapshot()
    chunks, cut = transcript_chunks(m.transcript)
    header = _header(m)
    said: dict[str, dict] = {}
    model = ""
    for i, chunk in enumerate(chunks or [""]):
        body = header + (f"\n\nTranscript (part {i + 1} of {len(chunks)}):\n{chunk}" if chunk else "")
        res = await asyncio.to_thread(llm.complete, "media.investigate",
                                      [SystemMessage(content=SYSTEM), HumanMessage(content=body)],
                                      engines=engines, temperature=0.1, max_tokens=1400, validate=parse_mentions)
        model = res.records[-1].served_model if res.records else model
        for x in res.parsed:
            key = f"{x['owner']}/{x['repo']}".lower() if x["repo"] else _squash(x["name"])
            if key and key not in said:
                said[key] = x
    out = []
    for x in said.values():
        where = ground(m, x)
        if where is None:
            logger.info("investigate: dropped %r, which the material does not mention", x["name"])
            continue
        if x["url"] and not _in_material(m, x["url"]):
            x["url"] = ""
        out.append({**x, **where})
    return out, model, cut


# ---------------------------------------------------------------------------
# GitHub (sync httpx: call through asyncio.to_thread)
# ---------------------------------------------------------------------------

def _headers(accept: str = "application/vnd.github+json") -> dict:
    h = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "argus-research-bot"}
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _raise_limit(r: httpx.Response) -> None:
    if r.status_code == 429 or (r.status_code == 403 and (
            r.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in r.text.lower())):
        raise GitHubLimit(int(r.headers.get("x-ratelimit-reset") or 0))


def gh_repo(owner: str, repo: str) -> dict | None:
    """The public repo ``owner/repo`` (renames followed), or None for 404 /
    451 / private. ``GitHubLimit`` on the rate limit; ``GitHubError`` on any
    other answer or a network failure."""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as c:
            r = c.get(f"{_API}/repos/{owner}/{repo}", headers=_headers())
    except httpx.HTTPError as e:
        raise GitHubError(f"{type(e).__name__}: {e}") from e
    _raise_limit(r)
    if r.status_code in (404, 451):
        return None
    if r.status_code != 200:
        raise GitHubError(f"GitHub answered HTTP {r.status_code} for {owner}/{repo}")
    d = r.json()
    if d.get("private") or not d.get("full_name"):
        return None
    return {"name": d["full_name"], "url": d.get("html_url") or f"https://github.com/{d['full_name']}",
            "about": " ".join(str(d.get("description") or "").split())[:300],
            "stars": int(d.get("stargazers_count") or 0), "archived": bool(d.get("archived")),
            "fork": bool(d.get("fork"))}


def gh_search(name: str) -> list[dict]:
    """Repos whose name contains ``name``, most stars first (≤ 10):
    ``[{full_name, name, stars, about}]``. Raises like ``gh_repo``."""
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(f"{_API}/search/repositories", headers=_headers(),
                      params={"q": f"{name} in:name", "sort": "stars", "order": "desc", "per_page": 10})
    except httpx.HTTPError as e:
        raise GitHubError(f"{type(e).__name__}: {e}") from e
    _raise_limit(r)
    if r.status_code != 200:
        raise GitHubError(f"GitHub search answered HTTP {r.status_code}")
    return [{"full_name": it["full_name"], "name": it.get("name") or "", "stars": int(it.get("stargazers_count") or 0),
             "about": " ".join((it.get("description") or "").split())[:200]}
            for it in r.json().get("items") or [] if it.get("full_name") and not it.get("private")]


def gh_readme(owner: str, repo: str) -> str:
    """The repo's README as text ("" when it has none). Raises like ``gh_repo``."""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as c:
            r = c.get(f"{_API}/repos/{owner}/{repo}/readme", headers=_headers("application/vnd.github.raw+json"))
    except httpx.HTTPError as e:
        raise GitHubError(f"{type(e).__name__}: {e}") from e
    _raise_limit(r)
    if r.status_code == 404:
        return ""
    if r.status_code != 200:
        raise GitHubError(f"GitHub answered HTTP {r.status_code} for the README of {owner}/{repo}")
    return r.text[:200_000]


def readme_lead(md: str, limit: int = 300) -> str:
    """The README's first paragraph of prose: headings, badges, images, HTML,
    tables, quotes, code and link-only lines skipped; links kept as their text."""
    para: list[str] = []
    fence = False
    for raw in (md or "").splitlines():
        line = raw.strip()
        if line.startswith(("```", "~~~")):
            fence = not fence
            continue
        bare = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\)|<[^>]+>", "", line).strip()
        skip = (fence or not bare or bare.startswith(("#", "|", ">", "---", "===", "- [", "* ["))
                or re.fullmatch(r"(?:\[[^\]]*\]\([^)]*\)[\s|·•-]*)+", bare) is not None)
        if skip:
            if para:
                break
            continue
        para.append(re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", bare))
    text = re.sub(r"(\*\*|__|\*|_|`)(.+?)\1", r"\2", " ".join(para))
    text = " ".join(unescape(text).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def about_from_github(owner: str, repo: str) -> str:
    """The repo's own short description: its GitHub description, else its
    README's first paragraph. ``ValueError`` when it has neither or is gone;
    ``GitHubLimit`` / ``GitHubError`` as ``gh_repo``."""
    r = gh_repo(owner, repo)
    if r is None:
        raise ValueError(f"{owner}/{repo} is not a public repo on GitHub (any more)")
    if r["about"]:
        return r["about"]
    lead = readme_lead(gh_readme(*r["name"].split("/", 1)))
    if not lead:
        raise ValueError(f"{r['name']} has no description and no README text")
    return lead


# ---------------------------------------------------------------------------
# sites
# ---------------------------------------------------------------------------

def open_page(url: str) -> tuple[str, str]:
    """(final URL, page title) of a GET through the SSRF guard (redirects
    followed). ``ValueError`` saying why when the page does not load."""
    client = fetchers._http_client(_FETCH)
    try:
        final, body, ctype = fetchers._get(client, url, _FETCH)
    except (fetchers._RungError, fetchers._Refused) as e:
        raise ValueError(str(e)) from e
    finally:
        client.close()
    title = ""
    if "html" in (ctype or "").lower() or body[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
        hit = re.search(rb"<title[^>]*>(.*?)</title>", body[:300_000], re.I | re.S)
        if hit:
            title = " ".join(unescape(hit.group(1).decode("utf-8", "replace")).split())[:160]
    return final, title


def expand_short(m: Material) -> list[str]:
    """Follow the short links in the title, description and comments into
    ``m.expanded``; the ones that did not load, each with why."""
    seen: list[str] = []
    for text in (m.title, m.description, "\n".join(m.comments)):
        for url in extract_urls(text):
            if domain(url) in SHORT_HOSTS and url not in seen:
                seen.append(url)
    errors = []
    for url in seen[:MAX_SHORT]:
        try:
            m.expanded[url] = open_page(url)[0]
        except ValueError as e:
            errors.append(f"{url}: {e}")
    return errors


def _site_ok(url: str) -> bool:
    reg = registrable_domain(url)
    return bool(reg) and reg not in _NOT_SITES and domain(url) not in _NOT_SITES


# ---------------------------------------------------------------------------
# the investigation
# ---------------------------------------------------------------------------

@dataclass
class _Run:
    checks: int = 0
    searches: int = 0
    web: int = 0
    sites: int = 0
    stopped: str = ""                    # why the GitHub checks stopped ("" = they did not)
    unchecked: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def stop(self, e: Exception) -> None:
        if isinstance(e, GitHubLimit):
            when = f", it resets at {datetime.fromtimestamp(e.reset).strftime('%H:%M')}" if e.reset else ""
            self.stopped = f"GitHub's rate limit{when} (a GITHUB_TOKEN in .env lifts it)"
        else:
            self.stopped = f"GitHub failed: {e}"
        logger.warning("investigate: GitHub checks stopped: %s", self.stopped)


async def _check(run: _Run, owner: str, repo: str) -> dict | None:
    """``gh_repo`` within the run's budget; None when gone, not checked or stopped."""
    label = f"{owner}/{repo}"
    if run.stopped or run.checks >= MAX_CHECKS:
        run.unchecked.append(label)
        return None
    run.checks += 1
    try:
        r = await asyncio.to_thread(gh_repo, owner, repo)
    except (GitHubLimit, GitHubError) as e:
        run.stop(e)
        run.unchecked.append(label)
        return None
    if r is None:
        run.not_found.append(label)
    return r


async def _pick(x: dict, cands: list[dict]) -> tuple[str, str] | None:
    """The candidate (``{full_name, stars, about}``) the model says IS the
    mention ``x``, or None: a repo that only shares the name is not it.
    Raises ``llm.EngineError`` when no engine answers."""
    names = {c["full_name"].lower(): c["full_name"] for c in cands}
    lines = []
    for c in cands:
        stars = f" (★{c['stars']})" if c.get("stars") is not None else ""
        lines.append(f"- {c['full_name']}{stars}: {c.get('about') or 'no description'}")
    body = (f"Name: {x['name']}\nThe video's words: {x['quote']}\nWhat the video says it is: {x['about'] or '-'}"
            "\n\nCandidates:\n" + "\n".join(lines))
    res = await asyncio.to_thread(llm.complete, "media.investigate",
                                  [SystemMessage(content=PICK), HumanMessage(content=body)],
                                  engines=await registry.active_snapshot(), temperature=0.1, max_tokens=200,
                                  validate=lambda t: parse_pick(t, list(names)))
    pick = names.get(res.parsed)
    return tuple(pick.split("/", 1)) if pick else None


async def _search_repo(run: _Run, x: dict) -> tuple[str, str] | None:
    """(owner, repo) of the repo the named mention ``x`` is: the model picks
    among GitHub's repos whose name contains it, then among the repos a web
    search links; None when neither has it."""
    want = _squash(x["name"])
    if len(want) < 3:
        return None
    seen: set[str] = set()
    if not run.stopped and run.searches < MAX_SEARCHES:
        run.searches += 1
        try:
            items = await asyncio.to_thread(gh_search, x["name"])
        except (GitHubLimit, GitHubError) as e:
            run.stop(e)
            items = []
        cands = [it for it in items if want in _squash(it["name"])][:6]
        seen = {c["full_name"].lower() for c in cands}
        if cands and (hit := await _pick(x, cands)):
            return hit
    if run.web < MAX_WEB:
        run.web += 1
        hits, err = await asyncio.to_thread(ddgs_hits, f"{x['name']} github", max_results=8)
        if err:
            run.errors.append(err)
        cands = []
        for h in hits:
            if want not in _squash(h["url"]):           # the repo, or a page in it, carries the name
                continue
            for o, r, _ in github_links(h["url"]):
                if f"{o}/{r}".lower() not in seen:
                    seen.add(f"{o}/{r}".lower())
                    cands.append({"full_name": f"{o}/{r}", "stars": None, "about": h.get("snippet") or ""})
        if cands:
            return await _pick(x, cands[:6])
    return None


async def _site(run: _Run, x: dict) -> dict | None:
    """A tool or site mention as a page that loads, or None (counted)."""
    urls = [x["url"]] if x["url"] and _site_ok(x["url"]) else []
    want = _squash(x["name"])
    if not urls and len(want) >= 3 and run.web < MAX_WEB:
        run.web += 1
        hits, err = await asyncio.to_thread(ddgs_hits, f"{x['name']} official site", max_results=6)
        if err:
            run.errors.append(err)
        for h in hits:
            label = _squash(registrable_domain(h["url"]).split(".")[0])
            if _site_ok(h["url"]) and label and (want in label or (len(label) >= 4 and label in want)):
                urls = [h["url"]]
                break
    if not urls or run.sites >= MAX_SITES:
        run.unresolved.append(x["name"])
        return None
    run.sites += 1
    try:
        final, title = await asyncio.to_thread(open_page, urls[0])
    except ValueError as e:
        run.unresolved.append(f"{x['name']} ({e})")
        return None
    return {"kind": "tool", "key": canonical_url(final), "name": x["name"], "url": final, "about": title,
            "stars": None, "archived": False, "fork": False, "renamed_from": None}


def _repo_item(r: dict, asked: str) -> dict:
    return {"kind": "repo", "key": r["name"].lower(), "name": r["name"], "url": r["url"], "about": r["about"],
            "stars": r["stars"], "archived": r["archived"], "fork": r["fork"],
            "renamed_from": asked if asked.lower() != r["name"].lower() else None}


async def find(m: Material, *, stage=None) -> dict:
    """Everything checked that the material mentions:
    ``{found, skipped, model, read}``. ``found``: repos first (sure ones
    first), then tools and sites, each ``{kind, key, name, url, about, said,
    stars, archived, fork, renamed_from, from, quote, at_s, how, sure, on}``.
    ``stage(text)`` (async), when given, hears each step. Raises
    ``llm.EngineError`` when no engine answers."""
    async def say(text: str) -> None:
        if stage is not None:
            await stage(text)

    run = _Run()
    await say("following short links" if any(domain(u) in SHORT_HOSTS for u in extract_urls(m.description)) else
              "reading the description and transcript")
    run.errors += await asyncio.to_thread(expand_short, m)
    sure = deterministic(m)
    await say("asking the model what the video names")
    said, model, cut = await mentions(m)
    await say("checking the candidates on GitHub and the web")
    by_repo = {f"{c['owner']}/{c['repo']}".lower(): c for c in sure}
    named: list[dict] = []
    for x in said:
        key = f"{x['owner']}/{x['repo']}".lower() if x["repo"] else ""
        twin = by_repo.get(key) if key else next((c for c in sure if _squash(c["repo"]) == _squash(x["name"])), None)
        if twin is not None:
            twin.setdefault("said", x["about"])
            continue
        links = github_links(x["url"]) if x["url"] else []
        if links:                           # a link in the material (kept only when it is there)
            x, how, sure = {**x, "owner": links[0][0], "repo": links[0][1]}, "link", True
        elif x["repo"]:                     # sure only when the words say both: "langflow AI slash langflow"
            spoken = _squash(x["quote"])
            how, sure = "said", _squash(x["owner"]) in spoken and _squash(x["repo"]) in spoken
        if x["repo"]:
            by_repo[f"{x['owner']}/{x['repo']}".lower()] = {
                "owner": x["owner"], "repo": x["repo"], "from": x["from"], "quote": x["quote"], "at_s": x["at_s"],
                "how": how, "sure": sure, "said": x["about"]}
        else:
            named.append(x)

    found: dict[str, dict] = {}

    def keep(item: dict, c: dict) -> None:
        if item["key"] not in found:
            found[item["key"]] = {**item, "said": c.get("said") or "", "from": c["from"], "quote": c["quote"],
                                  "at_s": c.get("at_s"), "how": c["how"], "sure": c["sure"], "on": c["sure"]}

    for c in by_repo.values():
        r = await _check(run, c["owner"], c["repo"])
        if r is not None:
            keep(_repo_item(r, f"{c['owner']}/{c['repo']}"), c)
    for x in named:
        c = {"from": x["from"], "quote": x["quote"], "at_s": x["at_s"], "said": x["about"], "sure": False}
        if x["kind"] == "repo":
            hit = await _search_repo(run, x)
            if hit is None:
                run.unresolved.append(x["name"])
                continue
            r = await _check(run, *hit)
            if r is not None:
                keep(_repo_item(r, "/".join(hit)), {**c, "how": "search"})
        else:
            item = await _site(run, x)
            if item is not None:
                keep(item, {**c, "how": "link" if x["url"] else "search"})
    items = sorted(found.values(), key=lambda i: (i["kind"] != "repo", not i["sure"]))
    return {"found": items[:MAX_FOUND], "model": model,
            "skipped": {"not_found": run.not_found, "unresolved": list(dict.fromkeys(run.unresolved)),
                        "unchecked": run.unchecked, "stopped": run.stopped, "errors": run.errors[:5],
                        "more": max(0, len(items) - MAX_FOUND)},
            "read": {"transcript": bool(m.transcript), "cut": cut, "short_links": len(m.expanded)}}
