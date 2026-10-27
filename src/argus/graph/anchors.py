"""Anchors — the links the user put in the question, read before the brief.

A question about "this project https://github.com/o/r" is about that page, but
the brief never sees URLs and the scout only searches, so without this step the
linked page is never read (run a255fdca: 60 sources, none of them the repo).

``read_anchors`` finds the http(s) links in the request (a regex: the user typed
them, so the brief's no-invented-URLs rule still holds) and reads each one:

- the link itself, through the fetch chain (its private-address guards
  included; a GitHub repo's README via the ``native`` rung), plus a repo's API
  metadata and top-level files;
- when the link is a ROOT (a repo, or a site's landing page) and the request
  does not say to stay on it ("only the README", "just this page"), up to
  ``EXTRA_MAX`` representative parts, ranked by how well they fit the request:
  a repo's latest releases, CHANGELOG, ``docs/`` files and homepage; a site's
  own links to docs, pricing, features, changelog… A deep link is read alone —
  the user pointed at that page.

It returns ``anchors`` (what the plan gate shows and the brief reads: ``{url,
title, kind, ok, error, chars, about, context, pages}``, ``context`` a short
excerpt, ``pages`` the extra parts read) and ``sources``: one pinned source per
page read, carrying its text, so research digests it without fetching again
(the ``exa_text`` rung) and never charges it to a budget. Links already read
(a plan-edit re-draft) are not read again.
"""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from ..engines import fetchers
from ..engines.ledger import SpendLedger
from ..engines.urls import canonical_url
from .search_providers import exa_enabled

logger = logging.getLogger("argus.anchors")

ANCHOR_MAX = 3                     # links read per question
EXTRA_MAX = 3                      # representative parts read per root link
CONTEXT_CHARS = 1600               # the link's own excerpt in the brief prompt
PAGE_CONTEXT_CHARS = 450           # each extra part's excerpt in the brief prompt
TEXT_CHARS = 60_000                # a pinned source's text (research chunks it)
MIN_CHARS = 400                    # less is a JS shell or a stub ("What do you want to know?"), not read
SOURCE = "anchor"

_LINK = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
# ponytail: a link that really ends in ")" (some wiki pages) loses it; balance parentheses if that bites
_TRAILING = ".,;:!?)]}>'\""
# the request keeps the reading on the link itself
_STAY = re.compile(r"\b(?:only|just)\b[^.?!\n]{0,40}\b(?:readme|this (?:page|link|repo)|that (?:page|link)"
                   r"|the (?:page|link)|landing page)\b", re.IGNORECASE)
_RELEASE_WORDS = frozenset({"release", "releases", "version", "versions", "changelog", "roadmap", "latest",
                            "update", "updates", "recent"})
_STOP = frozenset("""this that with from what which their there about into your have will would could should
research report present project potential opportunities using based these those them they when where while
also more most than then very just only like make made does doing done been being http https www com""".split())
_DOC_NAMES = frozenset({"index", "readme", "overview", "intro", "introduction", "getting-started", "quickstart",
                        "architecture", "usage", "guide", "how-it-works", "help", "faq"})
_SKIP_PATH = re.compile(r"(?:^|/)(?:log-?in|sign-?in|sign-?up|register|account|privacy|terms|legal|cookies?"
                        r"|careers|jobs|cart|checkout|status|press|security|wp-admin)(?:/|$|\.)", re.IGNORECASE)
_SKIP_EXT = re.compile(r"\.(?:png|jpe?g|gif|svg|webp|ico|zip|gz|tar|exe|dmg|mp4|mp3|woff2?|css|js)$", re.IGNORECASE)
_RELEASE_KIND = re.compile(r"changelog|releases?|blog|news|updates?|roadmap")
_SITE_KINDS = (("docs", re.compile(r"docs?|documentation|guide|manual|learn|tutorial|reference|api"), 1.5),
               ("pricing", re.compile(r"pricing|plans?|billing"), 1.3),
               ("features", re.compile(r"features?|product|solutions?|how-it-works|overview"), 1.2),
               ("about", re.compile(r"about|company|team"), 1.0),
               ("updates", _RELEASE_KIND, 0.8))


def extract_links(text: str) -> list[str]:
    """The http(s) links in ``text``, trailing punctuation dropped, deduplicated
    by canonical URL, in order, at most ``ANCHOR_MAX``."""
    out, seen = [], set()
    for m in _LINK.finditer(text or ""):
        url = m.group(0).rstrip(_TRAILING)
        try:
            key = canonical_url(url)
        except ValueError:
            continue
        if key and key not in seen:
            seen.add(key)
            out.append(url)
    return out[:ANCHOR_MAX]


def query_words(request: str) -> frozenset[str]:
    """The request's content words (4+ letters, no stop words, no link parts)."""
    return frozenset(w for w in re.findall(r"[a-z][a-z0-9]{3,}", _LINK.sub(" ", request or "").lower())
                     if w not in _STOP)


def _overlap(name: str, words: frozenset[str]) -> int:
    """How many request words a file or page name shares (a shared 5-letter stem counts)."""
    parts = {p for p in re.split(r"[^a-z0-9]+", name.lower()) if len(p) >= 4}
    return sum(1 for w in words if any(p[:5] == w[:5] for p in parts))


def _raw(url: str, cfg) -> str:
    """A text file (a raw README or doc) through the fetch chain's guarded GET."""
    with fetchers._http_client(cfg) as client:
        _final, body, _ctype = fetchers._get(client, url, cfg)
    return body.decode("utf-8", errors="replace")


def _github_api(path: str):
    """GET ``https://api.github.com/<path>`` → the JSON, or ``None`` on a non-200 (keyless)."""
    with httpx.Client(timeout=12.0, follow_redirects=True,
                      headers={"Accept": "application/vnd.github+json"}) as c:
        r = c.get(f"https://api.github.com/{path}")
    return r.json() if r.status_code == 200 else None


def _github_meta(owner: str, repo: str) -> tuple[str, str, dict, list[str], str]:
    """A repo's API facts → ``(header text, one-line about, meta, top-level names, error)``."""
    try:
        meta = _github_api(f"repos/{owner}/{repo}")
        if meta is None:
            return "", "", {}, [], "GitHub API: the repository did not answer 200"
        listing = _github_api(f"repos/{owner}/{repo}/contents")
    except (httpx.HTTPError, ValueError) as e:
        return "", "", {}, [], f"GitHub API: {type(e).__name__}: {e}"
    names = ([f"{x['name']}{'/' if x.get('type') == 'dir' else ''}" for x in listing if isinstance(x, dict)]
             if isinstance(listing, list) else [])
    about = (meta.get("description") or "").strip()
    facts = [f"Language: {meta['language']}" if meta.get("language") else "",
             f"Stars: {meta.get('stargazers_count', 0)}", f"Forks: {meta.get('forks_count', 0)}",
             f"License: {(meta.get('license') or {}).get('spdx_id')}" if meta.get("license") else "",
             f"Created: {(meta.get('created_at') or '')[:10]}", f"Last push: {(meta.get('pushed_at') or '')[:10]}",
             f"Homepage: {meta['homepage']}" if meta.get("homepage") else ""]
    lines = [f"GitHub repository {meta.get('full_name') or f'{owner}/{repo}'} — {about or 'no description'}",
             " · ".join(f for f in facts if f)]
    if meta.get("topics"):
        lines.append("Topics: " + ", ".join(meta["topics"]))
    if names:
        lines.append("Top-level files: " + ", ".join(names[:60]))
    return "\n".join(lines), about, meta, names, ""


# ---------------------------------------------------------------------------
# representative parts: candidates are (score, url, title, why, reader | None)
# ---------------------------------------------------------------------------

def _repo_candidates(owner: str, repo: str, meta: dict, names: list[str], words: frozenset[str], cfg) -> list:
    """A repo's representative parts, scored against the request's words."""
    release_boost = 2.0 if words & _RELEASE_WORDS else 0.0
    base = f"https://github.com/{owner}/{repo}"
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD"

    def releases() -> str:
        items = _github_api(f"repos/{owner}/{repo}/releases?per_page=5") or []
        lines = [f"Releases of {owner}/{repo} (newest first):"]
        for r in items:
            body = " ".join((r.get("body") or "").split())[:600]
            lines.append(f"\n## {r.get('tag_name')} — {r.get('name') or ''} ({(r.get('published_at') or '')[:10]})"
                         f"\n{body}")
        if len(lines) == 1:
            raise fetchers._RungError("no releases published")
        return "\n".join(lines)

    out = [(1.0 + release_boost, f"{base}/releases", "Releases", "latest releases", releases)]
    lower = {n.lower(): n for n in names}
    for log in ("changelog.md", "changes.md", "history.md"):
        if log in lower:
            name = lower[log]
            out.append((0.9 + release_boost, f"{base}/blob/HEAD/{name}", name, "changelog",
                        lambda name=name: _raw(f"{raw}/{name}", cfg)))
            break
    for folder in ("docs/", "doc/", "documentation/"):
        if folder not in lower:
            continue
        listing = _github_api(f"repos/{owner}/{repo}/contents/{lower[folder].rstrip('/')}")
        for x in listing if isinstance(listing, list) else []:
            if not isinstance(x, dict) or x.get("type") != "file" or not re.search(
                    r"\.(?:md|mdx|rst|txt)$", x.get("name", ""), re.IGNORECASE):
                continue
            stem = x["name"].rsplit(".", 1)[0].lower()
            score = 1.0 + (0.5 if stem in _DOC_NAMES else 0.0) + 2.0 * _overlap(stem, words)
            out.append((score, f"{base}/blob/HEAD/{x['path']}", x["path"], "documentation",
                        lambda path=x["path"]: _raw(f"{raw}/{path}", cfg)))
        break
    home = (meta.get("homepage") or "").strip()
    if home.startswith(("http://", "https://")):
        out.append((1.2, home, urlsplit(home).netloc, "homepage", None))
    return out


class _Links(HTMLParser):
    """Every ``<a href>`` of a page with its text."""

    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href, self._text = dict(attrs).get("href"), []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a":
            if self._href:
                self.links.append((self._href, " ".join("".join(self._text).split())[:80]))
            self._href = None


def _site_candidates(url: str, words: frozenset[str], cfg) -> list:
    """A landing page's own links to representative pages, scored against the request's words."""
    with fetchers._http_client(cfg) as client:
        final, body, ctype = fetchers._get(client, url, cfg)
    if "html" not in ctype.lower():
        return []
    parser = _Links()
    parser.feed(body.decode("utf-8", errors="replace"))
    host = (urlsplit(final).hostname or "").removeprefix("www.")
    release_boost = 2.0 if words & _RELEASE_WORDS else 0.0
    seen = {canonical_url(final), canonical_url(url)}
    out = []
    for href, text in parser.links:
        try:
            parts = urlsplit(urljoin(final, href))
            link = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
            key = canonical_url(link)
        except ValueError:
            continue
        if (parts.scheme not in ("http", "https") or (parts.hostname or "").removeprefix("www.") != host
                or key in seen or _SKIP_PATH.search(parts.path) or _SKIP_EXT.search(parts.path)):
            continue
        seen.add(key)
        kind = next(((name, pat, w) for name, pat, w in _SITE_KINDS if pat.search(parts.path.lower())), None)
        score = (kind[2] if kind else 0.0) + 2.0 * _overlap(f"{parts.path} {text}", words)
        if kind and kind[1] is _RELEASE_KIND:
            score += release_boost
        if score >= 1.0:
            out.append((score, link, text or parts.path, kind[0] if kind else "matches the question", None))
    return out


def _read_page(url: str, reader, *, cfg, ledger: SpendLedger) -> tuple[str, str, str]:
    """One page → ``(text, title, error)``: ``reader()`` when given, else the fetch chain."""
    if reader is not None:
        try:
            return reader(), "", ""
        except (fetchers._RungError, fetchers._Refused, httpx.HTTPError, ValueError) as e:
            return "", "", str(e)
    res = fetchers.fetch({"url": url, "title": ""}, cfg=cfg, ledger=ledger, exa_enabled=exa_enabled(), pinned=True)
    if not res.ok:
        return "", "", res.error
    with open(res.markdown_path, encoding="utf-8", errors="replace") as f:
        return f.read(), res.title or "", ""


def _source(url: str, title: str, kind: str, text: str, about: str, of: str) -> dict:
    return {"url": url, "title": title, "kind": kind, "text": text[:TEXT_CHARS], "snippet": about,
            "summary": about, "provider": "user", "source": SOURCE, "pinned": True, "sub_qs": [], "anchor_of": of}


def read_anchor(url: str, *, rc, ledger: SpendLedger, words: frozenset[str] = frozenset(),
                explore: bool = True) -> tuple[dict, list[dict], list[str]]:
    """Read one link (and, at a root, its representative parts) → ``(anchor, pinned sources, errors)``."""
    errors: list[str] = []
    repo = fetchers._github_repo(url)
    header, about, meta, names = "", "", {}, []
    if repo:
        header, about, meta, names, err = _github_meta(*repo)
        if err:
            errors.append(f"anchors: {url}: {err}")
    body, page_title, err = _read_page(url, None, cfg=rc.fetch, ledger=ledger)
    if err:
        errors.append(f"anchors: {url}: not read ({err})")
    text = "\n\n".join(p for p in (header, body.strip()) if p)[:TEXT_CHARS]
    title = (f"{repo[0]}/{repo[1]}" if repo else page_title.strip()) or url.split("//", 1)[-1][:70]
    if not about:
        about = next((ln.strip("# ").strip() for ln in body.splitlines()
                      if len(ln.strip("# ").strip()) > 30), "")[:200]
    ok = len(text.strip()) >= MIN_CHARS
    kind = "repo" if repo else "web"
    anchor = {"url": url, "title": title, "kind": kind, "ok": ok,
              "error": "" if ok else (err or f"only {len(text.strip())} chars of text (a script-built page?)"),
              "chars": len(text),
              "about": about, "context": text[:CONTEXT_CHARS] if ok else "", "pages": []}
    sources = [_source(url, title, kind, text, about, url)] if ok else []

    if explore and (repo or not urlsplit(url).path.strip("/")):
        try:
            candidates = (_repo_candidates(*repo, meta, names, words, rc.fetch) if repo
                          else _site_candidates(url, words, rc.fetch))
        except (fetchers._RungError, fetchers._Refused, httpx.HTTPError, ValueError) as e:
            candidates = []
            errors.append(f"anchors: {url}: no representative parts ({e})")
        for _score, purl, ptitle, why, reader in sorted(candidates, key=lambda c: -c[0])[:EXTRA_MAX]:
            ptext, fetched_title, perr = _read_page(purl, reader, cfg=rc.fetch, ledger=ledger)
            pok = len(ptext.strip()) >= MIN_CHARS
            page = {"url": purl, "title": ptitle or fetched_title or purl, "why": why, "ok": pok,
                    "chars": len(ptext), "error": "" if pok else (perr or "too little text"),
                    "context": ptext.strip()[:PAGE_CONTEXT_CHARS] if pok else ""}
            anchor["pages"].append(page)
            if pok:
                sources.append(_source(purl, f"{title} — {page['title']}", kind, ptext, why, url))
            else:
                errors.append(f"anchors: {purl}: not read ({page['error']})")
    return anchor, sources, errors


def read_anchors(request: str, *, rc, ledger: SpendLedger, known: list[dict] | None = None
                 ) -> tuple[list[dict], list[dict], list[str]]:
    """The links in ``request`` → ``(anchors, new pinned sources, errors)``;
    an anchor in ``known`` (read by an earlier draft) is kept as it was."""
    by_url = {canonical_url(a["url"]): a for a in known or []}
    words, explore = query_words(request), not _STAY.search(request or "")
    anchors, sources, errors = [], [], []
    for url in extract_links(request):
        prev = by_url.get(canonical_url(url))
        if prev is not None:
            anchors.append(prev)
            continue
        anchor, new, errs = read_anchor(url, rc=rc, ledger=ledger, words=words, explore=explore)
        anchors.append(anchor)
        sources += new
        errors += errs
        logger.info("anchor %s: %s (%d chars, %d extra part(s) read)", url, "read" if anchor["ok"] else "not read",
                    anchor["chars"], sum(1 for p in anchor["pages"] if p["ok"]))
    return anchors, sources, errors


def brief_context(anchors: list[dict]) -> str:
    """The prompt block the brief reads: each anchor that was read, excerpted, with its extra parts."""
    parts = []
    read = [a for a in anchors if a.get("ok") or any(p["ok"] for p in a.get("pages") or [])]
    for i, a in enumerate(read, 1):
        block = [f"[{i}] {a['title']} ({a['kind']})", a.get("context") or ""]
        block += [f"--- {p['title']} ({p['why']}) ---\n{p['context']}" for p in a.get("pages") or [] if p["ok"]]
        parts.append("\n".join(b for b in block if b))
    if not parts:
        return ""
    return ("CONTEXT FROM THE USER'S LINKS (already read; the request is about or refers to these):\n"
            + "\n\n".join(parts))


def plan_view(anchors: list[dict]) -> list[dict]:
    """The anchors as the plan gate shows them (no excerpts)."""
    keep = ("url", "title", "kind", "ok", "error", "chars", "about")
    return [{**{k: a.get(k) for k in keep},
             "pages": [{k: p.get(k) for k in ("url", "title", "why", "ok", "chars", "error")}
                       for p in a.get("pages") or []]} for a in anchors]


__all__ = ["extract_links", "query_words", "read_anchor", "read_anchors", "brief_context", "plan_view",
           "ANCHOR_MAX", "EXTRA_MAX", "SOURCE"]
