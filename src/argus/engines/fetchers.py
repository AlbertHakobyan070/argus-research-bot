"""The fetch chain (v4 spec §2.8): one source → markdown on disk.

Rungs, in ``FetchConfig.chain`` order, first success wins:

- ``exa_text``     the search hit already carries Exa's page text — no request;
- ``native``       httpx + trafilatura / pypdf, by URL class: arXiv → ar5iv HTML,
                   else the PDF; a GitHub repo → its raw README; ``.pdf`` → pypdf;
                   anything else → ``trafilatura.extract(html, include_links=False,
                   favor_recall=True)``;
- ``exa_contents`` ``POST https://api.exa.ai/contents`` (pages counted in the
                   ledger, ``costDollars`` added to it);
- ``intel_*``      the intel-stack scripts (snatch / crawl / stealth) — in the
                   chain only when both ``INTEL_*`` paths exist (checked at import).

Guards: before every network rung, and on every redirect hop of ``native``,
a URL whose host resolves to a non-global address (loopback, RFC 1918,
link-local, …) or is this server's own endpoint (the proxy, the console's
public URL) is refused — ``error = "refused: private address (…)"`` and the
chain stops. Every download stops at ``FetchConfig.max_bytes``; text on disk
is capped at ``max_chars_on_disk``. A source no rung can fetch comes back
with ``ok=False``, every rung's error and ``rungs_tried``.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
import tempfile
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

import httpx
import pypdf
import trafilatura

from .. import tools
from ..tools import crawl_url, snatch_url
from .urls import canonical_url

if TYPE_CHECKING:
    from ..graph.run_config import FetchConfig
    from .ledger import SpendLedger

INTEL_RUNGS = frozenset({"intel_snatch", "intel_crawl", "intel_stealth"})
_MIN_CHARS = 200            # less text than this is a failed rung (the digest skips it anyway)
_EXA_TEXT_MIN = 300
_MAX_REDIRECTS = 5
_EXA_CONTENTS = "https://api.exa.ai/contents"
_UA = "Mozilla/5.0 (compatible; Argus research)"
_REDIRECTS = (301, 302, 303, 307, 308)


def intel_available(stack_dir: Path, python_bin: Path) -> bool:
    """The intel-stack rungs need both its scripts folder and its interpreter."""
    return Path(stack_dir).is_dir() and Path(python_bin).is_file()


INTEL_AVAILABLE = intel_available(tools.INTEL_STACK_DIR, tools.INTEL_PYTHON_BIN)


def available_chain(chain: list[str]) -> list[str]:
    return [r for r in chain if r not in INTEL_RUNGS or INTEL_AVAILABLE]


@dataclass
class FetchResult:
    ok: bool
    markdown_path: str | None = None
    title: str = ""
    chars: int = 0
    fetcher: str = ""
    ms: int = 0
    error: str = ""
    rungs_tried: list[str] = field(default_factory=list)


class _Refused(Exception):
    """The SSRF guard refused the target: no rung may fetch it."""


class _RungError(Exception):
    """This rung failed; the chain moves on."""


# ---------------------------------------------------------------------------
# the SSRF guard
# ---------------------------------------------------------------------------

def _resolve(host: str) -> list[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})


def _port(parts) -> int:
    return parts.port or (443 if parts.scheme == "https" else 80)


def _own_endpoints() -> set[tuple[str, int]]:
    """This server's own endpoints outside loopback: the proxy (``FREELLMAPI_BASE_URL``)
    and the console's public URL. Their loopback defaults are refused as private anyway."""
    out = set()
    for env in ("FREELLMAPI_BASE_URL", "ARGUS_WEBAPP_PUBLIC_URL"):
        parts = urlsplit(os.environ.get(env, "").strip())
        if parts.hostname:
            out.add((parts.hostname.lower(), _port(parts)))
    return out


def check_target(url: str) -> None:
    """Raise ``_Refused`` when the URL's host resolves to a non-global address
    or is one of this server's own endpoints."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host or parts.scheme not in ("http", "https"):
        raise _RungError(f"not an http(s) URL: {url!r}")
    if (host, _port(parts)) in _own_endpoints():
        raise _Refused(f"refused: private address (own endpoint {host}:{_port(parts)})")
    try:
        addrs = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            addrs = _resolve(host)
        except socket.gaierror as e:
            raise _RungError(f"dns: {host} did not resolve ({e})") from e
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%", 1)[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise _Refused(f"refused: private address ({host} → {ip})")
    # ponytail: resolve-then-connect leaves a DNS-rebinding window (TOCTOU); pin the
    # connection to the checked address if a hostile resolver ever matters here.


# ---------------------------------------------------------------------------
# HTTP with the guard on every hop and the byte cap
# ---------------------------------------------------------------------------

def _http_client(cfg: FetchConfig) -> httpx.Client:
    return httpx.Client(timeout=cfg.timeout_s, follow_redirects=False, headers={"User-Agent": _UA})


def _get(client: httpx.Client, url: str, cfg: FetchConfig) -> tuple[str, bytes, str]:
    """(final URL, body, content-type) of a GET: ≤ 5 redirects, each hop through
    the guard; the body stops at ``cfg.max_bytes``."""
    for _ in range(_MAX_REDIRECTS + 1):
        check_target(url)
        with client.stream("GET", url) as r:
            if r.status_code in _REDIRECTS:
                location = r.headers.get("location")
                if not location:
                    raise _RungError(f"HTTP {r.status_code} without a Location")
                url = urljoin(url, location)
                continue
            if r.status_code >= 400:
                raise _RungError(f"HTTP {r.status_code}")
            declared = int(r.headers.get("content-length") or 0)
            if declared > cfg.max_bytes:
                raise _RungError(f"too large ({declared} > {cfg.max_bytes} bytes)")
            body = bytearray()
            for chunk in r.iter_bytes():
                body += chunk
                if len(body) > cfg.max_bytes:
                    raise _RungError(f"too large (> {cfg.max_bytes} bytes)")
            return url, bytes(body), r.headers.get("content-type", "")
    raise _RungError(f"more than {_MAX_REDIRECTS} redirects")


def _is_pdf(url: str, ctype: str, body: bytes) -> bool:
    return "pdf" in ctype.lower() or urlsplit(url).path.lower().endswith(".pdf") or body[:5] == b"%PDF-"


def _pdf_text(body: bytes, limit: int) -> str:
    """The PDF's text, page by page, until ``limit`` chars (all that is kept on disk)."""
    try:
        pages, total = [], 0
        for page in pypdf.PdfReader(BytesIO(body)).pages:
            pages.append((page.extract_text() or "").strip())
            total += len(pages[-1]) + 2
            if total >= limit:
                break
        return "\n\n".join(pages).strip()
    except pypdf.errors.PyPdfError as e:
        raise _RungError(f"pdf: {type(e).__name__}: {e}") from e


def _html_text(html: bytes, url: str) -> tuple[str, str]:
    """trafilatura on the raw bytes (it detects the encoding itself)."""
    text = trafilatura.extract(html, url=url, include_links=False, favor_recall=True) or ""
    if len(text.strip()) < _MIN_CHARS:
        raise _RungError(f"no article text extracted ({len(text.strip())} chars)")
    meta = trafilatura.extract_metadata(html, default_url=url)
    return text, (meta.title if meta is not None and meta.title else "")


def _page(client: httpx.Client, url: str, cfg: FetchConfig) -> tuple[str, str]:
    final, body, ctype = _get(client, url, cfg)
    if _is_pdf(final, ctype, body):
        return _pdf_text(body, cfg.max_chars_on_disk), ""
    return _html_text(body, final)


_ARXIV_ABS = "https://arxiv.org/abs/"
_AR5IV_HOSTS = ("ar5iv.labs.arxiv.org", "ar5iv.org")


def _arxiv(client: httpx.Client, paper: str, cfg: FetchConfig) -> tuple[str, str]:
    """ar5iv's HTML rendering, else the PDF through pypdf."""
    try:
        final, body, _ctype = _get(client, f"https://ar5iv.labs.arxiv.org/html/{paper}", cfg)
        if urlsplit(final).hostname not in _AR5IV_HOSTS:
            raise _RungError(f"ar5iv has no rendering (redirected to {final})")
        return _html_text(body, final)
    except _RungError as e:
        ar5iv_error = e
    try:
        _final, body, _ctype = _get(client, f"https://arxiv.org/pdf/{paper}", cfg)
        return _pdf_text(body, cfg.max_chars_on_disk), ""
    except _RungError as e:
        raise _RungError(f"ar5iv: {ar5iv_error}; pdf: {e}") from e


def _github_repo(url: str) -> tuple[str, str] | None:
    parts = urlsplit(canonical_url(url))
    segments = [s for s in parts.path.split("/") if s]
    return (segments[0], segments[1]) if parts.hostname == "github.com" and len(segments) == 2 else None


def _native(url: str, source: dict, cfg: FetchConfig) -> tuple[str, str]:
    with _http_client(cfg) as client:
        canonical = canonical_url(url)
        if canonical.startswith(_ARXIV_ABS):
            return _arxiv(client, canonical[len(_ARXIV_ABS):], cfg)
        repo = _github_repo(url)
        if repo:
            readme = f"https://raw.githubusercontent.com/{repo[0]}/{repo[1]}/HEAD/README.md"
            try:
                _final, body, _ctype = _get(client, readme, cfg)
            except _RungError as e:
                raise _RungError(f"README: {e}") from e
            return body.decode("utf-8", errors="replace"), f"{repo[0]}/{repo[1]}"
        return _page(client, url, cfg)


def _exa_contents(url: str, cfg: FetchConfig, ledger: SpendLedger | None) -> tuple[str, str]:
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        raise _RungError("no EXA_API_KEY")
    if ledger is None or not ledger.take("exa_contents_pages"):
        raise _RungError("page budget spent")
    with _http_client(cfg) as client:
        r = client.post(_EXA_CONTENTS, json={"urls": [url], "text": {"maxCharacters": cfg.max_chars_on_disk}},
                        headers={"x-api-key": key})
    if r.status_code >= 400:
        raise _RungError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    cost = (data.get("costDollars") or {}).get("total")
    if cost:
        ledger.add_usd(float(cost))
    results = data.get("results") or []
    text = (results[0].get("text") if results else "") or ""
    if len(text.strip()) < _MIN_CHARS:
        raise _RungError(f"no text (statuses: {data.get('statuses')})")
    return text, results[0].get("title") or ""


def _intel(rung: str, url: str) -> tuple[str, str]:
    if rung == "intel_snatch":
        res = snatch_url(url, kind="auto")
    elif rung == "intel_crawl":
        res = crawl_url(url, deep=False, max_pages=4)
    else:
        res = snatch_url(url, kind="auto", stealth=True)
    if not (res.ok and res.markdown_path):
        raise _RungError(res.error or "not ok")
    return Path(res.markdown_path).read_text(encoding="utf-8", errors="replace"), getattr(res, "title", "")


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------

def _scratch_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "argus_fetch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store(rung: str, url: str, text: str, cfg: FetchConfig) -> tuple[str, str]:
    """The text on disk, LF line ends, capped at ``max_chars_on_disk``; ``chars`` is its length."""
    text = re.sub(r"\r\n?", "\n", text).strip()[:cfg.max_chars_on_disk]
    path = _scratch_dir() / f"{rung}_{hashlib.sha1(url.encode()).hexdigest()[:16]}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return str(path), text


def fetch(source: dict, *, cfg: FetchConfig, ledger: SpendLedger | None, exa_enabled: bool) -> FetchResult:
    """Fetch one search hit through the available chain (see the module doc)."""
    url = source.get("url") or ""
    t0 = time.monotonic()
    tried: list[str] = []
    errors: list[str] = []

    def ms() -> int:
        return int((time.monotonic() - t0) * 1000)

    for rung in available_chain(cfg.chain):
        if rung == "exa_text":
            text = (source.get("text") or "").strip()
            if len(text) <= _EXA_TEXT_MIN:
                continue                       # no inline text on this hit: not applicable
            title = source.get("title") or ""
        elif rung == "exa_contents" and not exa_enabled:
            continue
        else:
            tried.append(rung)
            try:
                check_target(url)
                text, title = (_native(url, source, cfg) if rung == "native" else
                               _exa_contents(url, cfg, ledger) if rung == "exa_contents" else
                               _intel(rung, url))
                if len(text.strip()) < _MIN_CHARS:
                    raise _RungError(f"only {len(text.strip())} chars of text")
            except _Refused as e:
                return FetchResult(ok=False, error=str(e), rungs_tried=tried, ms=ms())
            except _RungError as e:
                errors.append(f"{rung}: {e}")
                continue
            except Exception as e:             # one broken rung must not lose the source: recorded, next rung
                errors.append(f"{rung}: {type(e).__name__}: {e}")
                continue
        if rung == "exa_text":
            tried.append(rung)
        path, text = _store(rung, url, text, cfg)
        return FetchResult(ok=True, markdown_path=path, title=title or source.get("title") or "",
                           chars=len(text), fetcher=rung, ms=ms(), rungs_tried=tried)
    return FetchResult(ok=False, error="; ".join(errors) or f"no fetch rung applies (chain: {cfg.chain})",
                       rungs_tried=tried, ms=ms())


__all__ = ["FetchResult", "fetch", "available_chain", "intel_available", "check_target", "INTEL_AVAILABLE"]
