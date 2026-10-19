"""The fetch chain (v4 spec §2.8): one source → markdown on disk.

Rungs, in ``FetchConfig.chain`` order, first success wins:

- ``exa_text``     the search hit already carries Exa's page text — no request;
- ``native``       httpx + trafilatura / pypdf, by URL class: arXiv → ar5iv HTML,
                   else the PDF; a GitHub repo → its raw README; a PDF → pypdf;
                   anything else → ``trafilatura.extract(html, include_links=False,
                   favor_recall=True)``;
- ``exa_contents`` ``POST https://api.exa.ai/contents`` (the page taken from the
                   ledger's cap — or only counted, for a pinned source — and
                   ``costDollars`` added to it);
- ``intel_*``      the intel-stack scripts (snatch / crawl / stealth) — in the
                   chain only when both ``INTEL_*`` paths exist (checked at import).

Guards (spec §2.8, WP-B review):

- the URL is read the WHATWG way first (``\\`` is ``/`` in an http(s) URL), for
  every rung, so the host the guard checks is the host any rung reaches;
- before every network rung, and on every redirect hop of ``native``, every
  address the host resolves to must be global — an IPv6 address carrying an
  IPv4 one (mapped, compatible, NAT64) is judged by that IPv4 — and must not be
  this server's own endpoint (the proxy, the console's public URL; compared by
  normalised name and port, never by address); else ``error = "refused: private
  address (…)"`` and the chain stops;
- ``native`` sends each hop to the address the guard checked — never a second
  DNS lookup — with the original ``Host`` header and TLS ``sni_hostname``;
  through an HTTP(S) proxy from the environment (httpx's ``trust_env``
  selection, ``NO_PROXY`` included) the hop goes by name instead: the proxy
  resolves it, so the pin cannot apply (the guard still checks first);
- every download stops at ``FetchConfig.max_bytes`` decoded bytes: requests send
  ``Accept-Encoding: gzip, deflate``, bodies go through a bounded decompressor,
  any other ``Content-Encoding`` is refused. Text on disk is capped at
  ``max_chars_on_disk``.

A rung's expected failures — HTTP status, network and timeout errors, size, a
third-party parser's exception — are its own error type, recorded with the
exception type; anything else propagates. ``rungs_tried`` lists every rung tried
with its error (``""`` for the one that succeeded), on success too.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit, urlunsplit

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
_ACCEPT_ENCODING = "gzip, deflate"          # decoded here, under the byte cap
_WBITS = {"gzip": 16 + zlib.MAX_WBITS, "x-gzip": 16 + zlib.MAX_WBITS, "deflate": zlib.MAX_WBITS}
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_HTTP_SCHEME = re.compile(r"\s*https?:", re.IGNORECASE)


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
    rungs_tried: list[dict] = field(default_factory=list)   # [{"rung", "error"}]; "" for the winner


class _Refused(Exception):
    """The SSRF guard refused the target: no rung may fetch it."""


class _RungError(Exception):
    """An expected failure of this rung (HTTP status, network / timeout, size, a
    third-party parser's exception, with its type); the chain moves on."""


# ---------------------------------------------------------------------------
# the SSRF guard
# ---------------------------------------------------------------------------

def _whatwg(url: str) -> str:
    """An http(s) URL with ``\\`` read as ``/`` before its query or fragment, as
    browsers read it (WHATWG): ``http://127.0.0.1\\@evil.example/`` goes to
    127.0.0.1, whatever ``urlsplit`` makes of it."""
    if not _HTTP_SCHEME.match(url):
        return url
    cut = min([i for i in (url.find("?"), url.find("#")) if i >= 0] or [len(url)])
    return url[:cut].replace("\\", "/") + url[cut:]


def _norm_host(host: str) -> str:
    """The host as the resolver and the request see it: IDNA-encoded (``。``, ``．``
    and ``｡`` are label separators too), lowercase, no trailing dot. ``UnicodeError``
    for a name IDNA cannot encode."""
    return host.encode("idna").decode("ascii").lower().rstrip(".")


def _port(parts) -> int:
    return parts.port or (443 if parts.scheme == "https" else 80)


def _split(url: str) -> tuple[str, int]:
    """``(host, port)`` of an http(s) URL, the host normalised (``_norm_host``)."""
    try:
        parts = urlsplit(url)
        port = _port(parts)
    except ValueError as e:                       # a bad port, an unclosed bracket
        raise _RungError(f"malformed URL {url!r} ({e})") from e
    try:
        host = _norm_host(parts.hostname or "")
    except UnicodeError as e:                     # an empty or overlong label, a name IDNA rejects
        raise _RungError(f"dns: {parts.hostname} is not a valid host name (UnicodeError: {e})") from e
    if not host or parts.scheme not in ("http", "https"):
        raise _RungError(f"not an http(s) URL: {url!r}")
    return host, port


def _resolve(host: str) -> list[str]:
    """Every address ``host`` resolves to, in the resolver's order."""
    return list(dict.fromkeys(info[4][0] for info in socket.getaddrinfo(host, None)))


def _addresses(host: str) -> list[str]:
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    try:
        return _resolve(host)
    except socket.gaierror as e:
        raise _RungError(f"dns: {host} did not resolve (gaierror: {e})") from e


def _judged(addr: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """The address the guard judges: an IPv6 address carrying an IPv4 one — mapped
    (``::ffff:a.b.c.d``), compatible (``::a.b.c.d``, not ``::`` or ``::1``) or NAT64
    (``64:ff9b::/96``) — by that IPv4 address."""
    ip = ipaddress.ip_address(addr.split("%", 1)[0])
    if ip.version == 6:
        if ip.ipv4_mapped:
            return ip.ipv4_mapped
        if (int(ip) >> 32 == 0 and int(ip) > 1) or ip in _NAT64:
            return ipaddress.IPv4Address(int(ip) & 0xFFFF_FFFF)
    return ip


def _own_endpoints() -> set[tuple[str, int]]:
    """This server's own endpoints — the proxy (``FREELLMAPI_BASE_URL``) and the
    console's public URL — as normalised ``(host, port)`` pairs. Never compared by
    address: a public endpoint's address is shared (a CDN, a tunnel edge), and a
    private one is refused by the private-address rule anyway."""
    own: set[tuple[str, int]] = set()
    for env in ("FREELLMAPI_BASE_URL", "ARGUS_WEBAPP_PUBLIC_URL"):
        parts = urlsplit(os.environ.get(env, "").strip())
        host = _norm_host(parts.hostname or "")
        if host:
            own.add((host, _port(parts)))
    return own


def check_target(url: str) -> list[str]:
    """The addresses ``url``'s host resolves to, each one checked: ``_Refused`` for
    this server's own endpoint or a non-global address, ``_RungError`` for a
    malformed URL or a name that does not resolve."""
    host, port = _split(url)
    if (host, port) in _own_endpoints():
        raise _Refused(f"refused: private address (own endpoint {host}:{port})")
    addrs = _addresses(host)
    for a in addrs:
        ip = _judged(a)
        if not ip.is_global:
            raise _Refused(f"refused: private address ({host} → {ip})")
    return addrs


# ---------------------------------------------------------------------------
# HTTP: every hop through the guard and to the address it checked, the byte cap
# ---------------------------------------------------------------------------

def _http_client(cfg: FetchConfig, *, pinned: bool = False) -> httpx.Client:
    """No keep-alive: the pool keys connections by address, so a later hop to another
    name on the same address would reuse a TLS session verified for the first name.
    httpx's ``trust_env`` mounts the environment's HTTP(S) proxies (``NO_PROXY``
    honoured); a ``pinned`` client has an explicit transport, so httpx mounts none and
    a request to a checked address never goes through a proxy."""
    limits = httpx.Limits(max_keepalive_connections=0)
    return httpx.Client(timeout=cfg.timeout_s, follow_redirects=False,
                        headers={"User-Agent": _UA, "Accept-Encoding": _ACCEPT_ENCODING}, limits=limits,
                        transport=httpx.HTTPTransport(limits=limits) if pinned else None)


def _open(env: httpx.Client, pinned: httpx.Client, url: str, addrs: list[str]) -> httpx.Response:
    """A streamed GET of ``url``. When an HTTP(S) proxy from the environment applies
    to it, by name through ``env``: the proxy resolves the name, so the pin cannot
    apply. Otherwise on ``pinned``, to the checked addresses in order (the next one
    when a connection fails or times out, as a connect by name would):
    ``scheme://<address>[:port]/path?query`` with the original ``Host`` header and
    ``sni_hostname``, so TLS still verifies the real name."""
    # httpx's own trust_env selection (it has no public call for it): the one send() makes
    if env._transport_for_url(httpx.URL(url)) is not env._transport:
        return env.send(env.build_request("GET", url, headers={"Accept-Encoding": _ACCEPT_ENCODING}),
                        stream=True)
    parts = urlsplit(url)
    name = _norm_host(parts.hostname or "")        # the name the guard checked, IDNA-encoded
    host = f"[{name}]" if ":" in name else name
    if parts.port and parts.port != (443 if parts.scheme == "https" else 80):
        host += f":{parts.port}"

    def send(addr: str) -> httpx.Response:
        netloc = (f"[{addr}]" if ":" in addr else addr) + (f":{parts.port}" if parts.port else "")
        request = pinned.build_request(
            "GET", urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, "")),
            headers={"Host": host, "Accept-Encoding": _ACCEPT_ENCODING}, extensions={"sni_hostname": name})
        return pinned.send(request, stream=True)

    *rest, last = addrs
    for addr in rest:
        try:
            return send(addr)
        except (httpx.ConnectError, httpx.ConnectTimeout):   # a refused or a blackholed address
            continue
    return send(last)


def _too_large(cfg: FetchConfig) -> _RungError:
    return _RungError(f"too large (> {cfg.max_bytes} bytes)")


def _read_body(r: httpx.Response, cfg: FetchConfig) -> bytes:
    """The body, decoded as it streams in and never more than ``cfg.max_bytes``
    decoded bytes (+ one chunk): gzip / deflate through a bounded ``zlib``
    decompressor; any other ``Content-Encoding`` is refused before a byte is decoded."""
    declared = int(r.headers.get("content-length") or 0)
    if declared > cfg.max_bytes:
        raise _RungError(f"too large ({declared} > {cfg.max_bytes} bytes)")
    encoding = r.headers.get("content-encoding", "")
    codings = [c for c in (t.strip().lower() for t in encoding.split(",")) if c and c != "identity"]
    if len(codings) > 1 or (codings and codings[0] not in _WBITS):
        raise _RungError(f"Content-Encoding {encoding!r} refused (only gzip or deflate is decoded)")
    inflate = zlib.decompressobj(_WBITS[codings[0]]) if codings else None
    body, raw = bytearray(), 0
    for chunk in r.iter_raw():
        raw += len(chunk)
        if raw > cfg.max_bytes:
            raise _too_large(cfg)
        if inflate is None:
            body += chunk
        while inflate is not None and chunk and not inflate.eof:
            try:
                body += inflate.decompress(chunk, cfg.max_bytes - len(body) + 1)
            except zlib.error as e:
                raise _RungError(f"{codings[0]}: zlib.error: {e}") from e
            if len(body) > cfg.max_bytes:
                raise _too_large(cfg)
            chunk = inflate.unconsumed_tail
        if len(body) > cfg.max_bytes:
            raise _too_large(cfg)
        if inflate is not None and inflate.eof:
            break                                  # past the end of the compressed stream: nothing to keep
    return bytes(body)


def _get(client: httpx.Client, url: str, cfg: FetchConfig) -> tuple[str, bytes, str]:
    """(final URL, decoded body, content-type) of a GET: at most ``_MAX_REDIRECTS``
    redirects, every hop read the WHATWG way, through the guard and sent to the
    address it checked — or by name through the environment's proxy, when one
    applies (``_open``); the body under the byte cap. ``client`` trusts the
    environment; the pinned hops go on a client of their own."""
    pinned = _http_client(cfg, pinned=True)
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            url = _whatwg(url)
            r = _open(client, pinned, url, check_target(url))
            try:
                if r.status_code in _REDIRECTS:
                    location = r.headers.get("location")
                    if not location:
                        raise _RungError(f"HTTP {r.status_code} without a Location")
                    try:
                        url = urljoin(url, location)
                    except ValueError as e:
                        raise _RungError(f"HTTP {r.status_code} to a malformed Location {location!r} ({e})") from e
                    continue
                if r.status_code >= 400:
                    raise _RungError(f"HTTP {r.status_code}")
                return url, _read_body(r, cfg), r.headers.get("content-type", "")
            finally:
                r.close()
    except (httpx.TransportError, httpx.InvalidURL) as e:     # network, timeout, protocol; a URL httpx rejects
        raise _RungError(f"{type(e).__name__}: {e}") from e
    finally:
        pinned.close()
    raise _RungError(f"more than {_MAX_REDIRECTS} redirects")


def _is_pdf(url: str, ctype: str, body: bytes) -> bool:
    """A PDF by content type, ``.pdf`` path or ``%PDF-`` bytes — but an HTML
    response is HTML whatever its path says."""
    mtype = ctype.split(";", 1)[0].strip().lower()
    if mtype in ("text/html", "application/xhtml+xml"):
        return False
    return "pdf" in mtype or urlsplit(url).path.lower().endswith(".pdf") or body[:5] == b"%PDF-"


def _pdf_text(body: bytes, limit: int) -> str:
    """The PDF's text, page by page, until ``limit`` chars (all that is kept on disk)."""
    pages, total = [], 0
    try:
        for page in pypdf.PdfReader(BytesIO(body)).pages:
            pages.append((page.extract_text() or "").strip())
            total += len(pages[-1]) + 2
            if total >= limit:
                break
    except Exception as e:      # a third-party parser on untrusted bytes: what it raises is this rung's failure
        raise _RungError(f"pdf: {type(e).__name__}: {e}") from e
    return "\n\n".join(pages).strip()


def _html_text(html: bytes, url: str) -> tuple[str, str]:
    """trafilatura on the raw bytes (it detects the encoding itself)."""
    try:
        text = trafilatura.extract(html, url=url, include_links=False, favor_recall=True) or ""
        meta = trafilatura.extract_metadata(html, default_url=url) if len(text.strip()) >= _MIN_CHARS else None
    except Exception as e:      # a third-party parser on untrusted bytes: what it raises is this rung's failure
        raise _RungError(f"trafilatura: {type(e).__name__}: {e}") from e
    if len(text.strip()) < _MIN_CHARS:
        raise _RungError(f"no article text extracted ({len(text.strip())} chars)")
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


def _exa_contents(url: str, cfg: FetchConfig, ledger: SpendLedger | None, pinned: bool) -> tuple[str, str]:
    """``pinned``: the page is counted, never taken from the cap (spec §2.8) — the
    counter is shared, so it does reduce what remains for unpinned pages."""
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        raise _RungError("no EXA_API_KEY")
    if ledger is None:
        raise _RungError("page budget spent")
    if pinned:
        ledger.count("exa_contents_pages")
    elif not ledger.take("exa_contents_pages"):
        raise _RungError("page budget spent")
    try:
        with _http_client(cfg) as client:
            r = client.post(_EXA_CONTENTS, json={"urls": [url], "text": {"maxCharacters": cfg.max_chars_on_disk}},
                            headers={"x-api-key": key})
    except httpx.TransportError as e:
        raise _RungError(f"{type(e).__name__}: {e}") from e
    if r.status_code >= 400:
        raise _RungError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        data = r.json()
    except ValueError as e:
        raise _RungError(f"malformed reply: not JSON ({type(e).__name__}: {e})") from e
    text, title = _exa_reply(data, ledger)
    if len(text.strip()) < _MIN_CHARS:
        raise _RungError(f"no text (statuses: {data.get('statuses')})")
    return text, title


def _exa_reply(data, ledger: SpendLedger) -> tuple[str, str]:
    """The first result's text and title from a ``/contents`` reply, with
    ``costDollars.total`` added to the ledger first (it was billed). A reply of
    any other shape is this rung's ``malformed reply: …`` error."""
    def kind(value) -> str:
        return "missing" if value is None else f"a JSON {type(value).__name__}"

    if not isinstance(data, dict):
        raise _RungError(f"malformed reply: {kind(data)}, not an object")
    cost = data.get("costDollars")
    if not isinstance(cost, dict):
        raise _RungError(f"malformed reply: costDollars is {kind(cost)}, not an object")
    total = cost.get("total")
    if isinstance(total, bool) or not isinstance(total, int | float):
        raise _RungError(f"malformed reply: costDollars.total is {kind(total)}, not a number")
    if total:
        ledger.add_usd(float(total))
    results = data.get("results")
    if not isinstance(results, list):
        raise _RungError(f"malformed reply: results is {kind(results)}, not a list")
    first = results[0] if results else {}
    if not isinstance(first, dict):
        raise _RungError(f"malformed reply: a result is {kind(first)}, not an object")
    text, title = first.get("text") or "", first.get("title") or ""
    if not isinstance(text, str) or not isinstance(title, str):
        raise _RungError(f"malformed reply: a result's text is {kind(text)}, its title {kind(title)}")
    return text, title


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


def fetch(source: dict, *, cfg: FetchConfig, ledger: SpendLedger | None, exa_enabled: bool,
          pinned: bool = False) -> FetchResult:
    """Fetch one search hit through the available chain (see the module doc).
    ``pinned``: a user-appended source, which never calls ``take``."""
    url = _whatwg(source.get("url") or "")
    t0 = time.monotonic()
    tried: list[dict] = []

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
            try:
                check_target(url)
                text, title = (_native(url, source, cfg) if rung == "native" else
                               _exa_contents(url, cfg, ledger, pinned) if rung == "exa_contents" else
                               _intel(rung, url))
                if len(text.strip()) < _MIN_CHARS:
                    raise _RungError(f"only {len(text.strip())} chars of text")
            except _Refused as e:
                tried.append({"rung": rung, "error": str(e)})
                return FetchResult(ok=False, error=str(e), rungs_tried=tried, ms=ms())
            except _RungError as e:
                tried.append({"rung": rung, "error": str(e)})
                continue
        tried.append({"rung": rung, "error": ""})
        path, text = _store(rung, url, text, cfg)
        return FetchResult(ok=True, markdown_path=path, title=title or source.get("title") or "",
                           chars=len(text), fetcher=rung, ms=ms(), rungs_tried=tried)
    return FetchResult(ok=False, error="; ".join(f"{t['rung']}: {t['error']}" for t in tried)
                       or f"no fetch rung applies (chain: {cfg.chain})", rungs_tried=tried, ms=ms())


__all__ = ["FetchResult", "fetch", "available_chain", "intel_available", "check_target", "INTEL_AVAILABLE"]
