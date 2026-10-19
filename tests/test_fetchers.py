"""engines/fetchers.py (v4 spec §2.8): the fetch chain, the native rung on
fixtures, the SSRF guard and the byte cap, Exa text / contents, chain
availability — plus the research-side seams that turn a FetchResult into a
FetchedItem and keep pinned /append sources outside every cap.

Hermetic: an httpx MockTransport stands in for the network and ``_resolve``
for DNS; nothing leaves the machine."""
from __future__ import annotations

import json
import textwrap
from io import BytesIO
from pathlib import Path

import httpx
import pytest

from argus.engines import fetchers
from argus.engines.ledger import SpendLedger
from argus.graph import research
from argus.graph.run_config import FetchConfig

from helpers import as_stream, request_url

PUBLIC_IP = "93.184.216.34"

ARTICLE = """<html><head><title>Interrupts in LangGraph</title></head><body>
<nav><a href="/">Home</a> | <a href="/docs">Docs</a> | Navigation menu boilerplate</nav>
<article><h1>Interrupts in LangGraph</h1>
<p>The interrupt function pauses a graph at a node and surfaces a value to the caller.
The run is checkpointed, so nothing is lost while the graph waits for a human.</p>
<p>To resume, the caller invokes the graph again with a Command carrying the resume value.
The node that called interrupt runs again from its start and receives that value.</p>
<p>Because the node re-runs, any side effect before the interrupt call executes twice,
which is why side effects belong after the interrupt or in a separate node.</p>
<p>Static breakpoints are configured at compile time with interrupt_before and
interrupt_after, while dynamic interrupts are raised from inside a node at run time.</p>
</article><footer>Copyright 2026 Example Corp. All rights reserved.</footer></body></html>"""


class _Net:
    """A fake network: ``routes[url]`` is an httpx.Response or a callable
    ``request → Response``, keyed by the URL the request is for (``Host``
    header + path: ``request.url`` carries the checked address);
    ``dns[host]`` the addresses a name resolves to."""

    def __init__(self):
        self.routes: dict = {}
        self.dns: dict[str, list[str]] = {}
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.routes.get(request_url(request))
        if route is None:
            return httpx.Response(404, text="not found")
        return as_stream(route(request) if callable(route) else route)

    def hosts(self) -> list[str]:
        return [r.headers["host"] for r in self.requests]


@pytest.fixture
def net(monkeypatch, tmp_path):
    n = _Net()
    monkeypatch.setattr(fetchers, "_http_client", lambda cfg, **_: httpx.Client(
        transport=httpx.MockTransport(n.handler), timeout=cfg.timeout_s))
    monkeypatch.setattr(fetchers, "_resolve", lambda host: n.dns.get(host, [PUBLIC_IP]))
    monkeypatch.setattr(fetchers, "_scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("FREELLMAPI_BASE_URL", raising=False)
    monkeypatch.delenv("ARGUS_WEBAPP_PUBLIC_URL", raising=False)
    return n


def _fetch(src: dict, *, cfg: FetchConfig | None = None, ledger: SpendLedger | None = None,
           exa: bool = False) -> fetchers.FetchResult:
    return fetchers.fetch(src, cfg=cfg or FetchConfig(), ledger=ledger or SpendLedger({}),
                          exa_enabled=exa)


def _rungs(res: fetchers.FetchResult) -> list[str]:
    return [t["rung"] for t in res.rungs_tried]


def _text(res: fetchers.FetchResult) -> str:
    return Path(res.markdown_path).read_text(encoding="utf-8")


def _pdf(text: str) -> bytes:
    from reportlab.pdfgen import canvas
    buf = BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in textwrap.wrap(text, 80):
        c.drawString(40, y, line)
        y -= 14
    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Review Focus 3 — no intel-stack and no Exa text
# ---------------------------------------------------------------------------

def test_native_rung_without_intel_stack_or_exa(net):
    cfg = FetchConfig()
    assert fetchers.available_chain(cfg.chain) == ["exa_text", "native", "exa_contents"], (
        "without both INTEL_* paths the intel rungs are filtered out")

    # 1. native fetches fixture HTML on its own
    net.routes["https://docs.example.org/interrupts"] = httpx.Response(200, html=ARTICLE)
    res = _fetch({"url": "https://docs.example.org/interrupts", "title": "t"}, cfg=cfg)
    assert (res.ok, res.fetcher, res.rungs_tried) == (True, "native", [{"rung": "native", "error": ""}])
    text = _text(res)
    assert "pauses a graph at a node" in text
    assert "Navigation menu boilerplate" not in text, "trafilatura keeps the article, not the chrome"
    assert res.chars == len(text) and res.title
    assert res.ms >= 0

    # 2. a source no rung can fetch is reported with the rungs tried, never dropped
    net.routes["https://dead.example/x"] = httpx.Response(503)
    res = _fetch({"url": "https://dead.example/x"}, cfg=cfg)
    assert (res.ok, _rungs(res)) == (False, ["native"])
    assert "503" in res.error
    fetched, errors = research._parallel_fetch([{"url": "https://dead.example/x", "title": "d"}],
                                               cfg=cfg, ledger=SpendLedger({}), exa_on=False)
    assert fetched == []
    assert errors == ["fetch https://dead.example/x failed: native: HTTP 503 (rungs tried: native)"]

    # 3. a pinned /append source is ingested even with the cap spent
    ledger = SpendLedger({"fetch_first": 0})
    pinned = {"url": "https://docs.example.org/interrupts", "title": "mine", "kind": "search_result",
              "summary": "", "source": "appended-url"}
    web = {"url": "https://other.example/y", "title": "web", "sub_qs": [0]}
    fetched, errors = research._parallel_fetch([pinned, web], cfg=cfg, ledger=ledger, exa_on=False,
                                               kind="fetch_first")
    assert [f["url"] for f in fetched] == [pinned["url"]]
    assert fetched[0]["pinned"] is True and fetched[0]["fetcher"] == "native"
    assert "other.example" not in net.hosts(), "no budget left: the web source is not fetched"
    assert ledger.snapshot()["fetch_first"] == 0, "pinned sources never call take()"


# ---------------------------------------------------------------------------
# Review Focus 7 — SSRF and size guards
# ---------------------------------------------------------------------------

def test_native_refuses_private_targets_and_caps_bytes(net, monkeypatch):
    net.dns.update({"localhost": ["127.0.0.1"], "internal.example": ["10.9.8.7"],
                    "proxy.example.net": [PUBLIC_IP]})
    monkeypatch.setenv("FREELLMAPI_BASE_URL", "https://proxy.example.net:3001/v1")
    private = [
        "http://127.0.0.1:8787/api/runs",             # the console's port
        "http://localhost:3001/v1/models",            # the proxy's port (admin UI / API)
        "https://proxy.example.net:3001/admin",       # the proxy's own endpoint, even on a public address
        "http://10.0.0.5/admin",                      # RFC 1918
        "http://169.254.169.254/latest/meta-data/",   # link-local (cloud metadata)
        "http://[::1]:8080/",                         # IPv6 loopback
        "http://internal.example/x",                  # a name resolving to a private address
        # WP-B review: forms that slipped through
        "http://[::127.0.0.1]/",                      # IPv4-compatible IPv6, judged by its IPv4
        "http://[64:ff9b::7f00:1]/",                  # NAT64 prefix embedding 127.0.0.1
        "http://127.0.0.1\\@evil.example/",           # WHATWG: "\" is "/", so the host is 127.0.0.1
        "https://proxy.example.net.:3001/admin",      # the proxy's own name with a trailing dot
    ]
    for url in private:
        res = _fetch({"url": url})
        assert not res.ok
        assert res.error.startswith("refused: private address"), (url, res.error)
    assert net.requests == [], "no request ever reached a private target"

    # a public page redirecting to a private address is refused at that hop
    net.routes["https://public.example/r"] = httpx.Response(
        302, headers={"location": "http://10.0.0.7/secret"})
    res = _fetch({"url": "https://public.example/r"})
    assert res.error.startswith("refused: private address"), res.error
    assert net.hosts() == ["public.example"], "the private hop is never requested"

    # a 10 MB download stops at max_bytes
    consumed: list[int] = []

    def big(request):
        def body():
            for _ in range(160):                      # 160 × 64 KiB ≈ 10 MB
                consumed.append(65536)
                yield b"<p>" + b"x" * 65533
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body())

    net.routes["https://big.example/page"] = big
    res = _fetch({"url": "https://big.example/page"}, cfg=FetchConfig(max_bytes=1_000_000))
    assert not res.ok and "too large" in res.error, res.error
    assert sum(consumed) <= 1_000_000 + 65536, f"read {sum(consumed)} bytes past a 1 MB cap"


# ---------------------------------------------------------------------------
# rungs
# ---------------------------------------------------------------------------

def test_chain_availability_needs_both_intel_paths(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    python = tmp_path / "python.exe"
    python.write_text("")
    assert fetchers.intel_available(scripts, python) is True
    assert fetchers.intel_available(scripts, tmp_path / "missing.exe") is False
    assert fetchers.intel_available(tmp_path / "nope", python) is False
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", True)
    assert fetchers.available_chain(FetchConfig().chain) == FetchConfig().chain


def test_exa_text_rung_needs_no_request(net):
    res = _fetch({"url": "https://exa.example/p", "title": "P", "text": "Inline page text. " * 40}, exa=True)
    assert (res.ok, res.fetcher, _rungs(res)) == (True, "exa_text", ["exa_text"])
    assert "Inline page text." in _text(res) and net.requests == []


def test_exa_contents_counts_pages_and_cost(net, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    net.routes["https://walled.example/a"] = httpx.Response(403)

    def contents(request):
        body = json.loads(request.content)
        assert body["urls"] == ["https://walled.example/a"]
        assert request.headers["x-api-key"] == "test-key"
        return httpx.Response(200, json={
            "results": [{"url": "https://walled.example/a", "title": "Walled", "text": "Walled text. " * 80}],
            "costDollars": {"total": 0.001}})

    net.routes["https://api.exa.ai/contents"] = contents
    ledger = SpendLedger({"exa_contents_pages": 1})
    res = _fetch({"url": "https://walled.example/a"}, ledger=ledger, exa=True)
    assert (res.ok, res.fetcher) == (True, "exa_contents")
    assert res.rungs_tried == [{"rung": "native", "error": "HTTP 403"}, {"rung": "exa_contents", "error": ""}], (
        "a failed rung keeps its error when a later rung succeeds")
    assert "Walled text." in _text(res)
    snap = ledger.snapshot()
    assert snap["exa_contents_pages"] == 1 and snap["exa_usd"] == pytest.approx(0.001)

    res = _fetch({"url": "https://walled.example/a"}, ledger=ledger, exa=True)
    assert not res.ok and _rungs(res) == ["native", "exa_contents"]
    assert "exa_contents: page budget spent" in res.error


def test_native_arxiv_reads_ar5iv_then_the_pdf(net):
    net.routes["https://ar5iv.labs.arxiv.org/html/1706.03762"] = httpx.Response(404)
    net.routes["https://arxiv.org/pdf/1706.03762"] = httpx.Response(
        200, headers={"content-type": "application/pdf"},
        content=_pdf("The Transformer relies entirely on attention to draw global dependencies. " * 6))
    res = _fetch({"url": "http://arxiv.org/abs/1706.03762v7", "kind": "paper"})
    assert (res.ok, res.fetcher) == (True, "native")
    assert "relies entirely on attention" in _text(res)

    net.routes["https://ar5iv.labs.arxiv.org/html/1706.03762"] = httpx.Response(200, html=ARTICLE)
    res = _fetch({"url": "https://arxiv.org/pdf/1706.03762v2"})
    assert res.ok and "pauses a graph at a node" in _text(res), "ar5iv HTML is preferred when it exists"


def test_native_github_repo_reads_the_raw_readme(net):
    net.routes["https://raw.githubusercontent.com/o/r/HEAD/README.md"] = httpx.Response(
        200, text="# r\n\nThe README explains how the library resumes a paused graph. " * 10)
    res = _fetch({"url": "https://github.com/o/r?tab=readme-ov-file", "kind": "repo"})
    assert (res.ok, res.fetcher) == (True, "native")
    assert "resumes a paused graph" in _text(res)
    assert net.hosts() == ["raw.githubusercontent.com"]


def test_native_pdf_url_is_parsed_with_pypdf(net):
    net.routes["https://site.example/paper.pdf"] = httpx.Response(
        200, headers={"content-type": "application/octet-stream"},
        content=_pdf("Checkpointers persist graph state between supersteps. " * 8))
    res = _fetch({"url": "https://site.example/paper.pdf"})
    assert res.ok and "persist graph state" in _text(res)


def test_markdown_on_disk_is_capped(net):
    long_page = ARTICLE.replace("</article>", "<p>" + "More detail about resuming. " * 4000 + "</p></article>")
    net.routes["https://docs.example.org/long"] = httpx.Response(200, html=long_page)
    res = _fetch({"url": "https://docs.example.org/long"}, cfg=FetchConfig(max_chars_on_disk=5000))
    assert res.ok and res.chars == 5000 == len(_text(res))


def test_intel_rungs_run_after_native_when_available(net, monkeypatch, tmp_path):
    from argus.tools import CrawlResult, SnatchResult
    md = tmp_path / "snatched.md"
    md.write_text("# Snatched\n\nRecovered by the intel-stack snatch rung. " * 10, encoding="utf-8")
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", True)
    calls: list[str] = []

    def snatch(url, **kw):
        calls.append("stealth" if kw.get("stealth") else "snatch")
        return SnatchResult(ok=True, url=url, markdown_path=str(md), title="Snatched")

    def crawl(url, **kw):
        calls.append("crawl")
        return CrawlResult(ok=False, error="down")

    monkeypatch.setattr(fetchers, "snatch_url", snatch)
    monkeypatch.setattr(fetchers, "crawl_url", crawl)
    net.routes["https://walled.example/b"] = httpx.Response(403)
    res = _fetch({"url": "https://walled.example/b"})
    assert (res.ok, res.fetcher, _rungs(res)) == (True, "intel_snatch", ["native", "intel_snatch"])
    assert calls == ["snatch"]
    assert "Recovered by the intel-stack" in _text(res)


# ---------------------------------------------------------------------------
# research-side: FetchResult → FetchedItem
# ---------------------------------------------------------------------------

def test_fetched_item_carries_the_v4_fields(net):
    raw = "https://www.docs.example.org/interrupts/?utm_source=x"   # fetched as given; canonical = the dedupe key
    net.routes[raw] = httpx.Response(200, html=ARTICLE)
    item, errors = research.fetch_one_source(
        {"url": raw, "title": "Interrupts",
         "kind": "official_doc", "sub_qs": [1], "provider": "exa", "published": "2026-09-01"},
        cfg=FetchConfig(), ledger=SpendLedger({}), exa_enabled=False)
    assert errors == []
    assert item["canonical_url"] == "https://docs.example.org/interrupts"
    assert item["domain"] == "docs.example.org"
    assert (item["fetcher"], item["published"], item["pinned"]) == ("native", "2026-09-01", False)
    assert item["chars"] > 0 and item["fetch_ms"] >= 0
    assert item["authority_prior"] == 2, "an official doc by URL heuristics: 2 (3 is the curated primary tier)"
    assert item["sub_qs"] == [1] and item["provider"] == "exa"
    assert item["rungs_tried"] == [{"rung": "native", "error": ""}]


# ---------------------------------------------------------------------------
# WP-B review, fix round 1
# ---------------------------------------------------------------------------

def test_each_hop_goes_to_the_address_the_guard_checked(net):
    """A1: the request is sent to the checked address with the original Host and
    TLS sni_hostname (no second DNS lookup); a redirect hop is pinned the same way."""
    net.dns.update({"docs.example.org": [PUBLIC_IP], "cdn.example.net": ["151.101.1.69", "2606:4700::6810:84e5"]})
    net.routes["https://docs.example.org:8443/a?x=1"] = httpx.Response(
        302, headers={"location": "https://cdn.example.net/b"})
    net.routes["https://cdn.example.net/b"] = httpx.Response(200, html=ARTICLE)
    res = _fetch({"url": "https://docs.example.org:8443/a?x=1"})
    assert res.ok, res.error
    first, second = net.requests
    assert (str(first.url), first.headers["host"]) == (f"https://{PUBLIC_IP}:8443/a?x=1", "docs.example.org:8443")
    assert first.extensions["sni_hostname"] == "docs.example.org"
    assert (str(second.url), second.headers["host"]) == ("https://151.101.1.69/b", "cdn.example.net")
    assert second.extensions["sni_hostname"] == "cdn.example.net"
    assert first.headers["accept-encoding"] == "gzip, deflate"

    # IPv6 in brackets; a refused connection moves to the next checked address
    net.requests.clear()
    net.dns["v6.example"] = ["2606:4700::6810:84e5", "151.101.1.69"]

    def first_address_down(request):
        if request.url.host == "2606:4700::6810:84e5":
            raise httpx.ConnectError("network unreachable", request=request)
        return httpx.Response(200, html=ARTICLE)

    net.routes["https://v6.example/c"] = first_address_down
    res = _fetch({"url": "https://v6.example/c"})
    assert res.ok, res.error
    assert [str(r.url) for r in net.requests] == ["https://[2606:4700::6810:84e5]/c", "https://151.101.1.69/c"]


def test_dns_rebinding_never_reaches_a_loopback_server(monkeypatch, tmp_path):
    """A1 (Critical): the guard's resolver answers a public address, the system
    resolver loopback (the rebinding answer). The real transport connects to the
    checked public address (refused here, so nothing leaves the machine) and the
    loopback server receives nothing. At 98ebc5b httpx resolved the name again
    and fetched the loopback page."""
    import http.server
    import socket
    import threading

    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            body = ARTICLE.encode()
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    real_gai, real_connect = socket.getaddrinfo, socket.create_connection
    later_lookups: list[str] = []
    connects: list[tuple] = []

    def rebinding_gai(host, *a, **kw):
        if host == "rebind.test":
            later_lookups.append(host)
            return real_gai("127.0.0.1", *a, **kw)
        return real_gai(host, *a, **kw)

    def connect(address, *a, **kw):
        connects.append(address)
        if address[0] == PUBLIC_IP:
            raise ConnectionRefusedError("hermetic: no connection leaves the machine")
        return real_connect(address, *a, **kw)

    monkeypatch.setattr(fetchers, "_resolve", lambda host: [PUBLIC_IP])        # what the guard checked
    monkeypatch.setattr(socket, "getaddrinfo", rebinding_gai)                  # what a second lookup gets
    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr(fetchers, "_scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", False)
    for env in ("EXA_API_KEY", "FREELLMAPI_BASE_URL", "ARGUS_WEBAPP_PUBLIC_URL"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")              # no proxy from this machine's environment or registry
    try:
        res = _fetch({"url": f"http://rebind.test:{port}/admin"})
    finally:
        server.shutdown()
    assert hits == [], "the loopback server received the request"
    assert connects == [(PUBLIC_IP, port)], "the transport connects to the checked address only"
    assert later_lookups == [], "the name is never resolved a second time"
    assert not res.ok and "native: ConnectError" in res.error


def _gzip(data: bytes) -> bytes:
    import zlib
    c = zlib.compressobj(wbits=31)
    return c.compress(data) + c.flush()


def test_the_byte_cap_holds_on_decoded_bytes(net):
    """A2: a small gzip body inflating far past max_bytes is refused with bounded
    memory; a br / zstd body is refused before a byte is decoded; a normal gzip
    page decodes. (At 98ebc5b a 241-byte brotli body expanded to 300 MB.)"""
    import tracemalloc
    import zlib

    cap = 1_000_000
    c = zlib.compressobj(wbits=31)
    bomb = b"".join(c.compress(b"\0" * (1 << 20)) for _ in range(200)) + c.flush()   # 200 MiB inflated
    assert len(bomb) < cap // 4
    net.routes["https://big.example/bomb"] = lambda r: httpx.Response(
        200, headers={"content-type": "text/html", "content-encoding": "gzip"}, content=iter([bomb]))
    tracemalloc.start()
    try:
        res = _fetch({"url": "https://big.example/bomb"}, cfg=FetchConfig(max_bytes=cap))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert not res.ok and "too large" in res.error, res.error
    assert peak < 8 * cap, f"peak {peak / 1e6:.1f} MB under a {cap / 1e6:.0f} MB cap"

    pulled: list[bytes] = []

    def unread(data: bytes):
        def gen():
            pulled.append(data)
            yield data
        return gen()

    for coding in ("br", "zstd"):
        net.routes[f"https://big.example/{coding}"] = lambda r, coding=coding: httpx.Response(
            200, headers={"content-type": "text/html", "content-encoding": coding},
            content=unread(b"\x1b\xff" * 64))
        res = _fetch({"url": f"https://big.example/{coding}"}, cfg=FetchConfig(max_bytes=cap))
        assert not res.ok and f"Content-Encoding '{coding}' refused" in res.error, res.error
    assert pulled == [], "a refused encoding is never read, let alone decoded"

    net.routes["https://docs.example.org/gz"] = lambda r: httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8", "content-encoding": "gzip"},
        content=iter([_gzip(ARTICLE.encode())]))
    res = _fetch({"url": "https://docs.example.org/gz"})
    assert res.ok and "pauses a graph at a node" in _text(res), res.error


def test_a_backslash_url_reaches_every_rung_the_whatwg_way(net, monkeypatch, tmp_path):
    """A4: the host every rung sees — intel_* included — is the host the guard
    checked: a backslash is read as "/" before the host is taken."""
    from argus.tools import SnatchResult
    md = tmp_path / "s.md"
    md.write_text("Snatched page text about interrupts. " * 20, encoding="utf-8")
    given: list[str] = []

    def snatch(url, **kw):
        given.append(url)
        return SnatchResult(ok=True, url=url, markdown_path=str(md), title="S")

    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", True)
    monkeypatch.setattr(fetchers, "snatch_url", snatch)
    cfg = FetchConfig(chain=["intel_snatch"])
    res = _fetch({"url": "http://127.0.0.1\\@evil.example/"}, cfg=cfg)
    assert res.error.startswith("refused: private address (127.0.0.1"), res.error
    assert given == [], "the intel rung never ran"
    res = _fetch({"url": "https://public.example\\docs\\page?q=a\\b"}, cfg=cfg)
    assert res.ok and given == ["https://public.example/docs/page?q=a\\b"], "the query keeps its backslash"


def test_expected_rung_failures_are_typed_and_anything_else_propagates(net, monkeypatch):
    """A6: network errors and a third-party parser's exceptions are the rung's own
    failure, recorded with the exception type; an unexpected exception propagates."""
    def refused(request):
        raise httpx.ConnectError("connection refused", request=request)

    net.routes["https://down.example/a"] = refused
    res = _fetch({"url": "https://down.example/a"})
    assert res.rungs_tried == [{"rung": "native", "error": "ConnectError: connection refused"}]

    net.routes["https://site.example/broken.pdf"] = httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 not really a pdf")
    res = _fetch({"url": "https://site.example/broken.pdf"})
    assert res.error.startswith("native: pdf: ") and not res.ok, res.error

    net.routes["https://docs.example.org/p"] = httpx.Response(200, html=ARTICLE)

    def broken_extract(*a, **kw):
        raise ValueError("lxml choked")

    monkeypatch.setattr(fetchers.trafilatura, "extract", broken_extract)
    res = _fetch({"url": "https://docs.example.org/p"})
    assert res.error == "native: trafilatura: ValueError: lxml choked"

    def bug(*a, **kw):
        raise KeyError("a bug, not a fetch failure")

    monkeypatch.setattr(fetchers, "_page", bug)
    with pytest.raises(KeyError):
        _fetch({"url": "https://docs.example.org/p"})


def test_an_html_response_is_html_whatever_its_path(net):
    """A7: a text/html response at a .pdf path goes to trafilatura, not pypdf."""
    net.routes["https://site.example/landing.pdf"] = httpx.Response(200, html=ARTICLE)
    res = _fetch({"url": "https://site.example/landing.pdf"})
    assert res.ok and "pauses a graph at a node" in _text(res), res.error


def test_a_pinned_source_counts_its_exa_contents_page_and_never_takes(net, monkeypatch):
    """C2 (the reviewer's pinned_contents probe): every budget spent, native 403 —
    a pinned /append URL is still fetched through exa_contents; the page is
    counted, never taken (at 98ebc5b take() refused it: "page budget spent")."""
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    net.routes["https://walled.example/mine"] = httpx.Response(403)
    net.routes["https://api.exa.ai/contents"] = httpx.Response(200, json={
        "results": [{"text": "Walled text. " * 80, "title": "W"}], "costDollars": {"total": 0.001}})
    taken: list[str] = []

    class Ledger(SpendLedger):
        def take(self, kind, n=1):
            taken.append(kind)
            return super().take(kind, n)

    ledger = Ledger({"fetch_extend": 0, "exa_contents_pages": 0})
    pinned = {"url": "https://walled.example/mine", "title": "mine", "source": "appended-url"}
    fetched, errors = research._parallel_fetch([pinned], cfg=FetchConfig(), ledger=ledger, exa_on=True,
                                               kind="fetch_extend")
    assert errors == [] and taken == [], "a pinned source never calls take()"
    (item,) = fetched
    assert (item["url"], item["fetcher"], item["pinned"]) == (pinned["url"], "exa_contents", True)
    assert item["rungs_tried"] == [{"rung": "native", "error": "HTTP 403"}, {"rung": "exa_contents", "error": ""}]
    snap = ledger.snapshot()
    assert snap["exa_contents_pages"] == 1 and snap["exa_usd"] == pytest.approx(0.001), "counted, so it is visible"


# ---------------------------------------------------------------------------
# WP-B fix round 2
# ---------------------------------------------------------------------------

def test_own_endpoints_are_matched_by_host_and_port_never_by_address(net, monkeypatch):
    """§2.8: a public endpoint's address is shared (a CDN, a tunnel edge), so only its
    normalised name and port are refused; a private endpoint is refused by the
    private-address rule whatever reaches it. (At da8762f every site on the
    endpoint's address and port was refused.)"""
    net.dns.update({"proxy.example.net": [PUBLIC_IP], "neighbour.example.org": [PUBLIC_IP]})
    monkeypatch.setenv("FREELLMAPI_BASE_URL", "https://proxy.example.net:3001/v1")
    for url in ("https://proxy.example.net.:3001/admin", "https://PROXY.Example.NET:3001/admin"):
        res = _fetch({"url": url})
        assert res.error == "refused: private address (own endpoint proxy.example.net:3001)", (url, res.error)
    assert net.requests == []

    for url in ("https://neighbour.example.org:3001/page", f"https://{PUBLIC_IP}:3001/page"):
        net.routes[url] = httpx.Response(200, html=ARTICLE)
        res = _fetch({"url": url})
        assert res.ok, (url, res.error)
    assert [r.url.host for r in net.requests] == [PUBLIC_IP, PUBLIC_IP], "the endpoint's own address and port"

    monkeypatch.setenv("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
    res = _fetch({"url": "http://127.0.0.1:3001/v1/models"})
    assert res.error == "refused: private address (127.0.0.1 → 127.0.0.1)", res.error


@pytest.fixture
def fake_proxy():
    """A loopback HTTP proxy recording each request line and, after a ``CONNECT``,
    the TLS ClientHello that follows; then it hangs up, so nothing leaves the machine."""
    import socket
    import threading

    seen: list[tuple[str, bytes]] = []
    server = socket.create_server(("127.0.0.1", 0))

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:                        # the server socket closed: the test is over
                return
            with conn:
                head = b""
                while b"\r\n\r\n" not in head and (chunk := conn.recv(4096)):
                    head += chunk
                line, hello = head.split(b"\r\n", 1)[0].decode(), b""
                if line.startswith("CONNECT "):
                    conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    while len(hello) < 5 or len(hello) < 5 + int.from_bytes(hello[3:5], "big"):
                        if not (chunk := conn.recv(4096)):
                            break
                        hello += chunk
                seen.append((line, hello))

    threading.Thread(target=serve, daemon=True).start()
    yield server.getsockname()[1], seen
    server.close()


def _real_transport_offline(monkeypatch, tmp_path, dns: dict[str, list[str]]) -> list[tuple]:
    """The real httpx transport; every connection but to loopback refused (and
    recorded), ``_resolve`` answering from ``dns``, no proxy from the environment."""
    import socket

    real_connect = socket.create_connection
    connects: list[tuple] = []

    def connect(address, *a, **kw):
        connects.append(address)
        if address[0] != "127.0.0.1":
            raise ConnectionRefusedError("hermetic: no connection leaves the machine")
        return real_connect(address, *a, **kw)

    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr(fetchers, "_resolve", lambda host: dns.get(host, [PUBLIC_IP]))
    monkeypatch.setattr(fetchers, "_scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", False)
    for env in ("EXA_API_KEY", "FREELLMAPI_BASE_URL", "ARGUS_WEBAPP_PUBLIC_URL",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(env, raising=False)
    return connects


def test_through_an_environment_proxy_the_hop_goes_by_name(monkeypatch, tmp_path, fake_proxy):
    """§2.8: when an HTTP(S) proxy from the environment applies (httpx's trust_env
    selection), the hop goes to it by name — the proxy resolves, so the IP pin cannot
    apply — and TLS starts for the name. The guard still runs first, on our own
    resolution. (At da8762f the hop went as an IP URL: ``CONNECT <address>``, and TLS
    was verified against the address.)"""
    port, seen = fake_proxy
    connects = _real_transport_offline(monkeypatch, tmp_path, {"inner.example": ["10.0.0.9"]})
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")

    res = _fetch({"url": "https://docs.example.org/page"})
    assert [line for line, _ in seen] == ["CONNECT docs.example.org:443 HTTP/1.1"], seen
    assert b"docs.example.org" in seen[0][1], "the ClientHello names the host (SNI), as no IP URL would"
    assert connects == [("127.0.0.1", port)], "only the proxy is connected to, never the checked address"
    assert not res.ok and res.error.startswith("native: ConnectError"), res.error    # the fake proxy hangs up

    seen.clear()
    connects.clear()
    res = _fetch({"url": "https://inner.example/x"})
    assert res.error == "refused: private address (inner.example → 10.0.0.9)", res.error
    assert seen == [] and connects == [], "the guard runs first, on our own resolution"


def test_no_proxy_covering_the_host_keeps_the_ip_pin(monkeypatch, tmp_path, fake_proxy):
    """A host ``NO_PROXY`` covers goes to the checked address and never through the
    proxy — not even as the IP URL, which ``NO_PROXY``'s name does not cover."""
    port, seen = fake_proxy
    connects = _real_transport_offline(monkeypatch, tmp_path, {})
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("NO_PROXY", "docs.example.org")

    res = _fetch({"url": "https://docs.example.org/page"})
    assert connects == [(PUBLIC_IP, 443)], "the pinned connection to the checked address"
    assert seen == [], "the proxy is never asked"
    assert not res.ok and res.error.startswith("native: ConnectError"), res.error
