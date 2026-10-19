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
    ``request → Response``; ``dns[host]`` the addresses a name resolves to."""

    def __init__(self):
        self.routes: dict = {}
        self.dns: dict[str, list[str]] = {}
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.routes.get(str(request.url))
        if route is None:
            return httpx.Response(404, text="not found")
        return route(request) if callable(route) else route

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]


@pytest.fixture
def net(monkeypatch, tmp_path):
    n = _Net()
    monkeypatch.setattr(fetchers, "_http_client", lambda cfg: httpx.Client(
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
    assert (res.ok, res.fetcher, res.rungs_tried) == (True, "native", ["native"])
    text = _text(res)
    assert "pauses a graph at a node" in text
    assert "Navigation menu boilerplate" not in text, "trafilatura keeps the article, not the chrome"
    assert res.chars == len(text) and res.title
    assert res.ms >= 0

    # 2. a source no rung can fetch is reported with the rungs tried, never dropped
    net.routes["https://dead.example/x"] = httpx.Response(503)
    res = _fetch({"url": "https://dead.example/x"}, cfg=cfg)
    assert (res.ok, res.rungs_tried) == (False, ["native"])
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
    assert (res.ok, res.fetcher, res.rungs_tried) == (True, "exa_text", ["exa_text"])
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
    assert (res.ok, res.fetcher, res.rungs_tried) == (True, "exa_contents", ["native", "exa_contents"])
    assert "Walled text." in _text(res)
    snap = ledger.snapshot()
    assert snap["exa_contents_pages"] == 1 and snap["exa_usd"] == pytest.approx(0.001)

    res = _fetch({"url": "https://walled.example/a"}, ledger=ledger, exa=True)
    assert not res.ok and res.rungs_tried == ["native", "exa_contents"]
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
    assert (res.ok, res.fetcher, res.rungs_tried) == (True, "intel_snatch", ["native", "intel_snatch"])
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
    assert item["authority_prior"] == 3, "an official doc gets the top prior"
    assert item["sub_qs"] == [1] and item["provider"] == "exa"
