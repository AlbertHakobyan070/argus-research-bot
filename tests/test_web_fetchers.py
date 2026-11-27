"""The free-web fetchers and what uses them: the scrapling / jina / browser
rungs and Office conversion (engines/fetchers.py), the "free" web mode
(RunConfig, the plan knob, /research /free), /read and its link buttons
(bot/read.py), the /engines fetcher lines, and the intel-stack crawl fix.

Hermetic: an httpx MockTransport and a patched ``_resolve`` stand in for the
network and DNS, and fake modules for Scrapling, curl_cffi, Playwright and
MarkItDown — none of the optional install groups is needed (CI has none)."""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from argus import tools
from argus.bot import engines_cmd, read, research as bot_research, runs
from argus.engines import fetchers
from argus.graph import anchors
from argus.graph.run_config import FREE_FETCH_CHAIN, FREE_PROVIDERS, FetchConfig, RunConfig

from helpers import as_stream, fake_engines_snapshot, request_url

PUBLIC_IP = "93.184.216.34"
ARTICLE = ("<html><head><title>Rendered article</title></head><body><article><h1>Rendered article</h1>"
           + "".join(f"<p>Paragraph {i} explains how the interrupt pauses the graph and how the run resumes "
                     f"with a command carrying the value the node receives.</p>" for i in range(8))
           + "</article></body></html>")


class _Net:
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


@pytest.fixture
def net(monkeypatch, tmp_path):
    n = _Net()
    monkeypatch.setattr(fetchers, "_http_client", lambda cfg, **_: httpx.Client(
        transport=httpx.MockTransport(n.handler), timeout=cfg.timeout_s))
    monkeypatch.setattr(fetchers, "_resolve", lambda host: n.dns.get(host, [PUBLIC_IP]))
    monkeypatch.setattr(fetchers, "_scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", False)
    for env in ("EXA_API_KEY", "FREELLMAPI_BASE_URL", "ARGUS_WEBAPP_PUBLIC_URL", "JINA_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(fetchers, "_sleep", lambda s: None)
    return n


def _fetch(url: str, chain: list[str]) -> fetchers.FetchResult:
    return fetchers.fetch({"url": url, "title": ""}, cfg=FetchConfig(chain=chain), ledger=None, exa_enabled=False,
                          pinned=True)


# ---------------------------------------------------------------------------
# availability, dispatch
# ---------------------------------------------------------------------------

def test_available_chain_drops_rungs_this_install_cannot_run(monkeypatch):
    monkeypatch.setattr(fetchers, "SCRAPLING_AVAILABLE", False)
    monkeypatch.setattr(fetchers, "BROWSER_AVAILABLE", False)
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", True)
    assert fetchers.available_chain(FREE_FETCH_CHAIN) == ["exa_text", "native", "jina", "intel_crawl"]
    missing = fetchers.missing_rungs(FREE_FETCH_CHAIN)
    assert [m.split(":", 1)[0] for m in missing] == ["scrapling", "browser"]
    assert fetchers.INSTALL["web"] in missing[0] and fetchers.INSTALL["chromium"] in missing[1]
    monkeypatch.setattr(fetchers, "SCRAPLING_AVAILABLE", True)
    monkeypatch.setattr(fetchers, "BROWSER_AVAILABLE", True)
    assert fetchers.available_chain(FREE_FETCH_CHAIN) == FREE_FETCH_CHAIN and not fetchers.missing_rungs(FREE_FETCH_CHAIN)


def test_an_unknown_rung_raises_instead_of_running_an_intel_script(net):
    with pytest.raises(ValueError, match="unknown fetch rung 'bogus'"):
        fetchers.fetch({"url": "https://example.com/a"}, cfg=FetchConfig.model_construct(chain=["bogus"]),
                       ledger=None, exa_enabled=False)


def test_the_exa_mode_default_chain_is_unchanged():
    assert FetchConfig().chain == ["exa_text", "native", "exa_contents", "intel_snatch", "intel_crawl",
                                   "intel_stealth"]


def test_status_names_what_each_missing_rung_needs(monkeypatch):
    monkeypatch.setattr(fetchers, "SCRAPLING_AVAILABLE", False)
    monkeypatch.setattr(fetchers, "BROWSER_AVAILABLE", False)
    monkeypatch.setattr(fetchers, "MARKITDOWN_AVAILABLE", False)
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    rows = {r["rung"]: r for r in fetchers.status()}
    assert rows["scrapling"]["ok"] is False and fetchers.INSTALL["web"] in rows["scrapling"]["detail"]
    assert rows["browser"]["ok"] is False and fetchers.INSTALL["chromium"] in rows["browser"]["detail"]
    assert fetchers.INSTALL["convert"] in rows["native"]["detail"]
    assert rows["exa_contents"] == {"rung": "exa_contents", "ok": False, "detail": "needs EXA_API_KEY"}
    assert "keyless" in rows["jina"]["detail"] and rows["jina"]["ok"] is True


# ---------------------------------------------------------------------------
# scrapling
# ---------------------------------------------------------------------------

def _page(status=200, body=b"", headers=None, encoding="utf-8"):
    return SimpleNamespace(status=status, body=body, headers=headers or {}, encoding=encoding)


def test_scrapling_pins_each_hop_and_follows_redirects_through_the_guard(net, monkeypatch):
    calls = []
    pages = {"http://example.com/a": _page(301, headers={"Location": "https://www.example.com/a"}),
             "https://www.example.com/a": _page(200, ARTICLE.encode(), {"Content-Type": "text/html"})}

    def get(url, pin, cfg):
        calls.append((url, pin))
        return pages[url]
    monkeypatch.setattr(fetchers, "_scrapling_get", get)
    net.dns["www.example.com"] = ["2606:2800:220:1::1", PUBLIC_IP]
    res = _fetch("http://example.com/a", ["scrapling"])
    assert res.ok and res.fetcher == "scrapling" and "interrupt pauses the graph" in Path(res.markdown_path).read_text()
    assert calls == [("http://example.com/a", f"example.com:80:{PUBLIC_IP}"),
                     ("https://www.example.com/a", f"www.example.com:443:[2606:2800:220:1::1],{PUBLIC_IP}")]


def test_scrapling_sends_curl_the_host_its_pin_names(net, monkeypatch):
    """Capitals, a trailing dot, user info, a fragment: curl gets the host as the
    guard read it, so the RESOLVE entry matches and curl never looks it up itself."""
    calls = []
    pages = {"http://example.com:8080/a?q=1": _page(302, headers={"Location": "https://WWW.Example.com./b"}),
             "https://www.example.com/b": _page(200, ARTICLE.encode(), {"Content-Type": "text/html"})}

    def get(url, pin, cfg):
        calls.append((url, pin))
        return pages[url]
    monkeypatch.setattr(fetchers, "_scrapling_get", get)
    assert _fetch("http://user:pw@Example.COM.:8080/a?q=1#top", ["scrapling"]).ok
    assert calls == [("http://example.com:8080/a?q=1", f"example.com:8080:{PUBLIC_IP}"),
                     ("https://www.example.com/b", f"www.example.com:443:{PUBLIC_IP}")]


def test_scrapling_refuses_a_redirect_to_a_private_address(net, monkeypatch):
    monkeypatch.setattr(fetchers, "_scrapling_get", lambda url, pin, cfg: _page(
        302, headers={"location": "http://internal.example/admin"}))
    net.dns["internal.example"] = ["10.0.0.7"]
    res = _fetch("https://example.com/a", ["scrapling", "jina"])
    assert not res.ok and res.error.startswith("refused: private address (internal.example")
    assert [t["rung"] for t in res.rungs_tried] == ["scrapling"]          # the chain stops


def test_scrapling_http_error_moves_to_the_next_rung(net, monkeypatch):
    monkeypatch.setattr(fetchers, "_scrapling_get", lambda url, pin, cfg: _page(403, b"denied"))
    res = _fetch("https://example.com/a", ["scrapling"])
    assert not res.ok and res.rungs_tried == [{"rung": "scrapling", "error": "HTTP 403"}]


def test_scrapling_get_sets_the_pin_and_cap_on_its_curl_session(monkeypatch):
    seen = {}

    class CurlError(Exception):
        pass

    class Session:                                   # what Scrapling 0.5's FetcherSession yields
        def __init__(self):
            self._curl_session = SimpleNamespace(curl_options={})

        def get(self, url):
            seen["options"], seen["url"] = dict(self._curl_session.curl_options), url
            return _page(200, b"<html></html>")

    class FetcherSession:
        def __init__(self, **kw):
            seen["kw"] = kw

        def __enter__(self):
            return Session()

        def __exit__(self, *a):
            return False

    monkeypatch.setitem(sys.modules, "curl_cffi", SimpleNamespace(CurlOpt=SimpleNamespace(RESOLVE="R",
                                                                                         MAXFILESIZE_LARGE="M")))
    monkeypatch.setitem(sys.modules, "curl_cffi.curl", SimpleNamespace(CurlError=CurlError))
    monkeypatch.setitem(sys.modules, "scrapling.fetchers", SimpleNamespace(FetcherSession=FetcherSession))
    cfg = FetchConfig(timeout_s=12, max_bytes=1000)
    fetchers._scrapling_get("https://example.com/a", "example.com:443:1.2.3.4", cfg)
    assert seen["kw"] == {"follow_redirects": False, "retries": 1, "timeout": 12}
    assert seen["options"] == {"R": ["example.com:443:1.2.3.4"], "M": 1000} and seen["url"] == "https://example.com/a"

    class Bare(Session):
        def __init__(self):
            self._curl_session = None
    FetcherSession.__enter__ = lambda self: Bare()
    with pytest.raises(fetchers._RungError, match="no curl session to pin"):
        fetchers._scrapling_get("https://example.com/a", "pin", cfg)


# ---------------------------------------------------------------------------
# jina
# ---------------------------------------------------------------------------

def _jina_route(net, reply: dict, status: int = 200):
    def handler(request):
        net.jina = request
        return httpx.Response(status, json=reply)
    net.routes["https://r.jina.ai/https://example.com/page"] = handler


def test_jina_reads_the_json_reply_keyless_and_spaced(net, monkeypatch):
    _jina_route(net, {"code": 200, "data": {"title": "Rendered", "content": "x" * 400, "httpStatus": "200"}})
    waits = []
    monkeypatch.setattr(fetchers, "_sleep", waits.append)
    monkeypatch.setattr(fetchers, "_jina_last", [time.monotonic()])        # a keyless request just went out
    res = _fetch("https://example.com/page", ["jina"])
    assert res.ok and res.fetcher == "jina" and res.title == "Rendered" and res.chars == 400
    assert net.jina.headers["accept"] == "application/json" and "authorization" not in net.jina.headers
    assert len(waits) == 1 and 3.0 < waits[0] <= fetchers.JINA_KEYLESS_INTERVAL_S


def test_jina_with_a_key_sends_it_and_does_not_wait(net, monkeypatch):
    _jina_route(net, {"data": {"title": "t", "content": "y" * 400}})
    monkeypatch.setenv("JINA_API_KEY", "k-123")
    waits = []
    monkeypatch.setattr(fetchers, "_sleep", waits.append)
    monkeypatch.setattr(fetchers, "_jina_last", [time.monotonic()])
    assert _fetch("https://example.com/page", ["jina"]).ok
    assert net.jina.headers["authorization"] == "Bearer k-123" and waits == []


@pytest.mark.parametrize("reply,status,error", [
    ({"data": {"content": "z" * 400, "httpStatus": "404"}}, 200, "the site answered Jina with HTTP 404"),
    ({"code": 451, "name": "SecurityCompromiseError"}, 200, "jina: malformed reply, no data object (code 451)"),
    ({"code": 403}, 403, "HTTP 403"),
])
def test_jina_failures_are_its_own_errors(net, reply, status, error):
    _jina_route(net, reply, status)
    res = _fetch("https://example.com/page", ["jina"])
    assert not res.ok and res.rungs_tried == [{"rung": "jina", "error": error}]


def test_get_headers_never_follow_a_redirect_to_another_host(net):
    net.routes["https://r.jina.ai/x"] = httpx.Response(302, headers={"location": "https://other.example/y"})
    net.routes["https://other.example/y"] = httpx.Response(200, text="ok")
    with httpx.Client(transport=httpx.MockTransport(net.handler)) as client:
        fetchers._get(client, "https://r.jina.ai/x", FetchConfig(), headers={"Authorization": "Bearer k"})
    first, second = net.requests
    assert first.headers["authorization"] == "Bearer k" and "authorization" not in second.headers


# ---------------------------------------------------------------------------
# browser (a fake Playwright; the gate on real local sockets)
# ---------------------------------------------------------------------------

class _PWError(Exception):
    pass


class _PWTimeout(_PWError):
    pass


def _socks(port: int, host: str, dport: int) -> tuple[socket.socket, int | None]:
    """A SOCKS5 CONNECT to ``host:dport`` through the gate on ``port``, as Chromium
    makes it: the socket and the reply code (None: closed without one)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"\x05\x01\x00")
    assert fetchers._recv_exact(s, 2) == b"\x05\x00"
    name = host.encode()
    s.sendall(b"\x05\x01\x00\x03" + bytes([len(name)]) + name + dport.to_bytes(2, "big"))
    reply = s.recv(10)
    return s, (reply[1] if reply else None)


def _playwright(monkeypatch, *, requests, status=200, html=ARTICLE, launch_error=None, goto_error=None,
                content_error=None, via_gate=()):
    """A fake ``playwright.sync_api``: ``goto`` announces ``requests`` — ``(url,
    resource_type, redirect)`` — to the page's request listeners and routes the
    non-redirect ones through the context's route handler, recording each
    outcome; it connects to each host of ``via_gate`` through the launch's SOCKS5
    proxy, as Chromium would; then it raises ``goto_error`` when given.
    ``launched`` gets the launch's keyword arguments."""
    routed: list[tuple[str, str]] = []
    launched: dict = {}

    class Route:
        def __init__(self, request):
            self.request = request

        def abort(self, code):
            routed.append((self.request.url, "abort"))

        def continue_(self):
            routed.append((self.request.url, "continue"))

    class Page:
        url = "https://example.com/final"

        def __init__(self, context):
            self.context, self.listeners = context, []

        def on(self, event, fn):
            assert event == "request"
            self.listeners.append(fn)

        def goto(self, url, wait_until, timeout):
            for u, kind, redirect in requests:
                req = SimpleNamespace(url=u, resource_type=kind, redirected_from=object() if redirect else None)
                for fn in self.listeners:
                    fn(req)
                if not redirect:
                    self.context.handler(Route(req))
            gate_port = int(launched["proxy"]["server"].rsplit(":", 1)[1])
            for host in via_gate:
                _socks(gate_port, host, 443)[0].close()
            if goto_error:
                raise goto_error
            return SimpleNamespace(status=status)

        def wait_for_load_state(self, state, timeout):
            raise _PWTimeout("never idle")

        def content(self):
            if content_error:
                raise content_error
            return html

        def title(self):
            return "Rendered article"

    class Context:
        handler = None

        def route(self, pattern, handler):
            assert pattern == "**/*"
            self.handler = handler

        def new_page(self):
            return Page(self)

    class Browser:
        closed = False

        def new_context(self, **kw):
            assert kw == {"service_workers": "block"}
            return Context()

        def close(self):
            Browser.closed = True

    class PW:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(launch=self.launch))

        def __exit__(self, *a):
            return False

        def launch(self, **kw):
            launched.update(kw)
            if launch_error:
                raise _PWError(launch_error)
            return Browser()

    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(
        Error=_PWError, TimeoutError=_PWTimeout, sync_playwright=PW))
    return routed, Browser, launched


def test_browser_reads_the_page_and_blocks_private_and_heavy_requests(net, monkeypatch):
    net.dns["metadata.internal"] = ["169.254.169.254"]
    routed, browser, launched = _playwright(monkeypatch, requests=[
        ("https://example.com/page", "document", False), ("https://cdn.example.com/a.js", "script", False),
        ("http://metadata.internal/latest", "xhr", False), ("https://cdn.example.com/hero.jpg", "image", False),
        ("data:text/plain,hi", "fetch", False)])
    res = _fetch("https://example.com/page", ["browser"])
    assert res.ok and res.fetcher == "browser" and res.title == "Rendered article"
    assert routed == [("https://example.com/page", "continue"), ("https://cdn.example.com/a.js", "continue"),
                      ("http://metadata.internal/latest", "abort"), ("https://cdn.example.com/hero.jpg", "abort"),
                      ("data:text/plain,hi", "continue")]
    assert browser.closed


def test_browser_sends_every_connection_through_the_gate_and_closes_it(net, monkeypatch):
    _routed, _browser, launched = _playwright(monkeypatch, requests=[("https://example.com/page", "document", False)])
    assert _fetch("https://example.com/page", ["browser"]).ok
    server = launched["proxy"]["server"]
    assert server.startswith("socks5://127.0.0.1:") and launched["proxy"]["bypass"] == "<-loopback>"
    assert launched["args"] == ["--disable-quic", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"]
    assert launched["chromium_sandbox"] is True      # Playwright's default would pass --no-sandbox
    with pytest.raises(OSError):                     # the rung closed its gate
        socket.create_connection(("127.0.0.1", int(server.rsplit(":", 1)[1])), timeout=2)


def test_browser_discards_a_page_reached_through_a_private_redirect(net, monkeypatch):
    net.dns["router.lan"] = ["192.168.1.1"]
    _playwright(monkeypatch, requests=[("https://example.com/page", "document", False),
                                       ("http://router.lan/", "document", True)])
    res = _fetch("https://example.com/page", ["browser", "jina"])
    assert not res.ok and res.error.startswith("refused: private address (router.lan")
    assert "a redirect in the browser" in res.error and [t["rung"] for t in res.rungs_tried] == ["browser"]


def test_a_refused_redirect_wins_over_the_navigation_failing(net, monkeypatch):
    """The gate refuses the hop, so the navigation fails: still a refusal, and the
    chain stops (intel_crawl would not check the hop)."""
    net.dns["router.lan"] = ["192.168.1.1"]
    _playwright(monkeypatch, requests=[("https://example.com/page", "document", False),
                                       ("http://router.lan/", "document", True)],
                goto_error=_PWError("net::ERR_SOCKS_CONNECTION_FAILED at https://example.com/page"))
    res = _fetch("https://example.com/page", ["browser", "jina"])
    assert not res.ok and res.error.startswith("refused: private address (router.lan")
    assert [t["rung"] for t in res.rungs_tried] == ["browser"]


def test_a_playwright_error_after_the_page_loaded_is_the_rungs_failure(net, monkeypatch):
    _r, browser, _l = _playwright(monkeypatch, requests=[("https://example.com/page", "document", False)],
                                  content_error=_PWError("Unable to retrieve content because the page is navigating "
                                                         "and changing the content.\n  call log"))
    res = _fetch("https://example.com/page", ["browser"])
    assert res.rungs_tried == [{"rung": "browser", "error": "browser: Unable to retrieve content because the page is "
                                                           "navigating and changing the content."}]
    assert browser.closed


def test_browser_raises_what_the_gate_could_not_check(net, monkeypatch):
    def check(url):
        if "broken.example" in url:
            raise fetchers.EngineConfigError("FREELLMAPI_BASE_URL (host 'x') is not a valid endpoint URL")
        return [PUBLIC_IP]
    monkeypatch.setattr(fetchers, "check_target", check)
    _playwright(monkeypatch, requests=[("https://example.com/page", "document", False)], via_gate=["broken.example"])
    with pytest.raises(fetchers.EngineConfigError, match="FREELLMAPI_BASE_URL"):
        _fetch("https://example.com/page", ["browser"])


def test_browser_without_chromium_says_how_to_install_it(net, monkeypatch):
    _playwright(monkeypatch, requests=[], launch_error="Executable doesn't exist at C:\\x\\chrome.exe\n  long log")
    res = _fetch("https://example.com/page", ["browser"])
    assert not res.ok
    assert res.rungs_tried == [{"rung": "browser", "error": "browser: Chromium did not start (Executable doesn't exist "
                                                           f"at C:\\x\\chrome.exe) — {fetchers.INSTALL['chromium']}"}]


def test_browser_http_error_is_the_rungs_failure(net, monkeypatch):
    _playwright(monkeypatch, requests=[("https://example.com/page", "document", False)], status=404)
    assert _fetch("https://example.com/page", ["browser"]).rungs_tried == [{"rung": "browser", "error": "HTTP 404"}]


@pytest.fixture
def upstream():
    """A local server: it reads what a connection sends, answers ``state["reply"]``
    and closes; ``state["seen"]`` keeps what each connection sent."""
    srv = socket.create_server(("127.0.0.1", 0))
    state = {"reply": b"pong", "seen": []}

    def serve():
        while True:
            try:
                conn, _addr = srv.accept()
            except OSError:
                return
            with conn:
                state["seen"].append(conn.recv(100))
                conn.sendall(state["reply"])
    threading.Thread(target=serve, daemon=True).start()
    yield srv.getsockname()[1], state
    srv.close()


def test_the_gate_checks_each_host_and_connects_to_the_checked_address(monkeypatch, upstream):
    port, state = upstream
    checked = []

    def check(url):
        checked.append(url)
        if "private.example" in url:
            raise fetchers._Refused("refused: private address (private.example → 10.0.0.7)")
        if "nowhere.example" in url:
            raise fetchers._RungError("dns: nowhere.example did not resolve")
        return ["127.0.0.1"]            # the checked address: the gate connects there, no lookup of its own
    monkeypatch.setattr(fetchers, "check_target", check)
    gate = fetchers._Gate(FetchConfig(timeout_s=5))
    try:
        s, code = _socks(gate.port, "public.example", port)
        assert code == 0
        s.sendall(b"ping")
        assert fetchers._recv_exact(s, 4) == b"pong"
        s.close()
        for host, expected in (("private.example", 2), ("nowhere.example", 4)):
            s, code = _socks(gate.port, host, 80)
            s.close()
            assert code == expected
    finally:
        gate.close()
    assert state["seen"] == [b"ping"] and gate.failures == []
    assert gate.refused == ["refused: private address (private.example → 10.0.0.7)"]
    assert checked == [f"http://public.example:{port}/", "http://private.example:80/", "http://nowhere.example:80/"]


def test_the_gate_stops_past_its_byte_cap(monkeypatch, upstream):
    port, state = upstream
    state["reply"] = b"x" * 5000
    monkeypatch.setattr(fetchers, "check_target", lambda url: ["127.0.0.1"])
    gate = fetchers._Gate(FetchConfig(timeout_s=5, max_bytes=1000))     # 4000 bytes in all
    got = b""
    try:
        s, _code = _socks(gate.port, "public.example", port)
        s.sendall(b"go")
        try:
            while chunk := s.recv(65536):
                got += chunk
        except ConnectionResetError:
            pass                        # Windows resets a socket closed with bytes unread
        s.close()
    finally:
        gate.close()
    assert gate.too_large and len(got) <= 4000


def test_the_gate_keeps_an_error_it_cannot_judge_for_the_rung(monkeypatch):
    def check(url):
        raise fetchers.EngineConfigError("FREELLMAPI_BASE_URL (host 'x') is not a valid endpoint URL")
    monkeypatch.setattr(fetchers, "check_target", check)
    gate = fetchers._Gate(FetchConfig(timeout_s=5))
    try:
        s, code = _socks(gate.port, "public.example", 443)
        s.close()
    finally:
        gate.close()
    assert code is None and [type(e) for e in gate.failures] == [fetchers.EngineConfigError]


# ---------------------------------------------------------------------------
# Office files (MarkItDown)
# ---------------------------------------------------------------------------

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _zip(parts: dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def markitdown(monkeypatch):
    seen = {}

    class MarkItDown:
        def __init__(self, enable_plugins):
            assert enable_plugins is False

        def convert_stream(self, stream, file_extension):
            seen["bytes"], seen["ext"] = stream.read(), file_extension
            return SimpleNamespace(text_content="Word document text. " * 20)
    monkeypatch.setitem(sys.modules, "markitdown", SimpleNamespace(MarkItDown=MarkItDown))
    monkeypatch.setattr(fetchers, "MARKITDOWN_AVAILABLE", True)
    return seen


@pytest.mark.parametrize("url,ctype,suffix", [
    ("https://x.org/f", DOCX, ".docx"), ("https://x.org/deck.pptx", "application/octet-stream", ".pptx"),
    ("https://x.org/sheet.xlsx?dl=1", "", ".xlsx"), ("https://x.org/page.docx", "text/html; charset=utf-8", None),
    ("https://x.org/a.pdf", "application/pdf", None)])
def test_office_suffix(url, ctype, suffix):
    assert fetchers._office_suffix(url, ctype) == suffix


def test_native_reads_a_word_file_through_markitdown(net, markitdown):
    body = _zip({"word/document.xml": b"<w:document/>"})
    net.routes["https://example.com/report.docx"] = httpx.Response(200, content=body, headers={"content-type": DOCX})
    res = _fetch("https://example.com/report.docx", ["native"])
    assert res.ok and res.fetcher == "native" and markitdown == {"bytes": body, "ext": ".docx"}


def test_an_office_zip_bomb_is_refused_before_markitdown(net, markitdown, monkeypatch):
    monkeypatch.setattr(fetchers, "_OFFICE_INFLATED_MAX", 10_000)
    net.routes["https://example.com/bomb.docx"] = httpx.Response(200, content=_zip({"word/document.xml": b"0" * 20_000}))
    res = _fetch("https://example.com/bomb.docx", ["native"])
    assert res.rungs_tried == [{"rung": "native", "error": "a .docx file too large unzipped (> 10000 bytes)"}]
    assert markitdown == {}


def test_an_office_file_that_is_not_a_zip_is_the_rungs_failure(net, markitdown):
    net.routes["https://example.com/fake.xlsx"] = httpx.Response(200, content=b"not a zip at all")
    res = _fetch("https://example.com/fake.xlsx", ["native"])
    assert res.rungs_tried[0]["error"].startswith("a .xlsx file that does not unzip (BadZipFile:")
    assert markitdown == {}


def test_an_office_file_without_markitdown_names_the_install_command(net, monkeypatch):
    monkeypatch.setattr(fetchers, "MARKITDOWN_AVAILABLE", False)
    net.routes["https://example.com/deck.pptx"] = httpx.Response(200, content=b"PK")
    res = _fetch("https://example.com/deck.pptx", ["native"])
    assert res.rungs_tried == [{"rung": "native", "error": "a .pptx file: reading it needs the convert install "
                                                          f"group ({fetchers.INSTALL['convert']})"}]


# ---------------------------------------------------------------------------
# the intel-stack crawl: the page, never crawl.py's index.md summary
# ---------------------------------------------------------------------------

def test_crawl_url_returns_the_page_not_the_index_summary(monkeypatch, tmp_path):
    (tmp_path / "index.md").write_text("# Crawl of https://github.com/x\n\n1 pages saved", encoding="utf-8")
    (tmp_path / "langchain-ai.md").write_text("<!-- source: https://github.com/langchain-ai -->\n\npage", encoding="utf-8")
    monkeypatch.setattr(tools, "_run_script", lambda *a, **k: (0, json.dumps({"ok": True, "folder": str(tmp_path)}), ""))
    res = tools.crawl_url("https://github.com/langchain-ai")
    assert res.ok and Path(res.markdown_path).name == "langchain-ai.md" and len(res.pages) == 1


def test_crawl_url_with_only_the_summary_is_not_ok(monkeypatch, tmp_path):
    (tmp_path / "index.md").write_text("# Crawl of https://x.org\n\n0 pages saved", encoding="utf-8")
    monkeypatch.setattr(tools, "_run_script", lambda *a, **k: (0, json.dumps({"folder": str(tmp_path)}), ""))
    assert tools.crawl_url("https://x.org").ok is False


# ---------------------------------------------------------------------------
# the "free" web mode
# ---------------------------------------------------------------------------

def _rc(web="exa", **overrides) -> RunConfig:
    return RunConfig.from_preset(length="short", engines=fake_engines_snapshot(), overrides=overrides,
                                 origin="bench", web=web)


def test_free_mode_drops_exa_from_search_and_fetch():
    rc = _rc("free")
    assert rc.web == "free" and rc.search.providers_enabled == FREE_PROVIDERS
    assert rc.fetch.chain == FREE_FETCH_CHAIN and "exa_contents" not in rc.fetch.chain
    exa = _rc()
    assert exa.web == "exa" and "exa" in exa.search.providers_enabled and "exa_contents" in exa.fetch.chain


def test_console_overrides_win_over_the_mode_and_survive_a_rebuild():
    rc = _rc("free", **{"fetch.chain": ["native", "jina"]})
    assert rc.fetch.chain == ["native", "jina"]
    back = rc.rebuild(length="medium", engines=rc.engines)
    assert back.web == "free" and back.fetch.chain == ["native", "jina"]
    exa = rc.rebuild(length="medium", engines=rc.engines, web="exa")
    assert exa.web == "exa" and "exa" in exa.search.providers_enabled and exa.fetch.chain == ["native", "jina"]


def test_an_unknown_web_mode_raises():
    with pytest.raises(ValueError, match="unknown web mode 'paid'"):
        _rc("paid")


def test_a_run_config_without_web_reads_as_exa():
    data = _rc().model_dump()
    del data["web"]
    assert RunConfig.model_validate(data).web == "exa"


async def test_the_web_knob_cycles_and_refuses_unknown_values():
    c = runs.Choices(length="short")
    await runs._choose(None, c, "web", "next")
    assert c.web == "free"
    await runs._choose(None, c, "web", "next")
    assert c.web == "exa"
    with pytest.raises(ValueError, match="unknown web mode"):
        await runs._choose(None, c, "web", "paid")


def test_rebuilt_applies_the_web_choice():
    rc = _rc()
    assert runs._rebuilt(runs.Choices(length="short", web="free"), rc, rc.engines).web == "free"


@pytest.mark.parametrize("args,expected", [
    (["/free", "gold", "outlook"], ("free", "short", "gold outlook")),
    (["/length", "long", "/free", "gold"], ("free", "long", "gold")),
    (["/free", "/length", "medium", "gold"], ("free", "medium", "gold")),
    (["gold", "/free"], ("exa", "short", "gold /free")),             # a flag only before the topic
    (["/exa", "gold"], ("exa", "short", "gold")),
])
async def test_research_cmd_reads_the_web_flag(monkeypatch, args, expected):
    started = {}

    async def start_run(**kw):
        started.update(kw)
        return runs.Outcome("started", "", "r1")
    monkeypatch.setattr(bot_research.runs, "start_run", start_run)
    monkeypatch.setattr(bot_research, "_allowed", lambda s, uid: True)
    monkeypatch.setattr(bot_research, "get_settings", lambda: None)
    update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_chat=SimpleNamespace(id=5),
                             message=SimpleNamespace(reply_text=AsyncMock()))
    ctx = SimpleNamespace(args=args, application=SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock())))
    await bot_research.research_cmd(update, ctx)
    assert (started["web"], started["length"], started["topic"]) == expected


# ---------------------------------------------------------------------------
# /read and the pasted-link buttons
# ---------------------------------------------------------------------------

def _bot():
    return SimpleNamespace(send_message=AsyncMock(), send_document=AsyncMock())


def _texts(bot) -> list[str]:
    return [c.kwargs["text"] for c in bot.send_message.await_args_list]


@pytest.fixture
def reader(monkeypatch):
    """read.py with its fetching, model and rich send faked; ``state`` records the calls."""
    state = SimpleNamespace(prompt=None, rich=[], found=[], sources=[], requests=[])

    def read_anchors(request, *, rc, ledger):
        state.requests.append((request, list(rc.fetch.chain), ledger))
        return state.found, state.sources, []

    def complete(role, messages, **kw):
        state.prompt = (role, messages[1].content)
        return SimpleNamespace(text="It pauses the graph [1].")

    async def send_rich(bot, chat_id, html, *, reply_markup=None):
        state.rich.append((html, reply_markup))
    monkeypatch.setattr(read.anchors, "read_anchors", read_anchors)
    monkeypatch.setattr(read.llm, "complete", complete)
    monkeypatch.setattr(read.registry, "active_snapshot", AsyncMock(return_value=fake_engines_snapshot()))
    monkeypatch.setattr(read.tgx, "send_rich", send_rich)
    monkeypatch.setattr(read, "_reading", set())
    return state


def _found(ok=True, fetcher="jina", chars=5000):
    page = {"url": "https://example.com/post", "title": "A post", "kind": "web", "ok": ok, "fetcher": fetcher,
            "error": "" if ok else "native: HTTP 403; jina: HTTP 429", "chars": chars, "pages": []}
    source = {"url": "https://example.com/post", "title": "A post", "text": "The interrupt pauses the graph. " * 50}
    return [page], [source] if ok else []


def test_question_of_strips_the_links():
    assert read.question_of("https://a.org/x  what does it cost? https://b.org") == "what does it cost?"
    assert read.question_of("https://a.org/x") == ""


async def test_read_answers_from_the_page_without_exa_and_attaches_the_text(reader, tmp_path, monkeypatch):
    monkeypatch.setattr(read.tempfile, "gettempdir", lambda: str(tmp_path))
    reader.found, reader.sources = _found()
    bot = _bot()
    await read.read_request(bot, 7, "https://example.com/post how does it pause?")
    ((request, chain, ledger),) = reader.requests
    assert chain == FREE_FETCH_CHAIN and ledger is None and "exa_contents" not in chain
    role, prompt = reader.prompt
    assert role == "read.answer" and prompt.startswith("QUESTION: how does it pause?")
    assert "[1] A post\nURL: https://example.com/post" in prompt
    ((html, keyboard),) = reader.rich
    assert "It pauses the graph" in html and "read via jina · 1 page(s)" in html
    assert "<b>A post</b>" in html or "<strong>A post</strong>" in html
    assert [b.callback_data.split(":")[1] for b in keyboard.inline_keyboard[0]] == ["res"]   # not thin: no browser
    (doc,) = bot.send_document.await_args_list
    assert doc.kwargs["filename"].endswith(".md") and doc.kwargs["caption"] == "Text read from A post"
    assert not read._reading


async def test_a_link_that_could_not_be_read_offers_the_browser(reader, monkeypatch):
    monkeypatch.setattr(read.fetchers, "BROWSER_AVAILABLE", True)
    reader.found, reader.sources = _found(ok=False, fetcher="")
    bot = _bot()
    await read.read_request(bot, 7, "https://example.com/post")
    *_, failed = bot.send_message.await_args_list
    assert failed.kwargs["text"] == ("⚠️ Could not read it — https://example.com/post: native: HTTP 403; "
                                     "jina: HTTP 429")
    assert [b.text for b in failed.kwargs["reply_markup"].inline_keyboard[0]] == ["🌐 Try with the browser",
                                                                                   "🔎 Research this"]
    assert reader.prompt is None and bot.send_document.await_count == 0


async def test_try_with_the_browser_reads_with_the_browser_rung_only(reader, monkeypatch):
    monkeypatch.setattr(read.fetchers, "BROWSER_AVAILABLE", True)
    reader.found, reader.sources = _found(fetcher="browser")
    await read.read_request(_bot(), 7, "https://example.com/post", browser=True)
    assert reader.requests[0][1] == ["browser"]


async def test_the_browser_not_installed_is_said_with_the_install_commands(reader, monkeypatch):
    monkeypatch.setattr(read.fetchers, "BROWSER_AVAILABLE", False)
    bot = _bot()
    await read.read_request(bot, 7, "https://example.com/post", browser=True)
    assert fetchers.INSTALL["web"] in _texts(bot)[0] and fetchers.INSTALL["chromium"] in _texts(bot)[0]
    assert reader.requests == []


async def test_one_read_per_chat_at_a_time(reader):
    read._reading.add(7)
    bot = _bot()
    await read.read_request(bot, 7, "https://example.com/post")
    assert _texts(bot) == ["📖 Still reading the last link — one at a time."] and reader.requests == []


async def test_no_model_answering_is_said_and_the_text_still_comes(reader, monkeypatch, tmp_path):
    from argus import llm
    monkeypatch.setattr(read.tempfile, "gettempdir", lambda: str(tmp_path))

    def fail(*a, **k):
        raise llm.EngineError("role read.answer: every engine failed", role="read.answer", attempts=[])
    monkeypatch.setattr(read.llm, "complete", fail)
    reader.found, reader.sources = _found()
    bot = _bot()
    await read.read_request(bot, 7, "https://example.com/post")
    assert _texts(bot)[-1].startswith("⚠️ The page was read, but no model answered (role read.answer")
    assert bot.send_document.await_count == 1


async def test_pasted_links_that_are_not_media_get_read_and_research_buttons():
    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    await read.handle_text(update, None, 7, "look https://example.com/post and https://youtu.be/dQw4w9WgXcQ pricing?")
    (call,) = update.message.reply_text.await_args_list
    assert "https://example.com/post" in call.args[0] and "youtu.be" not in call.args[0]
    (row,) = call.kwargs["reply_markup"].inline_keyboard
    assert [b.text for b in row] == ["📖 Read", "🔎 Research"]
    key = row[0].callback_data.split(":")[2]
    assert read._pool_get(7, key) == "https://example.com/post look and pricing?"
    assert read._pool_get(8, key) is None                     # another chat's button never acts here


async def test_media_only_text_gets_no_read_buttons():
    update = SimpleNamespace(message=SimpleNamespace(reply_text=AsyncMock()))
    await read.handle_text(update, None, 7, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert update.message.reply_text.await_count == 0


async def test_read_callbacks(monkeypatch):
    bot = _bot()
    tasks = []
    ctx = SimpleNamespace(application=SimpleNamespace(bot=bot, create_task=lambda coro, update=None: tasks.append(coro)))
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=7), effective_user=SimpleNamespace(id=1))
    await read.handle_callback(update, ctx, "read", "nope")
    assert _texts(bot) == ["This button has expired — send /read <link> again."]
    key = read._pool_put(7, "https://example.com/post")
    started = AsyncMock(return_value=runs.Outcome("busy", "⏳ A research run is already working.", None))
    monkeypatch.setattr(read.runs, "start_run", started)
    await read.handle_callback(update, ctx, "res", key)
    assert started.await_args.kwargs["topic"] == "https://example.com/post"
    assert _texts(bot)[-1] == "⏳ A research run is already working."
    await read.handle_callback(update, ctx, "brw", key)
    assert len(tasks) == 1
    tasks[0].close()                                          # never run: the read itself is tested above


# ---------------------------------------------------------------------------
# /engines
# ---------------------------------------------------------------------------

async def test_engines_lists_the_fetchers_and_both_chains(monkeypatch):
    monkeypatch.setattr(engines_cmd.fetchers, "status", lambda: [
        {"rung": "native", "ok": True, "detail": "httpx"}, {"rung": "browser", "ok": False, "detail": "not installed"}])
    monkeypatch.setattr(fetchers, "SCRAPLING_AVAILABLE", True)
    monkeypatch.setattr(fetchers, "BROWSER_AVAILABLE", False)
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", False)
    lines = await engines_cmd._fetch_lines()
    assert lines == ["Fetchers:", "  ✅ native: httpx", "  — browser: not installed",
                     "  Exa mode: exa_text → native → exa_contents",
                     "  Free web mode and /read: exa_text → native → scrapling → jina"]


# ---------------------------------------------------------------------------
# anchors carry the rung that read them
# ---------------------------------------------------------------------------

def test_read_anchor_records_the_winning_fetcher(monkeypatch, tmp_path):
    md = tmp_path / "p.md"
    md.write_text("A deep page about interrupts. " * 40, encoding="utf-8")
    monkeypatch.setattr(anchors.fetchers, "fetch", lambda *a, **k: fetchers.FetchResult(
        ok=True, markdown_path=str(md), title="Deep page", chars=1200, fetcher="scrapling"))
    anchor, sources, errors = anchors.read_anchor("https://example.com/docs/deep", rc=SimpleNamespace(
        fetch=FetchConfig(chain=FREE_FETCH_CHAIN)), ledger=None)
    assert anchor["ok"] and anchor["fetcher"] == "scrapling" and len(sources) == 1 and errors == []
