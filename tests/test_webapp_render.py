"""Report HTML for the console (v4 spec §2.7, Review Focus 4): markdown →
allowlist sanitiser → ``[n]`` wrapped as cite spans in text nodes only; the
CSP backstop never allows inline script."""
from __future__ import annotations

import re

from aiohttp.test_utils import TestClient, TestServer

from argus.webapp import render, server

from webapp_fakes import FakeRuns, auth, make_app

HOSTILE = """\
# Title <img src=x onerror="alert(1)">

Plain claim [1] and another [12].

<script>alert("pwned")</script>

<div onclick="steal()">raw <b onmouseover="x()">html</b> block</div>

[click](javascript:alert(1)) · [spaced](  JaVaScRiPt:alert(2)) · [entity](&#106;avascript:alert(3))
[tab](java\tscript:alert(4)) · [data](data:text/html;base64,PHNjcmlwdD4=) · [ok](https://example.com/a?b=1&c=2)

<a href="vbscript:msgbox(1)">vb</a> <a href="https://good.example" onclick="bad()" target="_top">good</a>

<iframe src="https://evil.example"></iframe><svg onload="alert(5)"><script>alert(6)</script></svg>

<style>body{display:none}</style><form action="https://evil.example"><input name=x></form>

`inline [3] code` and

```python
x = "[4]"  # <script>alert(7)</script>
```

| a | b |
|---|---|
| [5] | <span style="color:red" onclick="z()">cell</span> |
"""


def test_hostile_markdown_is_sanitised():
    out = render.render_report(HOSTILE)
    low = out.lower()
    for bad in ("<script", "<iframe", "<svg", "<style", "<form", "<input", "<img", "<div",
                "onerror", "onclick", "onmouseover", "onload", "javascript:", "vbscript:", "data:text",
                "style=", "target="):
        assert bad not in low, bad
    assert 'alert("pwned")' not in out and "display:none" not in out   # dropped with their element
    # what survives: structure, safe links (with rel), escaped code, tables
    assert "<h1>" in out and "<table>" in out and "<td>" in out
    assert '<a href="https://example.com/a?b=1&amp;c=2" rel="noopener noreferrer">ok</a>' in out
    assert '<a href="https://good.example" rel="noopener noreferrer">good</a>' in out
    assert "<a>click</a>" in out or ">click<" in out                     # the link text stays, its href does not
    assert "raw <b>html</b> block" in out
    assert "&lt;script&gt;alert(7)&lt;/script&gt;" in out                # code shows the text, escaped


def test_every_tag_and_attribute_in_the_output_is_allowlisted():
    out = render.render_report(HOSTILE)
    for tag, attrs in re.findall(r"<([a-z0-9]+)([^>]*)>", out):
        assert tag in render.ALLOWED_TAGS, tag
        for name in re.findall(r'([a-z-]+)="', attrs):
            assert name in render.ALLOWED_ATTRS.get(tag, ()) or (tag, name) in (
                ("span", "class"), ("span", "data-n"), ("a", "rel")), (tag, name)


def test_citations_are_wrapped_in_text_nodes_only():
    out = render.render_report(HOSTILE)
    assert '<span class="cite" data-n="1">[1]</span>' in out
    assert '<span class="cite" data-n="12">[12]</span>' in out
    assert '<td><span class="cite" data-n="5">[5]</span></td>' in out    # a table cell is text
    assert "<code>inline [3] code</code>" in out                         # never inside code
    assert 'x = "[4]"' in out and 'data-n="4"' not in out                # nor in a fenced block


def test_cite_wrapping_never_reaches_into_attributes():
    out = render.render_report("See [the doc](https://x.example/[2]) and [7].")
    assert 'href="https://x.example/[2]"' in out
    assert 'data-n="2"' not in out and 'data-n="7"' in out


def test_entities_and_unclosed_tags_stay_well_formed():
    out = render.render_report("a < b & c > d <b>bold <i>both")
    assert "a &lt; b &amp; c &gt; d" in out
    assert out.count("<b>") == out.count("</b>") and out.count("<i>") == out.count("</i>")


# --- the CSP backstop ------------------------------------------------------

def _directives(csp: str) -> dict[str, list[str]]:
    return {d.split()[0]: d.split()[1:] for d in (p.strip() for p in csp.split(";")) if d}


def test_csp_has_no_inline_script():
    d = _directives(server.CSP)
    assert d["script-src"] == ["'self'", "https://telegram.org"]
    assert "'unsafe-inline'" not in server.CSP and "'unsafe-eval'" not in server.CSP
    assert d["default-src"] == ["'self'"] and d["connect-src"] == ["'self'"]
    assert d["frame-ancestors"] == ["'self'", "https://web.telegram.org", "https://*.telegram.org"]


async def test_security_headers_on_every_response(tmp_path):
    async with TestClient(TestServer(make_app(FakeRuns(), reports_root=tmp_path))) as c:
        for path, headers in (("/", {}), ("/api/me", {}), ("/api/me", auth()), ("/nope", {})):
            r = await c.get(path, headers=headers)
            assert r.headers["Content-Security-Policy"] == server.CSP, path
            assert r.headers["X-Content-Type-Options"] == "nosniff"
            assert r.headers["Referrer-Policy"] == "no-referrer"
