"""Report HTML for the console (v4 spec §2.7, Review Focus 4): markdown →
allowlist sanitiser → ``[n]`` wrapped as cite spans in text nodes only; the
CSP backstop never allows inline script."""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest
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


@pytest.mark.parametrize("hostile", ["Intro text.\n\n<style>unclosed <p>all gone?\n\nMore [1].",
                                     "Intro text.\n\n<script>alert(1)\n\n## Later\n\nMore [1]."])
def test_an_unclosed_script_or_style_cannot_blank_the_report(hostile):
    out = render.render_report(hostile)
    assert "Intro text." in out
    assert out.endswith("<p>[content removed by the sanitiser]</p>")
    assert "all gone" not in out and "alert" not in out and "More" not in out   # the dropped text stays dropped


def test_no_marker_when_nothing_is_left_open():
    assert "content removed" not in render.render_report("<script>x</script>ok")


# --- the report header follows the report's banners (final review I2) -------

RUN_JS = server.STATIC_DIR / "views" / "run.js"
NOT_CHECKED = "⚠ Not fact-checked — the grounding judge is off"
DEGRADED = "⚠ Degraded run — 3 of 4 sources could not be digested (429)"


def _node(probe: str, *args: str):
    """Run the ES-module ``probe`` in node (``process.argv[1:]`` = ``args``) and
    return the JSON it writes; run.js registers one document listener at load,
    so the probe gets a stub ``document``."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed: the console probes need it")
    out = subprocess.run([node, "--input-type=module", "-e",
                          "globalThis.document = { addEventListener() {} };" + probe, *args],
                         capture_output=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _header(cases: list) -> list[dict]:
    """``trustHeader(banners, evidence)`` of run.js for each case."""
    return _node("const m = await import(process.argv[1]);"
                 "const cases = JSON.parse(process.argv[2]);"
                 "process.stdout.write(JSON.stringify(cases.map(([b, ev]) => ({"
                 "  html: m.trustHeader(b, ev), banner: b.map((t) => m.BANNER.test(t))}))));",
                 RUN_JS.as_uri(), json.dumps(cases))


def test_the_console_header_never_claims_fact_checked_against_the_reports_banners():
    ok = {"degraded": False, "panel": {"judged": 5, "total": 5, "verdict": "pass"}}
    off = {"degraded": False, "panel": {"judged": 0, "total": 0, "verdict": "pass"}}   # judge off: not degraded
    cases = [
        [[NOT_CHECKED], off],
        [[DEGRADED], {**ok, "degraded": True}],
        [[], {**ok, "degraded": True}],                     # degraded with no banner found: still no claim
        [[NOT_CHECKED, DEGRADED], {**ok, "degraded": True, "panel": {"judged": 3, "total": 5, "verdict": "pass"}}],
        [[], ok],
        [[], {"missing": True}],                            # an older run without evidence.json
        [[], {"error": {"message": "x"}}],
        [[], off],
        [['⚠ Not fact-checked — section "<b>" was not fact-checked (unparseable)'], ok],
        [["⚠ Limited evidence — something else"], ok],
    ]
    got = _header(cases)
    for (banners, ev), r in zip(cases[:4], got[:4]):
        assert 'data-kind="trap"' in r["html"] and "Fact-checked:" not in r["html"], (banners, ev)
        assert all(r["banner"]), "the banner prefixes are recognised"
    assert all(b.removeprefix("⚠ ") in got[i]["html"] for i in (0, 1, 3) for b in cases[i][0])
    assert got[4]["html"].startswith('<p class="meta">Fact-checked: all 5 findings judged · panel verdict pass.')
    assert [r["html"] for r in got[5:8]] == ["", "", ""], "no evidence, or no finding judged: no claim either way"
    assert "&lt;b&gt;" in got[8]["html"] and "<b>" not in got[8]["html"] and "Fact-checked:" not in got[8]["html"]
    assert got[9]["banner"] == [False] and "Fact-checked: all 5" in got[9]["html"], "only the two trust banners"


def test_a_refresh_after_an_action_shows_a_network_failure_instead_of_rejecting():   # final review M14
    got = _node("const api = await import(process.argv[1]);"
                "const m = await import(process.argv[2]);"
                "api.get.run = async () => { throw new api.ApiError(0, 'network', 'the tunnel dropped'); };"
                "const slot = { hidden: true, innerHTML: '', scrollIntoView() {},"
                "  querySelector: () => ({ addEventListener() {} }) };"
                "let outcome = 'resolved';"
                "try { await m.refresh({ id: 'r1', notice: slot, stopped: false }); }"
                "catch (e) { outcome = 'rejected: ' + e.message; }"
                "process.stdout.write(JSON.stringify({ outcome, hidden: slot.hidden, html: slot.innerHTML }));",
                (server.STATIC_DIR / "api.js").as_uri(), RUN_JS.as_uri())
    assert got["outcome"] == "resolved" and got["hidden"] is False and "the tunnel dropped" in got["html"], got


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
