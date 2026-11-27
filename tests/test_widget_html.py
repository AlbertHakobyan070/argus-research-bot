"""The media card widget stays inside Telegram's widget profile 1 and limits
(core.telegram.org/bots/widgets), and its script draws every phase and every
fallback.

The bot owns the card's state and sends it as ``chat_data``: compact JSON of at
most 1024 UTF-8 bytes, any key may be missing::

    v   1                                  ph  read|pick|queue|dl|done|fail|x
    t   title                              m   meta line ("Fireship · 12:34 · youtube")
    f   v|a (video or mp3)                 vq  video heights ["2160", …, "360"]
    aq  mp3 options ["320", "v0", …]       q   chosen height   a  chosen mp3 option
    sz  size estimate ("≈ 84 MB")          p   percent 0-100 (dl)
    st  video|audio|download|merging|converting|finishing|queued|starting
    g   "35.2 / 84.0 MB"   e  ETA "0:31"   r   done summary    err  error text

The widget sends only "f:v", "f:a", "q:<height>", "a:<option>", "ok", "x",
"retry", and redraws from a JSON ``answer.result`` at once (the bot also pushes
it as chatData)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import zlib
from pathlib import Path

import pytest

WIDGET = Path(__file__).resolve().parents[1] / "src" / "argus" / "assets" / "widgets" / "media_card.html"
SCHEMA = {"v", "n", "ph", "t", "m", "f", "vq", "aq", "q", "a", "sz", "p", "st", "g", "e", "r", "err"}
REQUESTS = {"f:v", "f:a", "ok", "x", "retry"}
OPTION_REQUEST = re.compile(r"^(q|a):\w+$")
TITLE = ("Разбор архитектуры трансформера и механизма внимания " * 3)[:90]
META = "Канал «Технологии» · 1:02:34 · youtube"


def _raw() -> bytes:
    return WIDGET.read_bytes()


def _css() -> str:
    return re.search(r"<style>(.*?)</style>", _raw().decode("utf-8"), re.S).group(1)


def _script() -> str:
    return re.search(r"<script>(.*?)</script>", _raw().decode("utf-8"), re.S).group(1)


def _samples() -> dict[str, dict]:
    """A realistic chat_data per phase, with a 90-character Cyrillic title."""
    base = {"v": 1, "t": TITLE, "m": META}
    vq, aq = ["2160", "1440", "1080", "720", "480", "360"], ["320", "v0", "192", "128"]
    return {
        "read": {"v": 1, "ph": "read"},
        "pick": {**base, "ph": "pick", "f": "v", "vq": vq, "aq": aq, "q": "1080", "a": "320", "sz": "≈ 84 MB"},
        "pick_a": {**base, "ph": "pick", "f": "a", "vq": vq, "aq": aq, "q": "1080", "a": "320", "sz": "≈ 9.6 MB"},
        "queue": {**base, "ph": "queue", "f": "v", "q": "1080", "st": "queued"},
        "dl": {**base, "ph": "dl", "f": "v", "q": "1080", "st": "video", "p": 42, "g": "35.2 / 84.0 MB", "e": "0:31"},
        "done": {**base, "ph": "done", "f": "v", "r": "1080p MP4 · 84.1 MB"},
        "fail": {**base, "ph": "fail", "f": "v",
                 "err": "ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm you're not a bot. "
                        "Use --cookies-from-browser or --cookies for the authentication."},
        "x": {**base, "ph": "x", "f": "v"},
        "off": {"v": 1, "n": 9, "ph": "off", "t": TITLE, "err": "Moved to the regular card below."},
    }


def _dump(state: dict) -> str:
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def test_utf8_without_nul():
    raw = _raw()
    raw.decode("utf-8")
    assert b"\x00" not in raw


def test_within_size_limits():
    raw = _raw()
    packed = len(zlib.compress(raw, 9))
    print(f"media_card.html: {len(raw)} bytes, {packed} bytes after zlib level 9")
    assert len(raw) <= 32768
    assert packed <= 4096


def test_no_external_resources():
    text = _raw().decode("utf-8")
    assert not re.search(r"https?://", text)
    for token in ("<script src", "<link", "@import", "@font-face", "url("):
        assert token not in text, token


def test_no_removed_elements():
    text = _raw().decode("utf-8")
    hit = re.search(r"<\s*(input|select|textarea|progress|meter|video|audio|iframe|object|embed)\b", text, re.I)
    assert hit is None, hit and hit.group(0)


def test_no_unsupported_css():
    css = _css()
    for token in ("oklch(", "color-mix(", ":has(", "sticky", "line-clamp", "@container", "@layer"):
        assert token not in css, token
    assert not re.search(r"\d(vh|dvh|svh|lvh)\b", css)
    assert not re.search(r"overflow\s*:\s*(auto|scroll)", css)


def test_colours_come_from_the_theme():
    text = _raw().decode("utf-8")
    assert not re.search(r"(?<!&)#[0-9a-fA-F]{3,8}\b", _css()), "hard-coded hex colour"
    assert not re.search(r"\b(rgba?|hsla?)\(", text)
    for block in re.findall(r"\{([^{}]*)\}", _css()):
        for decl in filter(None, (d.strip() for d in block.split(";"))):
            prop, value = (s.strip().lower() for s in decl.split(":", 1))
            if prop in {"color", "background", "background-color", "fill", "stroke", "border-color"}:
                assert "var(--tg-theme-" in value or value in {"transparent", "currentcolor"}, decl
            elif prop in {"border", "box-shadow", "outline"}:
                assert "var(--tg-theme-" in value or "transparent" in value or value in {"0", "none"}, decl


def test_requests_are_documented_and_sent_on_a_tap():
    js = _script()
    for literal in (*REQUESTS, "q:", "a:"):
        assert f"'{literal}'" in js, literal
    assert js.count(".request(") == 1
    assert "W.request(" in js[js.index("function ask("):js.index("function ask(") + 80]
    calls = [ln for ln in js.splitlines() if re.search(r"(?<!function )\bask\(", ln)]
    assert len(calls) == 1 and "addEventListener('click'" in calls[0], calls


def test_script_reads_only_documented_keys():
    assert set(re.findall(r"\bs\.(\w+)", _script())) <= SCHEMA


def test_sample_chat_data_fits_1024_bytes():
    for phase, state in _samples().items():
        assert set(state) <= SCHEMA, phase
        size = len(_dump(state).encode("utf-8"))
        print(f"chat_data {phase}: {size} bytes")
        assert size <= 1024, phase


HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const S = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const tick = () => new Promise(r => setTimeout(r, 0));
function boot(chatData, reply) {
  const els = {};
  const el = () => ({_h: '', writes: 0, textContent: '', hidden: false, className: '', style: {}, on: {},
    get innerHTML() { return this._h; }, set innerHTML(v) { this._h = v; this.writes++; },
    addEventListener(t, f) { this.on[t] = f; }});
  const ctx = {document: {getElementById: id => els[id] || (els[id] = el())}};
  ctx.window = ctx;
  let tw = null;
  if (chatData !== undefined) {
    tw = {chatData, H: {}, reqs: [], readies: 0, onEvent(k, f) { this.H[k] = f; }, ready() { this.readies++; },
      request(d) { this.reqs.push(d); return reply(d); }};
    ctx.TelegramWidget = tw;
  }
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  const click = r => els.w.on.click({target: {dataset: {r}, disabled: false}});
  const view = () => ({h: els.h._h, a: els.a._h, g: els.g._h, c: els.c._h, bar: !els.b.hidden,
    width: els.i.style.width, note: els.n.hidden ? null : els.n.textContent, err: els.n.className === 'e',
    taps: [...(els.a._h + els.c._h).matchAll(/data-r="([^"]*)"/g)].map(m => m[1])});
  return {els, tw, click, view};
}
const fail = code => () => Promise.reject(Object.assign(new Error(code), {name: 'WidgetRequestError', code}));
(async () => {
  const out = {phases: {}, errors: {}};
  out.no_tw = boot(undefined).els.w.textContent;
  for (const [k, v] of Object.entries(S)) { const b = boot(v); out.phases[k] = b.view(); out.ready = b.tw.readies; }
  { let done; const b = boot(S.pick, () => new Promise(r => { done = r; }));
    b.click('ok'); out.inflight = b.view(); b.click('x'); out.sent = b.tw.reqs.slice();
    done({result: S.dl}); await tick(); out.answered = b.view();
    const w = b.els.c.writes; b.tw.chatData = S.dl.replace('"p":42', '"p":43'); b.tw.H.chatDataChanged();
    out.pushed = b.view(); out.cancel_rewrites = b.els.c.writes - w; }
  for (const code of ['NOT_ALLOWED', 'TIMEOUT', 'RATE_LIMITED', 'FAILED', 'BUSY']) {
    const b = boot(S.pick, fail(code)); b.click('f:a'); await tick(); out.errors[code] = b.view(); }
  { let done; const b = boot(JSON.stringify({ph: 'dl', n: 3, st: 'video', p: 40}), () => new Promise(r => { done = r; }));
    b.click('x'); b.tw.chatData = JSON.stringify({ph: 'x', n: 5}); b.tw.H.chatDataChanged();
    done({result: JSON.stringify({ph: 'dl', n: 4, st: 'video', p: 41})}); await tick(); out.stale = b.view(); }
  { const b = boot(S.pick, fail('TIMEOUT')); b.click('ok'); await tick(); b.tw.chatData = S.dl; b.tw.H.chatDataChanged();
    out.cleared = b.view(); }
  { const b = boot(S.pick, () => Promise.resolve({result: 'Link expired'})); b.click('ok'); await tick(); out.text = b.view(); }
  { const b = boot(S.pick, () => Promise.resolve({result: null})); b.click('ok'); await tick(); out.nothing = b.view(); }
  { const b = boot(S.pick, () => { throw new TypeError('data too long'); }); b.click('ok'); await tick(); out.thrown = b.view(); }
  out.bad = boot('not json').view();
  out.empty = boot(null).view();
  out.esc = boot(JSON.stringify({ph: 'done', t: '<img src=x>', r: 'a&b'})).view();
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_script_renders_every_phase_and_fallback(tmp_path):
    (tmp_path / "w.js").write_text(_script(), encoding="utf-8")
    (tmp_path / "s.json").write_text(json.dumps({k: _dump(v) for k, v in _samples().items()}), encoding="utf-8")
    (tmp_path / "h.js").write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run([shutil.which("node"), str(tmp_path / "h.js"), str(tmp_path / "w.js"), str(tmp_path / "s.json")],
                          capture_output=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    ph = out["phases"]

    assert "Telegram" in out["no_tw"]
    assert out["ready"] == 1
    for name, view in ph.items():
        assert all(t in REQUESTS or OPTION_REQUEST.match(t) for t in view["taps"]), (name, view["taps"])
        assert view["note"] is None, name
    assert 'class="sk' in ph["read"]["a"] and ph["read"]["taps"] == []
    assert "…" in ph["pick"]["h"] and META in ph["pick"]["h"], "long title cut, meta shown"
    assert ph["pick"]["taps"] == ["f:a", "q:2160", "q:1440", "q:720", "q:480", "q:360", "ok", "x"]
    assert "1080p" in ph["pick"]["a"] and "≈ 84 MB" in ph["pick"]["a"] and not ph["pick"]["bar"]
    assert ph["pick_a"]["taps"] == ["f:v", "a:v0", "a:192", "a:128", "ok", "x"]
    assert "V0" in ph["pick_a"]["a"] and "kbps" in ph["pick_a"]["a"]
    assert ph["queue"]["bar"] and ph["queue"]["width"] == "0%" and ph["queue"]["taps"] == ["x"]
    assert ph["dl"]["bar"] and ph["dl"]["width"] == "42%" and "42%" in ph["dl"]["a"]
    assert "35.2 / 84.0 MB" in ph["dl"]["g"] and "ETA 0:31" in ph["dl"]["g"] and ph["dl"]["taps"] == ["x"]
    assert "1080p MP4 · 84.1 MB" in ph["done"]["a"] and ph["done"]["taps"] == [] and not ph["done"]["bar"]
    assert "Sign in to confirm" in ph["fail"]["a"] and ph["fail"]["taps"] == ["retry"]
    assert "cancelled" in ph["x"]["a"] and ph["x"]["taps"] == []
    assert "Moved to the regular card below." in ph["off"]["a"] and ph["off"]["taps"] == [] and not ph["off"]["bar"]

    assert out["sent"] == ["ok"], "a second tap while a request is in flight is not sent"
    assert out["inflight"]["note"] == "Sending…" and "disabled" in out["inflight"]["c"]
    assert out["answered"]["bar"] and out["answered"]["width"] == "42%" and out["answered"]["note"] is None
    assert "43%" in out["pushed"]["a"] and out["cancel_rewrites"] == 0, "a progress push keeps the Cancel button"

    assert out["errors"]["NOT_ALLOWED"]["note"] == "Use the buttons under this message"
    assert out["errors"]["TIMEOUT"]["note"] == "Argus did not answer — try again"
    assert out["errors"]["RATE_LIMITED"]["note"] == "Too fast — wait a moment"
    assert out["errors"]["FAILED"]["note"] == "FAILED"
    assert out["errors"]["BUSY"]["note"] == "Still sending the last tap — try again in a moment"
    for code, view in out["errors"].items():
        assert "disabled" not in view["c"], code
        assert view["err"], code
    assert "cancelled" in out["stale"]["a"] and out["stale"]["taps"] == [], "an older answer loses to a newer push"
    assert out["cleared"]["note"] is None and "42%" in out["cleared"]["a"], "a push clears an old error note"
    assert out["text"]["note"] == "Link expired"
    assert out["nothing"]["note"] is None
    assert out["thrown"]["note"] == "data too long" and "disabled" not in out["thrown"]["c"]
    assert out["bad"]["note"] == "Unreadable card state" and out["bad"]["err"]
    assert 'class="sk' in out["empty"]["a"]
    assert "<img" not in out["esc"]["h"] and "&#60;img" in out["esc"]["h"] and "a&#38;b" in out["esc"]["a"]
