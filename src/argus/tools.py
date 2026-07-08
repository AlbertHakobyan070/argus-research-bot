"""LangGraph tool wrappers around the intel-stack scripts.

These wrap, do not reimplement:
  - harvest.py  (primary-source harvester / intel-radar)
  - snatch.py   (universal downloader / web-hunter)
  - crawl.py    (crawl4ai wrapper)
  - article_convert.py (URL/local file -> clean markdown)

Each tool:
  - runs the script as a subprocess with a timeout
  - parses stdout / the resulting files
  - returns a typed Pydantic model the graph nodes can consume

We never import the intel-stack modules directly because they import
playwright/crawl4ai at module top level; keeping them as subprocesses
gives us isolation + lets us time-bound them.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("argus.tools")

INTEL_STACK_DIR = Path(r"A:\Hermes\Agents\intel-stack\scripts")
PYTHON_BIN = Path(sys.executable)  # the argus venv's python

# Some intel-stack scripts need their own venv (feedparser, crawl4ai,
# scrapling, markitdown, yt_dlp). Set INTEL_PYTHON to override; default
# to intel-stack's venv python on this host.
INTEL_PYTHON_BIN = Path(os.environ.get(
    "INTEL_PYTHON",
    r"A:\Hermes\Agents\intel-stack\venv\Scripts\python.exe",
))

DEFAULT_TIMEOUT_S = int(os.environ.get("ARGUS_TOOL_TIMEOUT", "120"))


class HarvestResult(BaseModel):
    """A single primary-source item surfaced by harvest.py."""
    section: str
    title: str
    url: str
    summary: str = ""
    published: str = ""
    source: str = ""


class HarvestReport(BaseModel):
    folder: str
    radar_md: str = ""
    items: list[HarvestResult] = Field(default_factory=list)
    raw_stdout: str = ""
    duration_s: float = 0.0


class SnatchResult(BaseModel):
    ok: bool
    folder: str | None = None
    markdown_path: str | None = None
    title: str = ""
    url: str = ""
    error: str | None = None
    duration_s: float = 0.0


class CrawlResult(BaseModel):
    ok: bool
    folder: str | None = None
    markdown_path: str | None = None
    pages: list[str] = Field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0


class NormalizeResult(BaseModel):
    ok: bool
    markdown_path: str | None = None
    markdown_text: str = ""
    title: str = ""
    error: str | None = None
    duration_s: float = 0.0


def _run_script(script: str, args: list[str], *, timeout: int = DEFAULT_TIMEOUT_S,
                env_extra: dict[str, str] | None = None,
                python_bin: Path | None = None) -> tuple[int, str, str]:
    """Run an intel-stack script and capture stdout/stderr.

    Default interpreter is INTEL_PYTHON_BIN because every intel-stack
    script depends on heavy modules (feedparser, crawl4ai, scrapling,
    markitdown, yt_dlp) that live only in the intel-stack venv. Routing
    them through PYTHON_BIN (the argus venv) returns ModuleNotFoundError
    instantly — see T2 of Argus Pattern E decomposition.

    We strip PYTHONPATH so the intel-stack subprocess doesn't accidentally
    inherit the argus venv paths or the global Hermes PYTHONPATH leak.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if env_extra:
        env.update(env_extra)
    py = python_bin or INTEL_PYTHON_BIN
    cmd = [str(py), str(INTEL_STACK_DIR / script), *args]
    logger.debug("exec: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=str(INTEL_STACK_DIR),
            capture_output=True, text=True,
            timeout=timeout, env=env, encoding="utf-8", errors="replace",
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or "") if isinstance(e.stdout, str) else "", (
            f"timeout after {timeout}s"
        )

def _parse_json_field(out: str, field: str) -> str:
    """Extract ``field`` from the last JSON-looking line of stdout.

    Several intel-stack scripts (snatch.py, crawl.py) print a single
    JSON object on stdout like ``{"ok": true, "folder": "A:\..."}``.
    Returns the value of ``field`` if found, else "".

    Defensive: scans *all* lines (not just the last) for one that
    starts with ``{`` and ends with ``}`` and parses; the canonical
    contract is the LAST line is the JSON line, but tools occasionally
    print status banners before it.
    """
    if not out:
        return ""
    candidates = []
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            candidates.append(ln)
    for ln in reversed(candidates):
        try:
            import json as _json
            obj = _json.loads(ln)
            v = obj.get(field)
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            continue
    return ""


def _parse_article_convert_path(out: str, *, prefer: str = "md") -> str | None:
    """Extract the artifact path from ``article_convert.py`` stdout.

    The script prints lines like ``md: A:\path\file.md`` or
    ``pdf: A:\path\file.pdf`` as its last data line. ``prefer``
    chooses which kind to return (default ``md``); if absent, returns
    the first md/pdf/html line found.
    """
    if not out:
        return None
    md_path = None
    pdf_path = None
    for raw in out.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        # Strip a leading "<kind>: " prefix if present.
        for kind in ("md", "pdf", "html"):
            prefix = f"{kind}: "
            if ln.lower().startswith(prefix):
                candidate = ln[len(prefix):].strip()
                # Defensive: confirm it looks like an absolute path
                # on this host (A:\... or /...).
                if (("\\" in candidate or "/" in candidate)
                        and Path(candidate).exists()):
                    if kind == "md" and md_path is None:
                        md_path = candidate
                    elif kind == "pdf" and pdf_path is None:
                        pdf_path = candidate
                break
    return md_path or pdf_path


def harvest_sources(*, hours: int = 72, top: int = 8,
                    sections: str = "papers,repos,news,blogs",
                    date: str | None = None,
                    timeout: int = DEFAULT_TIMEOUT_S) -> HarvestReport:
    """Run intel-stack harvest.py and parse its JSON output.

    harvest.py prints a single JSON line on stdout like
    ``{"ok": true, "dir": "A:\\...\\radar\\2026-07-07", "new_items": 213, ...}``.
    We read radar.md from that dir to extract primary-source items.

    Uses the intel-stack venv's python (which has feedparser, crawl4ai,
    scrapling, markitdown, yt_dlp installed).
    """
    args = ["--hours", str(hours), "--top", str(top),
            "--sections", sections]
    if date:
        args += ["--date", date]
    t0 = time.time()
    rc, out, err = _run_script("harvest.py", args, timeout=timeout)
    duration = time.time() - t0
    items: list[HarvestResult] = []
    radar_md = ""
    folder = ""
    if rc == 0:
        # harvest prints a single JSON line as its final stdout line.
        for line in reversed((out or "").strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    folder = payload.get("dir") or folder
                    break
                except Exception:
                    continue
        radar_path = Path(folder) / "radar.md" if folder else None
        if radar_path and radar_path.exists():
            radar_md = radar_path.read_text(encoding="utf-8", errors="replace")
            items = _parse_radar_md(radar_md)
    return HarvestReport(
        folder=folder,
        radar_md=radar_md[:8000],  # truncate for state
        items=items,
        raw_stdout=out[-2000:],
        duration_s=duration,
    )


# A very small markdown linker — extracts items from radar.md.
# Supports both `- [title](url)` bullets and `### [title](url)` headings.
def _parse_radar_md(text: str) -> list[HarvestResult]:
    out: list[HarvestResult] = []
    section = "general"
    current_item_title: str | None = None
    current_item_url: str | None = None
    current_item_score: str = ""
    current_item_summary: list[str] = []

    def _split_md_link(inner: str) -> tuple[str, str] | None:
        """Return (title, url) from a markdown [title](url) fragment.

        The URL may itself contain ')'. Walk to the LAST ') ' that is
        followed by content (so we don't truncate `https://.../x)`).
        """
        if "](" not in inner:
            return None
        title = inner.split("](", 1)[0].lstrip("[").strip()
        rest = inner.split("](", 1)[1]
        # Look for a closing `)` that is followed by space or end-of-string.
        # Find the last `)` that's followed by whitespace or end.
        idx = -1
        for i, ch in enumerate(rest):
            if ch == ")" and (i == len(rest) - 1 or rest[i + 1] in " \t\n"):
                idx = i
        if idx == -1:
            # fallback: last `)`
            idx = rest.rfind(")")
        if idx == -1:
            return None
        url = rest[:idx].strip()
        return title, url

    def _flush():
        nonlocal current_item_title, current_item_url, current_item_score
        nonlocal current_item_summary
        if current_item_title and current_item_url:
            summary_text = current_item_summary[-1] if current_item_summary else ""
            out.append(HarvestResult(
                section=section,
                title=current_item_title,
                url=current_item_url,
                summary=summary_text,
                published=current_item_score,
            ))
        current_item_title = None
        current_item_url = None
        current_item_score = ""
        current_item_summary = []

    def _find_url_end(s: str, start: int) -> int:
        """Return the index just past the URL in `s[start:]`.

        A URL char is alnum + /-_.?&=:%~+. We stop at the first ')' that
        is followed by a non-URL char (so we don't truncate
        `https://x(y)foo`).
        """
        url_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_.~:/?#[]@!$&'()*+,;=%"
        )
        i = start
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == ")":
                # If next char is non-URL (space, tab, newline, end) OR
                # next char is a markdown delimiter, this is our URL end.
                nxt = s[i + 1] if i + 1 < n else ""
                if not nxt or nxt in " \t\n*_`[":
                    return i + 1
                # Otherwise, the ')' is part of URL. Continue.
            i += 1
        return n

    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("## "):
            _flush()
            section = stripped[3:].strip().lower()
            continue
        if stripped.startswith("### "):
            _flush()
            inner = stripped[4:].strip()
            if "](" in inner:
                title_end = inner.index("](")
                title = inner[:title_end].lstrip("[").strip()
                url_start = title_end + 2
                url_end_idx = _find_url_end(inner, url_start)
                # url_end_idx points at the position AFTER ')'. Drop it.
                url = inner[url_start:url_end_idx - 1].strip()
                rest = inner[url_end_idx:].strip()
                current_item_title = title
                current_item_url = url
                current_item_score = rest
            else:
                current_item_title = inner
                current_item_url = ""
            continue
        if stripped.startswith("- "):
            try:
                link, _, rest = stripped[2:].partition(") ")
                parsed = _split_md_link(link) if "](" in link else None
                if parsed:
                    title, url = parsed
                    summary = rest.strip()
                    if title and url:
                        out.append(HarvestResult(
                            section=section, title=title, url=url,
                            summary=summary,
                        ))
            except Exception:
                continue
            continue
        if stripped.startswith(">") and (current_item_title
                                          or current_item_url):
            current_item_summary.append(stripped.lstrip("> ").strip())
    _flush()
    return out


def snatch_url(url: str, *, kind: str = "auto",
               dest: str | None = None,
               timeout: int = DEFAULT_TIMEOUT_S) -> SnatchResult:
    """Run snatch.py for a single URL. Returns the local markdown path.

    Contract (T5 fix): ``snatch.py`` prints a single JSON line on stdout
    like ``{"ok": true, "kind": "articles", "folder": "A:\\\\...\\\\dir"}``.
    Parse the JSON; use ``payload["folder"]`` as the directory. Only
    return ok=True when that directory actually exists and contains
    >=1 .md. Old behavior set ok=True whenever rc==0 even if folder
    parsing failed (e.g. grabbed the literal JSON line as folder,
    which never resolved on disk) — that was the silent-empty-report
    bug observed in t_7f2b625c.
    """
    args = [url, "--kind", kind]
    if dest:
        args += ["--dest", dest]
    t0 = time.time()
    rc, out, err = _run_script("snatch.py", args, timeout=timeout)
    duration = time.time() - t0
    if rc != 0:
        return SnatchResult(ok=False, url=url, error=(err or out)[-500:],
                            duration_s=duration)
    folder = _parse_json_field(out, "folder")
    md = None
    title = ""
    if folder and Path(folder).is_dir():
        mds = sorted(Path(folder).rglob("*.md"))
        if mds:
            md = str(mds[0])
            try:
                text = mds[0].read_text(encoding="utf-8", errors="replace")
                for ln in text.splitlines():
                    if ln.startswith("# "):
                        title = ln[2:].strip()
                        break
            except Exception:
                pass
    ok = bool(md) and Path(md).exists()
    return SnatchResult(ok=ok, folder=folder or None, markdown_path=md,
                        title=title, url=url, duration_s=duration)


def crawl_url(url: str, *, deep: bool = False, max_pages: int = 8,
              depth: int = 1,
              timeout: int = DEFAULT_TIMEOUT_S) -> CrawlResult:
    """Crawl a documentation site / SPA via crawl.py.

    Contract (T5 fix): ``crawl.py`` prints a single JSON line on stdout
    with at least ``{"ok": true, "folder": "A:\\\\...\\\\dir", ...}``.
    Parse the JSON; use ``payload["folder"]`` as the directory and
    collect up to ``max_pages`` .md files under it. Only return ok=True
    when the directory exists AND produced >=1 .md. Old behavior set
    ok=True whenever rc==0 even if folder parsing failed.
    """
    args = [url, "--max-pages", str(max_pages), "--depth", str(depth)]
    if deep:
        args.append("--deep")
    t0 = time.time()
    rc, out, err = _run_script("crawl.py", args, timeout=timeout)
    duration = time.time() - t0
    if rc != 0:
        return CrawlResult(ok=False, error=(err or out)[-500:],
                           duration_s=duration)
    folder = _parse_json_field(out, "folder")
    md = None
    pages: list[str] = []
    if folder and Path(folder).is_dir():
        mds = sorted(Path(folder).rglob("*.md"))
        if mds:
            md = str(mds[0])
            pages = [str(p) for p in mds[:max_pages]]
    ok = bool(md) and Path(md).exists()
    return CrawlResult(ok=ok, folder=folder or None, markdown_path=md,
                        pages=pages, duration_s=duration)


def normalize_to_markdown(source: str, *, md_only: bool = True,
                           timeout: int = DEFAULT_TIMEOUT_S) -> NormalizeResult:
    """Convert URL or local file -> clean markdown via article_convert.py.

    Contract (T5 fix): ``article_convert.py`` prints the destination
    **as a single-prefixed line**, e.g. ``md: A:\\\\path\\\\file.md``
    (with ``--md-only``, ``pdf:`` and ``html:`` are also possible). The
    old parser treated this line as a *folder* and ran ``rglob('*.md')``
    on it, which never matched because the line is a file path with a
    ``md: `` prefix — that was the silent-empty-report bug. New parser
    strips the ``<kind>: `` prefix and uses the path directly as the
    markdown file path (and verifies it exists).
    """
    args = [source, "--md-only"]
    t0 = time.time()
    rc, out, err = _run_script("article_convert.py", args, timeout=timeout)
    duration = time.time() - t0
    if rc != 0:
        return NormalizeResult(ok=False, error=(err or out)[-500:],
                               duration_s=duration)
    md_path = _parse_article_convert_path(out, prefer="md")
    md_text = ""
    title = ""
    if md_path and Path(md_path).is_file():
        try:
            md_text = Path(md_path).read_text(encoding="utf-8", errors="replace")
            for ln in md_text.splitlines():
                if ln.startswith("# "):
                    title = ln[2:].strip()
                    break
        except Exception:
            pass
    ok = bool(md_path) and Path(md_path).exists() and bool(md_text)
    return NormalizeResult(ok=ok, markdown_path=md_path if ok else None,
                           markdown_text=md_text[:20000],
                           title=title, duration_s=duration)


def markdown_to_pdf(md_text: str, pdf_path: str, *, title: str = "") -> None:
    """Render markdown -> PDF.

    Primary path: ReportLab Platypus (fast, no browser needed, reliable
    in the argus venv). Falls back to the intel-stack Chromium path
    (slow + memory hungry + breaks on some Windows pagefile configs)
    if ReportLab is not available.
    """
    try:
        _render_pdf_reportlab(md_text, pdf_path, title=title)
        return
    except Exception as e:
        logger.warning("ReportLab render failed (%s); trying Chromium.", e)
    # Fallback: chromium via the intel-stack helper.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_intel_common", INTEL_STACK_DIR / "_common.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    mod.markdown_to_pdf(md_text, Path(pdf_path), title=title)


def _render_pdf_reportlab(md_text: str, pdf_path: str, *, title: str) -> None:
    """T7 — designed PDF via ReportLab (Chromium-free fallback path).

    We render headings, paragraphs, bullet lists, code blocks, blockquotes,
    and tables with a real stylesheet (sans headings + serif body + monospace
    code + tinted dividers + coloured confidence markers). ReportLab is the
    fallback when the intel-stack Chromium renderer is unavailable; the
    primary Chromium route in ``_common.markdown_to_pdf`` produces the
    same visual identity via CSS.

    Why still maintain a ReportLab path
    -----------------------------------
    Some Windows machines (Albert's is one) have flaky pagefile behaviour
    when Chromium spins up. The ReportLab path is ~50ms and produces a
    deterministic, font-safe PDF even on a constrained host. The Chromium
    path is preferred because CSS gives us callouts/tables/dividers, but
    if it fails, the ReportLab fallback must not look like 1995.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Preformatted, Table, TableStyle,
                                     KeepTogether, HRFlowable)
    from reportlab.lib import colors

    # Colour palette mirrors the Chromium CSS so the two paths look the same.
    INK = colors.HexColor("#0f172a")        # heading ink
    BODY = colors.HexColor("#1a1a1a")       # body ink
    MUTED = colors.HexColor("#475569")      # meta / captions
    RULE = colors.HexColor("#1e40af")       # primary accent
    TINT = colors.HexColor("#f1f5f9")       # h2 background tint
    CODE_BG = colors.HexColor("#f8fafc")    # inline-code background
    CODE_INK = colors.HexColor("#0f172a")
    DIVIDER = colors.HexColor("#94a3b8")    # dashed hr
    QUALITY_BG = colors.HexColor("#eff6ff")
    QUALITY_BORDER = colors.HexColor("#2563eb")
    CONF_HIGH = colors.HexColor("#047857")
    CONF_MED = colors.HexColor("#b45309")
    CONF_LOW = colors.HexColor("#b91c1c")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "ArgusH1", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=INK, spaceAfter=8, spaceBefore=0,
    )
    h2 = ParagraphStyle(
        "ArgusH2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=INK, spaceBefore=14, spaceAfter=6,
        backColor=TINT, borderPadding=(4, 6, 4, 6),
        leftIndent=0,
    )
    h3 = ParagraphStyle(
        "ArgusH3", parent=styles["Heading3"],
        fontName="Helvetica-Bold", fontSize=12, leading=15,
        textColor=INK, spaceBefore=10, spaceAfter=4,
    )
    h4 = ParagraphStyle(
        "ArgusH4", parent=styles["Heading4"],
        fontName="Helvetica-Oblique", fontSize=10.5, leading=13,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8, spaceAfter=2,
    )
    body = ParagraphStyle(
        "ArgusBody", parent=styles["BodyText"],
        fontName="Times-Roman", fontSize=10.5, leading=14.5,
        textColor=BODY, spaceAfter=4, alignment=TA_LEFT,
    )
    bullet = ParagraphStyle(
        "ArgusBullet", parent=body, leftIndent=14, bulletIndent=4,
        spaceAfter=2,
    )
    quote = ParagraphStyle(
        "ArgusQuote", parent=body, leftIndent=18, rightIndent=8,
        textColor=MUTED, fontName="Times-Italic",
        borderPadding=(4, 6, 4, 8),
    )
    code = ParagraphStyle(
        "ArgusCode", parent=body, fontName="Courier",
        fontSize=9, leading=11, leftIndent=8,
        textColor=CODE_INK, backColor=CODE_BG,
    )
    codeblock = ParagraphStyle(
        "ArgusCodeBlock", parent=code,
        backColor=colors.HexColor("#0f172a"),
        textColor=colors.HexColor("#e2e8f0"),
        fontSize=8.5, leading=11,
        borderPadding=(6, 8, 6, 8),
    )
    meta = ParagraphStyle(
        "ArgusMeta", parent=body, fontName="Helvetica",
        fontSize=8.5, textColor=MUTED, leading=11,
    )
    quality = ParagraphStyle(
        "ArgusQuality", parent=body,
        backColor=QUALITY_BG, borderColor=QUALITY_BORDER,
        borderWidth=0, leftBorderColor=QUALITY_BORDER,
        leftBorderWidth=3,
        borderPadding=(8, 10, 8, 10),
        fontSize=10, leading=13,
        fontName="Helvetica",
    )
    conf_high = ParagraphStyle("ConfHigh", parent=body, fontName="Helvetica-Bold",
                                textColor=CONF_HIGH, fontSize=10)
    conf_med = ParagraphStyle("ConfMed", parent=body, fontName="Helvetica-Bold",
                               textColor=CONF_MED, fontSize=10)
    conf_low = ParagraphStyle("ConfLow", parent=body, fontName="Helvetica-Bold",
                               textColor=CONF_LOW, fontSize=10)
    divider = HRFlowable(
        width="100%", thickness=3,
        color=RULE, spaceBefore=14, spaceAfter=10,
        hAlign="CENTER",
    )

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))

    def conf_style(level: str) -> ParagraphStyle:
        return {"high": conf_high, "medium": conf_med,
                "low": conf_low}.get(level.lower(), conf_med)

    def md_inline_to_rl(text: str) -> str:
        """Light inline-MD -> ReportLab miniHTML: **bold**, *em*, `code`, [n].

        We deliberately do NOT try to be a full markdown renderer here;
        we just upgrade the bold/em/inline-code/inline-link spans so the
        body has the same visual rhythm as the Chromium route.
        """
        import re as _re
        # Escape first so user content can't inject miniHTML.
        s = esc(text)
        # Inline code: `...`
        s = _re.sub(r"`([^`]+)`",
                    r'<font name="Courier" color="#0f172a" '
                    r'backColor="#f8fafc">\1</font>', s)
        # Bold: **...**
        s = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        # Italic: *...*  (single-asterisk, non-greedy)
        s = _re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", s)
        # [n] -> superscript number badge
        s = _re.sub(r"\[(\d+)\]", r'<font color="#1d4ed8"><b>[\1]</b></font>', s)
        return s

    flow = []
    if title:
        flow.append(Paragraph(esc(title), h1))
        flow.append(divider)

    in_code = False
    code_buf: list[str] = []
    # Table support: collects lines between | markers, then flushes.
    table_buf: list[str] = []
    in_table = False
    last_was_quality = False

    def flush_table() -> list:
        if not table_buf:
            return []
        rows = []
        for row in table_buf:
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            rows.append([Paragraph(md_inline_to_rl(c), body) for c in cells])
        if len(rows) >= 2 and all(set(c.replace("|","").replace(":","").replace("-","").strip()) == set()
                                   for c in table_buf[1]):
            # second row is the markdown alignment marker (---|---|) — drop it
            rows.pop(1)
        tbl = Table(rows, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), RULE),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#f8fafc")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        return [tbl, Spacer(1, 8)]

    for raw in (md_text or "").splitlines():
        line = raw.rstrip()
        # Flush table if we hit a non-table line.
        if in_table and not line.lstrip().startswith("|"):
            flow.extend(flush_table())
            table_buf = []
            in_table = False
        # Code fences
        if line.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code_buf), codeblock))
                code_buf = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        # Tables
        if line.lstrip().startswith("|"):
            in_table = True
            table_buf.append(line)
            continue
        if not line.strip():
            flow.append(Spacer(1, 4))
            continue
        if line.startswith("# "):
            # If the previous flow item was a quality block, separate visually.
            if last_was_quality:
                flow.append(divider)
            flow.append(Paragraph(esc(line[2:].strip()), h1))
            flow.append(divider)
            last_was_quality = False
        elif line.startswith("## "):
            flow.append(Spacer(1, 4))
            flow.append(Paragraph(esc(line[3:].strip()), h2))
            last_was_quality = False
        elif line.startswith("### "):
            flow.append(Paragraph(esc(line[4:].strip()), h3))
        elif line.startswith("#### "):
            flow.append(Paragraph(esc(line[5:].strip()), h4))
        elif line.lstrip().startswith("> "):
            text = md_inline_to_rl(line.lstrip("> ").strip())
            if text.startswith("**Quality"):
                # Promote Argus quality blockquote to the tinted callout.
                flow.append(Paragraph(text, quality))
                last_was_quality = True
            else:
                flow.append(Paragraph(text, quote))
                last_was_quality = False
        elif line.lstrip().startswith("- "):
            text = md_inline_to_rl(line.lstrip("- ").strip())
            flow.append(Paragraph(text, bullet, bulletText="•"))
        elif line.lstrip()[:2].isdigit() and line.lstrip()[2:4] == ". ":
            # numbered list
            text = md_inline_to_rl(line.strip())
            flow.append(Paragraph(text, bullet, bulletText="•"))
        elif line.strip() == "---":
            flow.append(HRFlowable(width="60%", thickness=0.6,
                                    color=DIVIDER, spaceBefore=8, spaceAfter=8,
                                    hAlign="CENTER"))
        elif line.startswith("_") and line.endswith("_"):
            flow.append(Paragraph(md_inline_to_rl(line), meta))
        else:
            flow.append(Paragraph(md_inline_to_rl(line), body))

    # Tail-flushes.
    if in_table:
        flow.extend(flush_table())
    if in_code and code_buf:
        flow.append(Preformatted("\n".join(code_buf), codeblock))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=title or "Argus report",
        # Show the report topic in the PDF metadata so Acrobat / Preview
        # show "Argus - <topic>" in the title bar.
        author="Argus (coding-app)",
        subject=f"Length mode: {title or 'report'}",
    )
    doc.build(flow)


# LangChain tool wrappers (so the graph can call these as @tool).
def make_langchain_tools():
    from langchain_core.tools import tool

    @tool("harvest_sources", parse_docstring=True)
    def t_harvest(hours: int = 72, top: int = 8,
                  sections: str = "papers,repos,news,blogs") -> dict:
        """Pull primary-source items (papers, repos, news, blogs) from the
        intel-radar harvester. Returns a JSON-serialisable dict."""
        r = harvest_sources(hours=hours, top=top, sections=sections)
        return r.model_dump()

    @tool("snatch_url", parse_docstring=True)
    def t_snatch(url: str, kind: str = "auto") -> dict:
        """Download a single URL (paper/article/media) and convert to markdown."""
        return snatch_url(url, kind=kind).model_dump()

    @tool("crawl_url", parse_docstring=True)
    def t_crawl(url: str, deep: bool = False, max_pages: int = 8) -> dict:
        """Crawl a documentation site or SPA with crawl4ai and return markdown."""
        return crawl_url(url, deep=deep, max_pages=max_pages).model_dump()

    @tool("normalize_markdown", parse_docstring=True)
    def t_normalize(source: str) -> dict:
        """Convert a URL or local file into clean markdown via article_convert."""
        return normalize_to_markdown(source).model_dump()

    return [t_harvest, t_snatch, t_crawl, t_normalize]