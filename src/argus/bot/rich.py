"""Report and /ask Markdown → Telegram rich-message HTML (Bot API 10.1,
``sendRichMessage``; the "Rich HTML style" and "Rich Message Limits"
sections of the Bot API docs).

python-``markdown`` as the console renders it (``webapp/render.py``) but with
raw HTML off — the text is LLM and web content, so a tag written in it ("the
``<details>`` element") shows as written; ``<br>`` alone stays a line break —
then the console's sanitiser with the rich-message allowlist: only the tags and
attribute values listed here, links only with an ``http(s):``, ``mailto:``,
``tel:`` or ``#`` target (an unsafe link keeps its text), every other tag
unwrapped (``thead``/``tbody`` included: rich tables take ``tr`` directly),
``<script>``/``<style>`` dropped with their content, all text escaped, and
each ``[n]`` outside code bold, as the PDF prints it.

A message over a limit is never sent: ``report_html`` falls back to a
condensed report, then to None, as does ``answer_html``.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser

import markdown
from markdown.inlinepatterns import SubstituteTagInlineProcessor

from ..graph.assemble import CITE_RE
from ..webapp.render import _DROP_CONTENT, _INVISIBLE, _NO_CITE, _VOID, _Sanitiser

MAX_CHARS = 32768       # of text; the whole HTML's len() bounds it
MAX_BLOCKS = 500
MAX_DEPTH = 16
MAX_COLUMNS = 20

_NUM = re.compile(r"\d{1,4}")
_CELL = {"colspan": _NUM, "rowspan": _NUM, "align": re.compile("left|center|right"),
         "valign": re.compile("top|middle|bottom")}
_INLINE = frozenset({"b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "code", "mark", "sub", "sup",
                     "tg-spoiler", "a", "br"})
TAGS = _INLINE | frozenset({
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "footer", "hr", "ul", "ol", "li", "blockquote", "cite",
    "aside", "table", "caption", "tr", "th", "td", "details", "summary"})
# tag → {attribute: the pattern its whole value must match; None: a flag, written bare}
ATTRS: dict[str, dict[str, re.Pattern | None]] = {
    "a": {"href": re.compile(r"(?:https?:|mailto:|tel:|#)\S*", re.IGNORECASE), "name": re.compile(r"[\w.-]+")},
    "ol": {"start": _NUM, "type": re.compile("[1aAiI]"), "reversed": None},
    "li": {"value": _NUM, "type": re.compile("[1aAiI]")},
    "code": {"class": re.compile(r"language-[\w+#.-]+")},      # inside <pre> only
    "table": {"bordered": None, "striped": None, "compact": None},
    "th": _CELL, "td": _CELL,
    "blockquote": {"expandable": None},
    "details": {"open": None},
}
_MD_OPTS = {"extensions": ["tables", "fenced_code", "sane_lists"],
            "extension_configs": {"tables": {"use_align_attribute": True}}}
# Raw HTML off (block and inline), but an LLM's <br> (table cells) is still a line break.
# ponytail: one shared instance, reset per conversion; fine on the event loop, one per thread if ever threaded.
_MD = markdown.Markdown(**_MD_OPTS)
_MD.preprocessors.deregister("html_block")
_MD.inlinePatterns.deregister("html")
_MD.inlinePatterns.register(SubstituteTagInlineProcessor(r"(?i)<br\s*/?>", "br"), "br", 90)
_NOT_BLOCKS = frozenset({"summary", "caption", "cite", "th", "td"})     # parts of a block (rows count, not cells)
_LINK = re.compile(r"\[(.*)\]\((\S*)\)")
_SOURCES = re.compile(r"^## Sources[ \t]*$", re.M)
_SECTION_END = re.compile(r"^(?:#{1,2} |-{3,}[ \t]*$)", re.M)    # the next h1/h2 or a rule (the footer's)
_H2 = re.compile(r"^## ", re.M)
_TLDR = re.compile(r"^## [^\n]*TL;DR", re.M)
_ITEM = re.compile(r"^[ \t]*(\d{1,4})\.[ \t]+(.+?)[ \t]*$", re.M)
_TOO_LONG = ("<p><i>The full report is too long for one message: it is in the attached "
             "report.md / report.pdf.</i></p>")


class _Rich(_Sanitiser):
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP_CONTENT:
            self.dropping += 1
            return
        if self.dropping or tag not in TAGS:
            return
        kept = []
        for name, value in attrs:
            allowed = ATTRS.get(tag, {})
            if name not in allowed:
                continue
            clean = _INVISIBLE.sub("", value or "")        # browsers ignore these inside a URL scheme
            if allowed[name] is None:
                kept.append(f" {name}")
            elif allowed[name].fullmatch(clean) and (tag != "code" or self.open[-1:] == ["pre"]):
                kept.append(f' {name}="{html.escape(clean)}"')
        if tag == "a" and not kept:               # no safe target: the link text stays, unlinked
            return
        self.out.append(f"<{tag}{''.join(kept)}>")
        if tag not in _VOID:
            self.open.append(tag)

    def handle_data(self, data: str) -> None:
        if self.dropping:
            return
        text = html.escape(data)
        if not _NO_CITE.intersection(self.open):
            text = CITE_RE.sub(r"<b>[\1]</b>", text)
        self.out.append(text)


class _Measure(HTMLParser):
    """Counts blocks as the Bot API does: every element but the inline ones
    and a block's parts (``_NOT_BLOCKS``), plus the implicit paragraph that
    holds a list item's text (every ``li`` counts 2) and the text of a
    blockquote or details of its own; nesting over all elements; a row's
    columns with their colspans."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks = self.deepest = self.columns = self.row = 0
        self.open: list[list] = []          # [tag, its own text's paragraph still uncounted]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _INLINE:
            self._own_text()
        else:
            self.blocks += (tag not in _NOT_BLOCKS) + (tag == "li")
        if tag == "tr":
            self.row = 0
        elif tag in ("td", "th"):
            self.row += int(dict(attrs).get("colspan") or 1)
            self.columns = max(self.columns, self.row)
        if tag not in _VOID:
            self.open.append([tag, tag in ("blockquote", "details")])
            self.deepest = max(self.deepest, len(self.open))

    def handle_endtag(self, tag: str) -> None:
        if tag not in _VOID:
            self.open.pop()

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._own_text()

    def _own_text(self) -> None:
        """Text or an inline element straight inside a blockquote or details:
        its implicit paragraph, counted once."""
        if self.open and self.open[-1][1]:
            self.blocks += 1
            self.open[-1][1] = False


def fits(doc: str) -> bool:
    """``doc`` within every rich-message limit."""
    m = _Measure()
    m.feed(doc)
    m.close()
    return (len(doc) <= MAX_CHARS and m.blocks <= MAX_BLOCKS and m.deepest <= MAX_DEPTH
            and m.columns <= MAX_COLUMNS)


def md_to_rich_html(md: str) -> str:
    parser = _Rich()
    parser.feed(_MD.reset().convert(md))
    return parser.result().strip()


def _source(text: str) -> str:
    """One source line: ``[title](url)`` as a link, else the line as is. The
    title is a page's own and always text (MDN's read ``<style>: …``); the
    sanitiser checks the href (unsafe: the title unlinked) and bolds ``[n]``."""
    title, url = m.groups() if (m := _LINK.fullmatch(text)) else (text, "")
    parser = _Rich()
    parser.feed(f'<a href="{html.escape(url)}">{html.escape(title)}</a>')
    return parser.result()


def _folded(section: str) -> str:
    """A ``## Sources`` body (``n. [title](url)`` lines) as a collapsed
    list keeping each source's own number; "" with no item."""
    items = _ITEM.findall(section)
    if not items:
        return ""
    lis = "".join(f'<li value="{n}">{_source(text)}</li>' for n, text in items)
    return f"<details><summary>Sources ({len(items)})</summary><ol>{lis}</ol></details>"


def report_html(md_text: str) -> str | None:
    """report.md as one rich message, its Sources folded; over a limit, the
    title block, the TL;DR, a pointer to the attached files and the folded
    Sources; None when even that breaks a limit."""
    before, folded, after = md_text, "", ""
    if m := _SOURCES.search(md_text):
        end = _SECTION_END.search(md_text, m.end())
        stop = end.start() if end else len(md_text)
        before, folded, after = md_text[:m.start()], _folded(md_text[m.end():stop]), md_text[stop:]
    full = md_to_rich_html(before) + folded + md_to_rich_html(after)
    if fits(full):
        return full
    head = _H2.search(before)
    short_md = before[:head.start()] if head else before
    if t := _TLDR.search(before):
        nxt = _H2.search(before, t.end())
        short_md += "\n\n" + before[t.start():nxt.start() if nxt else len(before)]
    short = md_to_rich_html(short_md) + _TOO_LONG + folded
    return short if fits(short) else None


def answer_html(md: str) -> str | None:
    """A /ask answer as a rich message; None when empty or over a limit."""
    doc = md_to_rich_html(md)
    return doc if doc and fits(doc) else None
