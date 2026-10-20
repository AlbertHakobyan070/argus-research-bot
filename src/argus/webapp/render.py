"""Report markdown → HTML for the console (v4 spec §2.7, Review Focus 4).

python-``markdown`` (``tables``, ``fenced_code``, ``sane_lists``) → a stdlib
allowlist sanitiser → every ``[n]`` in a text node outside code wrapped as
``<span class="cite" data-n="n">``. The report is fetched web content, so
the sanitiser keeps only the tags and attributes listed here, links only
with an ``http(s):``, ``mailto:`` or ``#`` target, drops ``<script>`` and
``<style>`` with their content, and escapes all text. The page CSP (no
inline script) is the backstop."""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser

import markdown

from ..graph.assemble import CITE_RE

ALLOWED_TAGS = frozenset({
    "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "pre", "code",
    "em", "strong", "b", "i", "del", "sup", "sub", "a", "table", "thead", "tbody", "tr", "th", "td", "span"})
ALLOWED_ATTRS: dict[str, frozenset[str]] = {"a": frozenset({"href", "title"}), "ol": frozenset({"start"})}
_VOID = frozenset({"br", "hr"})
_DROP_CONTENT = frozenset({"script", "style"})
_NO_CITE = frozenset({"code", "pre"})
_SAFE_HREF = re.compile(r"(?:https?:|mailto:|#)", re.IGNORECASE)
_INVISIBLE = re.compile(r"[\x00-\x20\x7f]+")      # browsers ignore these inside a URL scheme
_CITE = '<span class="cite" data-n="{0}">[{0}]</span>'


class _Sanitiser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.open: list[str] = []
        self.dropping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP_CONTENT:
            self.dropping += 1
            return
        if self.dropping or tag not in ALLOWED_TAGS:
            return
        kept = []
        for name, value in attrs:
            if value is None or name not in ALLOWED_ATTRS.get(tag, ()):
                continue
            if name == "href" and not _SAFE_HREF.match(_INVISIBLE.sub("", value)):
                continue
            kept.append(f' {name}="{html.escape(value)}"')
        if tag == "a":
            kept.append(' rel="noopener noreferrer"')
        self.out.append(f"<{tag}{''.join(kept)}>")
        if tag not in _VOID:
            self.open.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_CONTENT:
            self.dropping = max(self.dropping - 1, 0)
            return
        if self.dropping or tag not in self.open:
            return
        while self.open:                          # close what it left open: the output stays well-formed
            closed = self.open.pop()
            self.out.append(f"</{closed}>")
            if closed == tag:
                return

    def handle_data(self, data: str) -> None:
        if self.dropping:
            return
        text = html.escape(data, quote=False)
        if not _NO_CITE.intersection(self.open):
            text = CITE_RE.sub(lambda m: _CITE.format(m.group(1)), text)
        self.out.append(text)

    def result(self) -> str:
        self.close()
        return "".join(self.out) + "".join(f"</{t}>" for t in reversed(self.open))


def sanitise(fragment: str) -> str:
    """``fragment`` with only the allowlisted tags and attributes, and its
    ``[n]`` citations wrapped outside code."""
    parser = _Sanitiser()
    parser.feed(fragment)
    return parser.result()


def render_report(md_text: str) -> str:
    return sanitise(markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"]))
