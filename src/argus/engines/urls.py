"""URL identity (v4 spec §2.8): the canonical dedupe key, the host, the
registrable domain. Pure functions — no I/O."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = re.compile(r"utm_.*|ref|fbclid|gclid")
_ARXIV_HOSTS = {"arxiv.org", "export.arxiv.org", "ar5iv.org", "ar5iv.labs.arxiv.org"}
_ARXIV_ID = r"(\d{4}\.\d{4,5}|[a-z][a-z.\-]*/\d{7})"
_ARXIV_PAGE = re.compile(rf"/(?:abs|pdf|html)/{_ARXIV_ID}")          # any version / .pdf suffix after it
_HF_PAPER = re.compile(r"/papers/(\d{4}\.\d{4,5})")
# Second-level suffixes under which the registrable domain keeps three labels.
# ponytail: a small hand list, not the Public Suffix List; add a suffix when a run shows one.
_SECOND_LEVEL = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.nz", "ac.nz", "co.jp", "ac.jp", "co.kr", "ac.kr", "com.br", "com.cn", "edu.cn",
    "co.in", "ac.in", "co.za", "com.sg", "com.hk", "co.il",
    "github.io", "readthedocs.io",          # one project per subdomain, not one publisher
})


def _host(parts) -> str:
    return (parts.hostname or "").removeprefix("www.")


def canonical_url(url: str) -> str:
    """Lowercase host without ``www.``; no default port, trailing slash,
    ``#fragment`` or tracking parameter (``utm_*``, ``ref``, ``fbclid``,
    ``gclid``); ``http`` → ``https``; ``export.arxiv.org`` → ``arxiv.org``;
    arXiv ``abs|pdf|html`` (any version) and ar5iv pages and Hugging Face
    paper pages → ``https://arxiv.org/abs/<id>``; ``tab`` dropped on
    github.com only (repo pages carry ``?tab=readme-ov-file``). Anything that
    is not http(s) — ``file:///`` appends — is returned as given."""
    raw = url.strip()
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        return raw
    host = _host(parts)
    page = _HF_PAPER if host == "huggingface.co" else _ARXIV_PAGE if host in _ARXIV_HOSTS else None
    m = page.match(parts.path) if page else None
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"
    if host == "export.arxiv.org":
        host = "arxiv.org"
    netloc = host if parts.port in (None, 80, 443) else f"{host}:{parts.port}"
    drop = (lambda k: _TRACKING.fullmatch(k) or k == "tab") if host == "github.com" else _TRACKING.fullmatch
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not drop(k)])
    return urlunsplit(("https", netloc, parts.path.rstrip("/"), query, ""))


def domain(url: str) -> str:
    """The lowercase host without ``www.`` or port; ``""`` for non-http URLs."""
    parts = urlsplit(url.strip())
    return _host(parts) if parts.scheme in ("http", "https") else ""


def registrable_domain(url: str) -> str:
    """The last two labels of the host (three under a second-level suffix
    such as ``co.uk``); an IP address or a single-label host as is."""
    host = domain(url)
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = host.split(".")
    keep = 3 if ".".join(labels[-2:]) in _SECOND_LEVEL else 2
    return ".".join(labels[-keep:])


__all__ = ["canonical_url", "domain", "registrable_domain"]
