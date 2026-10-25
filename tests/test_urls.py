"""engines/urls.py (v4 spec §2.8): the canonical dedupe key, domain and
registrable domain. Pure functions, no I/O."""
from __future__ import annotations

import pytest

from argus.engines.urls import canonical_url, domain, registrable_domain


@pytest.mark.parametrize("url,canonical", [
    ("https://www.Example.com/a/", "https://example.com/a"),                    # host case, www., trailing slash
    ("http://example.com:80/a#frag", "https://example.com/a"),                  # http → https, port, fragment
    ("https://example.com:443/a/", "https://example.com/a"),
    ("https://example.com:8443/a", "https://example.com:8443/a"),               # a non-default port stays
    ("https://example.com/", "https://example.com"),
    ("https://example.com/a?utm_source=x&id=3&ref=hn&fbclid=1&gclid=2&utm_medium=y",
     "https://example.com/a?id=3"),                                              # tracking params
    ("https://export.arxiv.org/abs/2401.12345", "https://arxiv.org/abs/2401.12345"),
    ("http://export.arxiv.org/api/query?search_query=x", "https://arxiv.org/api/query?search_query=x"),
    ("https://arxiv.org/abs/2401.12345v3", "https://arxiv.org/abs/2401.12345"),  # version
    ("https://arxiv.org/pdf/2401.12345v2", "https://arxiv.org/abs/2401.12345"),
    ("https://arxiv.org/pdf/2401.12345.pdf", "https://arxiv.org/abs/2401.12345"),
    ("https://arxiv.org/html/2401.12345v1", "https://arxiv.org/abs/2401.12345"),
    ("http://arxiv.org/abs/2503.16581v1", "https://arxiv.org/abs/2503.16581"),   # the arXiv Atom <id> form
    ("https://arxiv.org/abs/hep-th/9901001v2", "https://arxiv.org/abs/hep-th/9901001"),
    ("https://ar5iv.labs.arxiv.org/html/2401.12345", "https://arxiv.org/abs/2401.12345"),
    ("https://ar5iv.org/abs/2401.12345v2", "https://arxiv.org/abs/2401.12345"),
    ("https://huggingface.co/papers/2401.12345", "https://arxiv.org/abs/2401.12345"),
    ("https://huggingface.co/meta-llama/Llama-3", "https://huggingface.co/meta-llama/Llama-3"),
    ("https://github.com/langchain-ai/langgraph?tab=readme-ov-file",
     "https://github.com/langchain-ai/langgraph"),                               # github.com only
    ("https://www.github.com/a/b?tab=readme-ov-file&x=1", "https://github.com/a/b?x=1"),
    ("https://example.com/page?tab=2", "https://example.com/page?tab=2"),        # tab kept elsewhere
    ("file:///A:/vault/note.md", "file:///A:/vault/note.md"),                    # appends stay as they are
])
def test_canonical_url_table(url, canonical):
    assert canonical_url(url) == canonical


@pytest.mark.parametrize("url,canonical", [
    ("http://[::1]:8080/x/", "https://[::1]:8080/x"),
    ("https://[2001:DB8::1]/a/?utm_source=x", "https://[2001:db8::1]/a"),
    ("https://[2001:db8::1]:443/a", "https://[2001:db8::1]/a"),
])
def test_canonical_url_keeps_ipv6_brackets(url, canonical):
    assert canonical_url(url) == canonical


@pytest.mark.parametrize("url", [
    "http://example.com:99999/x",          # port out of range
    "http://example.com:abc/x",            # not a port
    "http://[::1/x",                       # unclosed bracket
    "https://arxiv.org:99999/abs/2401.12345",
])
def test_canonical_url_raises_on_a_malformed_url(url):
    with pytest.raises(ValueError):
        canonical_url(url)


def test_canonical_url_merges_the_v3_duplicates():
    dupes = ["https://www.Example.com/a/?utm_source=x", "http://example.com/a#x",
             "https://arxiv.org/abs/2401.00001v3", "https://export.arxiv.org/pdf/2401.00001",
             "https://huggingface.co/papers/2401.00001",
             "https://github.com/o/r", "https://github.com/o/r?tab=readme-ov-file"]
    assert len({canonical_url(u) for u in dupes}) == 3


@pytest.mark.parametrize("url,host,registrable", [
    ("https://www.Docs.Example.co.uk:8080/x", "docs.example.co.uk", "example.co.uk"),
    ("https://a.b.example.com/x", "a.b.example.com", "example.com"),
    ("https://cs.ox.ac.uk/p", "cs.ox.ac.uk", "ox.ac.uk"),
    ("https://abc.net.au/news", "abc.net.au", "abc.net.au"),
    ("https://langchain-ai.github.io/langgraph/", "langchain-ai.github.io", "langchain-ai.github.io"),
    ("https://example.com", "example.com", "example.com"),
    ("http://192.168.1.10:3001/v1", "192.168.1.10", "192.168.1.10"),
    ("http://localhost:3001/v1", "localhost", "localhost"),
    ("file:///A:/vault/note.md", "", ""),
])
def test_domain_and_registrable_domain(url, host, registrable):
    assert domain(url) == host
    assert registrable_domain(url) == registrable
