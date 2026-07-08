"""Tests for src/argus/sources_block.py (P3: empty [N] slot renumbering).

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_sources_block.py -q

The sanitize_sources_block helper is the post-pass that runs after
verify_citations. It drops dangling source lines (`[N] ` with no URL)
and renumbers body refs so the report no longer points at empty slots.
"""
from __future__ import annotations

import pytest

from argus.sources_block import sanitize_sources_block


SAMPLE_REPORT = """# GLM 5.2

Some prose that cites [1] and [8].

## TL;DR

GLM-5.2 is the latest flagship LLM [2] with a 1M-token context window [8].

## Sources

[1] https://thetechbriefs.com/thudm-releases-glm-4-
[2]
[3] https://glm45.org/?
[5] http://arxiv.org/abs/2406.12793v2
[7]
[8] https://build.nvidia.com/z-ai/glm-5.2/modelcard
[9] http://arxiv.org/abs/2402.11651v2
[10] http://arxiv.org/abs/2312.10793v3
"""


def test_drops_dangling_source_lines():
    """[2] and [7] with no URL must be removed from the Sources block."""
    out = sanitize_sources_block(SAMPLE_REPORT)
    body = out.cleaned_text
    # Dangling labels appear nowhere in the body either (they're stripped
    # from refs).
    for line in body.splitlines():
        s = line.strip()
        # No empty `[N]` line may survive in the Sources block.
        if s.startswith("[") and "]" in s:
            after = s.split("]", 1)[1].strip()
            assert after != "", f"dangling source survived: {line!r}"
    assert out.dropped_count == 2  # [2] and [7]


def test_renumbers_kept_sources_compactly():
    """[1], [3], [5], [8], [9], [10] become [1], [2], [3], [4], [5], [6]."""
    out = sanitize_sources_block(SAMPLE_REPORT)
    body = out.cleaned_text
    # Find the Sources block and parse.
    import re
    src_lines = [
        ln.strip() for ln in body.splitlines()
        if re.match(r"^\[\d+\]\s+https?://", ln.strip())
    ]
    # Now extract just the index of each source line.
    indices = [
        int(ln.split("]", 1)[0].lstrip("[")) for ln in src_lines
    ]
    assert indices == [1, 2, 3, 4, 5, 6]
    # Each new index must appear with its URL.
    assert "[1] https://thetechbriefs.com" in body
    assert "[2] https://glm45.org" in body
    assert "[3] http://arxiv.org/abs/2406.12793v2" in body
    assert "[4] https://build.nvidia.com/z-ai/glm-5.2/modelcard" in body


def test_body_citation_refs_renumbered():
    """The `[2]` and `[8]` refs in the body must map to the new compact
    indices (2 and 7 → 1 and 3 → 4 in the kept/renumbered set)."""
    out = sanitize_sources_block(SAMPLE_REPORT)
    body = out.cleaned_text
    # The original `[2]` (claiming 1M-token) referenced a now-dropped source.
    # That sentence must still contain the claim but the [2] ref is gone.
    # We don't change the prose — we just strip the orphaned ref.
    # The original `[8]` (also claiming 1M-token) becomes `[4]`.
    # The original `[1]` (techbriefs content farm) stays `[1]` (no shift).
    # The text in the body is:
    #   "cites [1] and [8]"           -> "cites [1] and [4]"
    #   "flag [2] with a 1M ..."      -> "flag with a 1M ..."  (orphan dropped)
    #   "context window [8]"          -> "context window [4]"
    assert "cites [1] and [4]" in body
    # The dropped `[2]` must not appear as a dangling label.
    # Allow "[2] https://" (kept source [3] renumbered to [2]) but not
    # "[2] " on its own.
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("[2]"):
            assert s.startswith("[2] http"), f"unexpected dangling [2] line: {line!r}"
    assert "context window [4]" in body


def test_clean_report_is_noop():
    """If there are no dangling slots, the renumber is a no-op (no churn)."""
    clean = """# Hello

Body cites [1].

## Sources

[1] https://example.com/a
"""
    out = sanitize_sources_block(clean)
    assert out.dropped_count == 0
    assert out.cleaned_text == clean


def test_report_with_no_sources_block_is_noop():
    """A report with no ## Sources heading is returned untouched."""
    noblock = "# Hello\n\nJust prose.\n"
    out = sanitize_sources_block(noblock)
    assert out.dropped_count == 0
    assert out.cleaned_text == noblock
    assert out.sources_block_found is False


def test_results_block_renumbers_references_heading_too():
    """Lecture mode uses ## References (not ## Sources). Drop a middle
    source and check the body's [4] ref points to the renumbered slot."""
    lecture = """# Lecture

Body cites [4].

## References

[1] https://a.com
[3]
[4] https://d.com
[10] https://j.com
"""
    out = sanitize_sources_block(lecture)
    body = out.cleaned_text
    # Source original [3] was dropped. Kept sources are [1], [4], [10]
    # which compact to [1], [2], [3]. Body's original [4] becomes [2].
    before_refs = body.split("## References", 1)[0]
    assert "cites [2]" in before_refs, (
        f"body [4] should renumber to [2] when source [3] dropped: {before_refs!r}"
    )
    # The dropped [3] must NOT appear as a dangling label.
    ref_block = body.split("## References", 1)[1]
    for line in ref_block.splitlines():
        s = line.strip()
        if s.startswith("[3]"):
            assert s != "[3]", f"dangling [3] survived in References block: {line!r}"


def test_multiple_sources_blocks_uses_last():
    """When two ## Sources blocks exist (rare), the LAST is sanitized and
    the earlier one is left untouched (it's almost certainly a section
    mid-prose, not the reference list)."""
    multi = """# Hello

## Sources (intermediate)

[1] https://intermediate.example.com

## Body

Body cites [1].

## Sources

[2] https://a.com
[3]
"""
    out = sanitize_sources_block(multi)
    assert out.sources_block_found is True
    # The empty `[3]` must be gone.
    cleaned_lower = out.cleaned_text.lower()
    # No `[3] ` (label-only) line in the final Sources block.
    block = out.cleaned_text.split("## Sources")[-1]
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("[3]"):
            assert s != "[3]", f"dangling [3] survived: {line!r}"


def test_natural_run_collected_keys():
    out = sanitize_sources_block(SAMPLE_REPORT)
    # Remap must be present (we dropped 2 sources).
    assert out.renumbered, "renumbered should be populated"
    # Original [2] is not in remap (was dropped).
    assert 2 not in out.renumbered
    assert 7 not in out.renumbered
    # All kept entries are mapped to compact indices 1..6.
    assert set(out.renumbered.values()) == set(range(1, 7))
