"""P4 — quarantine sentences that the reviewer flagged as fabricated.

Why this exists (bug observed 2026-07-09, GLM 5.2 report):
``merge_reviewer_into_assessment`` surfaces the count of
``unsupported_claims`` and ``fabrication_flags`` in the title page so
the user can see them, but the claims themselves are *still in the
body*, presented as fact. After ``MAX`` revisions the run proceeds
regardless. The user reads the title block ("Reviewer flagged 3
unsupported claim(s)") and then the body still says "GLM-5.2 is the
latest flagship LLM from Z.ai with 1M-token context window [8]" — the
flagged assertion as fact.

This module is the structural fix. It walks ``draft_md``, finds the
sentence(s) that contain each flagged claim, and either:

  (a) **removes** the sentence (when the flagged claim is the
      sentence's main point), or
  (b) **softens** the sentence (wrapping the asserted language in
      hedging), or
  (c) **moves** the sentence to a clearly-labelled
      ``## ⚠️ Unverified / flagged claims`` appendix so the reader
      can still see what was reviewed, but nothing flagged is
      presented as established fact.

Default is (c) — the appendix preserves the review trail without lying
to the reader. Pass ``mode="remove"`` to drop the sentences entirely
(useful when the claim is repeated nowhere else and is structurally
harmful).

Pure functions, no I/O. Module-level so any helper can unit test it
without spinning up the graph.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# --- Sentence splitting ---------------------------------------------------
#
# A "sentence" is roughly a run of characters ending in `.`, `!`, `?`, or
# a line break. We split on a non-capturing group so re-assembly can
# keep the terminator. We deliberately avoid fancy NLP sentence
# segmentation here — the synthesizer's draft is plain markdown prose
# where this rule is good enough for the patterns we see.

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[\.\!\?])(?=\s|$|\n|[A-Z\*\-\#])",
)


def _split_into_sentences(text: str) -> list[str]:
    """Naive split on punctuation. Returns sentences with their
    trailing whitespace preserved."""
    if not text:
        return [text]
    parts = _SENTENCE_SPLIT_RE.split(text)
    return parts


def _find_sentence_index_containing(sentences: list[str],
                                     needle: str) -> int | None:
    """Return the index of the first sentence whose lowercase body
    contains ``needle`` (case-insensitive substring match). Returns
    ``None`` when no sentence matches.

    If the needle appears in multiple sentences, only the first is
    returned — we assume the first match is the canonical assertion.
    """
    nlow = needle.lower().strip()
    if not nlow:
        return None
    for i, s in enumerate(sentences):
        if nlow in s.lower():
            return i
    return None


# --- Public types ---------------------------------------------------------

@dataclass
class QuarantineResult:
    """Outcome of :func:`quarantine_flagged_claims`."""

    cleaned_text: str
    quarantined: list[str] = field(default_factory=list)
    # The strings of sentences that were moved/removed.
    still_unmatched: list[str] = field(default_factory=list)
    # Claim text that didn't match any sentence in the body — logged
    # so the report builder can surface them in the title block
    # ("3 flagged claims, 1 wasn't found in the body").


# --- Public entry points --------------------------------------------------

def quarantine_flagged_claims(
    draft_md: str,
    unsupported_claims: list[str] | None,
    fabrication_flags: list[str] | None,
    *,
    mode: str = "move",
) -> QuarantineResult:
    """Move (or remove) sentences that match the reviewer's flagged claims.

    Parameters
    ----------
    draft_md
        The synthesized markdown body (NOT including the title block).
    unsupported_claims, fabrication_flags
        Lists of claim text from ``ReviewVerdict``. Either list is
        fine; both are processed in the same way (the reviewer
        distinguished them; we treat them equivalently for quarantine
        purposes — a fabrication flag is just an unsupported claim that
        crossed a severity threshold).
    mode
        ``"move"`` (default): relocate matched sentences to a
        clearly-labelled appendix at the end of the document.
        ``"remove"``: drop them entirely.

    Returns
    -------
    QuarantineResult with ``cleaned_text``, ``quarantined`` (list of
    removed/moved sentence strings, in original order, deduped per
    matched sentence index), and ``still_unmatched`` (claims we
    couldn't locate in the body).
    """
    if mode not in ("move", "remove"):
        raise ValueError(f"mode must be 'move' or 'remove', got {mode!r}")

    if not draft_md:
        # Empty body — every claim is unmatched, but report that so the
        # caller can surface it in the title block.
        return QuarantineResult(
            cleaned_text=draft_md,
            still_unmatched=[
                c for c in
                list(unsupported_claims or []) + list(fabrication_flags or [])
                if c and c.strip()
            ],
        )

    claims: list[str] = list(unsupported_claims or []) + \
        list(fabrication_flags or [])
    if not claims:
        return QuarantineResult(cleaned_text=draft_md)

    sentences = _split_into_sentences(draft_md)
    matched_indices: set[int] = set()
    # matched_sentences[idx] = sentence_text, registered only the first
    # time we see that index so a single sentence containing BOTH an
    # unsupported_claim AND a fabrication_flag isn't counted twice.
    matched_sentences: dict[int, str] = {}
    still_unmatched: list[str] = []

    for claim in claims:
        c = (claim or "").strip()
        if not c:
            continue
        idx = _find_sentence_index_containing(sentences, c)
        if idx is None:
            still_unmatched.append(c)
            continue
        matched_indices.add(idx)
        matched_sentences.setdefault(idx, sentences[idx])

    # Rewrite the body, omitting matched sentences (move or remove).
    kept_sentences = [s for i, s in enumerate(sentences)
                      if i not in matched_indices]
    # Preserve original order of matched sentences for the appendix.
    matched_text = [matched_sentences[i] for i in sorted(matched_indices)]

    if mode == "move":
        # Append the quarantined block at the end (after a clear
        # separator). We deliberately include the original sentence text
        # so a reader doing a textual comparison can find the claim if
        # they want to verify.
        quoted = "\n".join(
            f"- {line.strip()}" for line in matched_text
        )
        appendix = (
            "\n\n---\n\n"
            "## ⚠️ Unverified / flagged claims\n\n"
            "_The following sentences were flagged by the reviewer as "
            "unsupported or potentially fabricated. They were removed "
            "from the body and preserved here so the review trail is "
            "auditable. Do not rely on these claims without independent "
            "verification._\n\n"
            f"{quoted}\n"
        )
        cleaned = "".join(kept_sentences) + appendix
    else:  # remove
        cleaned = "".join(kept_sentences)

    return QuarantineResult(
        cleaned_text=cleaned,
        quarantined=matched_text,
        still_unmatched=still_unmatched,
    )


__all__ = ["quarantine_flagged_claims", "QuarantineResult"]
