"""Anchored quote verification (v4 spec §2.8, B4): normalised exact
substring, else word 4-gram anchors → the matched-span SequenceMatcher
similarity on windows around the top 3 anchors → ``≥ quote_verify_fuzzy``
**and** every number token and word of the quote a token of that window (the
WP-B review's token rule). The bound: ≤ 50 ms per 250-char quote on a
60 000-char document."""
from __future__ import annotations

import json
import random
import time

import pytest

from argus.graph.research import QuoteIndex, _digest_reply, _norm_quote, quote_verified

FUZZY = 0.9


def _doc(n: int = 60_000, seed: int = 7) -> str:
    rnd = random.Random(seed)
    words = ["".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rnd.randint(2, 9)))
             for _ in range(4000)]
    text, sentence = [], []
    while sum(len(w) + 1 for w in text) < n:
        w = rnd.choice(words)
        sentence.append(w)
        if len(sentence) >= rnd.randint(8, 20):
            text.append(" ".join(sentence).capitalize() + ".")
            sentence = []
    return " ".join(text)[:n]


DOC = _doc()


def _quote_at(start: int, length: int = 250) -> str:
    """About ``length`` chars of whole words: a word cut in half is no word of the
    document, and the token rule rejects it."""
    s = DOC.index(" ", start) + 1
    return DOC[s:s + length + 12].rsplit(" ", 1)[0]


def _perturb(q: str) -> str:
    """What a digest LLM does to a quote: curly quotes, case, spacing, a dropped word."""
    words = q.split()
    del words[len(words) // 2]
    out = " ".join(words).replace(" ", "  ", 2)
    return "“" + out[:40].upper() + out[40:] + "”"


def _miss() -> str:
    return _doc(400, seed=99)[:250]


def test_exact_fuzzy_and_miss_cases():
    idx = QuoteIndex(DOC)
    exact = _quote_at(31_000)
    assert quote_verified(exact, idx, FUZZY) is True
    assert quote_verified(_perturb(exact), idx, FUZZY) is True, "small LLM edits still verify"
    short = _quote_at(12_000, 90)
    assert quote_verified(_perturb(short), idx, FUZZY) is True, "a short paraphrased quote verifies too"
    assert quote_verified(_miss(), idx, FUZZY) is False
    assert quote_verified("", idx, FUZZY) is None, "no quote → not checked"
    assert quote_verified("   ", idx, FUZZY) is None


def test_normalisation_of_quotes_case_and_whitespace():
    idx = QuoteIndex("He said “the interrupt() call pauses the graph” — and resumes later.")
    assert quote_verified('"The INTERRUPT() call   pauses the graph"', idx, FUZZY) is True
    assert quote_verified("the graph pauses forever", idx, FUZZY) is False


def test_threshold_is_the_knob():
    """Every word in place (one in three repeated): only the similarity decides."""
    idx = QuoteIndex(DOC)
    stuttered = []
    for i, w in enumerate(_quote_at(5_000).split()):
        stuttered += [w, w] if i % 3 == 2 else [w]
    mangled = " ".join(stuttered)
    assert quote_verified(mangled, idx, FUZZY) is False
    assert quote_verified(mangled, idx, 0.5) is True


@pytest.mark.parametrize("case", ["exact", "fuzzy", "miss"])
def test_anchored_verification_bound(case):
    quote = {"exact": _quote_at(31_000), "fuzzy": _perturb(_quote_at(31_000)), "miss": _miss()}[case]
    assert len(DOC) == 60_000 and len(quote) >= 240
    best = float("inf")
    for _ in range(3):                       # best of three: the bound is on the work, not on a noisy timer
        t0 = time.perf_counter()
        result = quote_verified(quote, QuoteIndex(DOC), FUZZY)
        best = min(best, time.perf_counter() - t0)
    assert result is (case != "miss")
    assert best <= 0.050, f"{case}: {best * 1000:.1f} ms for one 250-char quote incl. indexing the document"


# ---------------------------------------------------------------------------
# the token rule (WP-B review): the span ratio alone verified altered figures,
# swapped entities and "at most" → "at least" at 0.95–0.99
# ---------------------------------------------------------------------------

PROSE = (
    "Performance bound (test): ≤ 50 ms per 250-char quote on a 60 000-char document. "
    "Engines retry: a timeout is retried **at most `timeout_retry_max`** times before the attempt fails. "
    "In the baseline the v3 medium research node ran 82 s, deeper presets fetch 1.5–2.4× more sources. "
    "**Drives are plain `asyncio.create_task`** tracked by `runs.py` (never `application.create_task`, "
    "whose tasks PTB awaits in `stop()` before `post_stop`). "
    "ExaConfig(BaseModel): max_searches: int (≤ 60); max_contents_pages: int (≤ 200). "
    "The [interrupt function](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) pauses "
    "**the graph** at a node and surfaces a value to the caller, and the run is checkpointed. "
    "It’s checkpointed — nothing is lost while the graph waits for a human to answer the question. "
    "Presets are the shipped defaults; the bench tunes them, and every knob outside a preset keeps "
    "the value the model declares, so an override is recorded in overrides_applied and survives a rebuild. "
)
PROSE_IDX = QuoteIndex(_doc(30_000, seed=3) + " " + PROSE + _doc(30_000, seed=5))
PRESETS = ("Presets are the shipped defaults; the bench tunes them, and every knob outside a preset keeps "
           "the value the model declares, so an override is recorded in overrides_applied and survives a rebuild.")


@pytest.mark.parametrize("label,quote", [
    ("one number changed", "Performance bound (test): ≤ 90 ms per 250-char quote on a 60 000-char document."),
    ("two numbers changed", "Performance bound (test): ≤ 90 ms per 250-char quote on a 80 000-char document."),
    ("at most → at least", "a timeout is retried at least timeout_retry_max times before the attempt fails"),
    ("short, numbers changed", "the v3 medium research node ran 28 s, deeper presets fetch 3.5–4.4× more sources"),
    ("entities swapped", "Drives are plain trio.create_task tracked by jobs.py (never application.create_task, "
                         "whose tasks PTB awaits in stop() before post_stop)"),
    ("one entity swapped", "Drives are plain asyncio.create_task tracked by runs.py (never "
                           "application.create_task, whose tasks aiogram awaits in stop() before post_stop)"),
    ("config numbers changed", "ExaConfig(BaseModel): max_searches: int (≤ 90); max_contents_pages: int (≤ 500)."),
    ("verbatim head + fabricated tail", PRESETS[:150] + " which cut costs by 40% in 2025"),
    ("several 'the' → 'Google'", PRESETS.replace(" the ", " Google ")),
    ("one word swapped in a long quote", PRESETS.replace("recorded", "ignored")),
])
def test_a_substituted_word_or_number_never_verifies(label, quote):
    assert quote_verified(quote, PROSE_IDX, FUZZY) is False, label


@pytest.mark.parametrize("label,quote", [
    ("markup in the source: a link and bold",
     "The interrupt function pauses the graph at a node and surfaces a value to the caller"),
    ("markup in the source: bold and code",
     "a timeout is retried at most timeout_retry_max times before the attempt fails"),
    ("punctuation, dash and quote-char differences",
     "It's checkpointed - nothing is lost, while the graph waits for a human to answer the question"),
    ("one dropped word", "nothing is lost while the graph waits for a human to answer question"),
    ("a word elided with an ellipsis", "nothing is lost while the graph … for a human to answer the question"),
    ("dropped word and curly quotes in a long quote", "“" + PRESETS.replace(" outside a preset", " outside preset")
     + "”"),
])
def test_legitimate_fuzzy_quotes_still_verify(label, quote):
    assert quote_verified(quote, PROSE_IDX, FUZZY) is True, label


# ---------------------------------------------------------------------------
# WP-B fix round 2: the digest cuts a long quote at a word boundary
# ---------------------------------------------------------------------------

LONG = ("When the user extends a finished run, the research node grants one fresh allowance of fetches and "
        "Exa searches, widens the queries for every sub-question, and runs its waves again; the evidence it "
        "finds is merged with the earlier notes, the outline keeps its section titles, and only the sections "
        "whose sub-questions gained strong notes are rewritten before the report is delivered again.")


def _digested_quote(quote: str) -> str:
    reply = json.dumps({"relevance": 4, "claims": [{"text": "a claim", "quote": quote}]})
    return _digest_reply(reply)["claims"][0]["quote"]


def test_the_digest_cuts_a_long_quote_at_a_word_boundary():
    """``_digest_reply`` keeps a long quote up to the last whitespace at or before
    its limit, so a quote it shortened never ends in a fragment the token rule
    rejects. (At da8762f the hard cut split a word, and an edited quote failed.)"""
    idx = QuoteIndex(_doc(20_000, seed=11) + " " + LONG + " " + _doc(20_000, seed=13))
    edited = LONG.replace(",", "").replace(";", " -")              # a digest LLM's punctuation edits
    assert edited[249].isalpha() and edited[250].isalpha(), "a hard cut at 250 would split a word"
    cut = _digested_quote(edited)
    assert edited.startswith(cut) and len(cut) <= 250 and edited[len(cut)].isspace(), cut
    assert quote_verified(cut, idx, FUZZY) is True
    assert quote_verified(edited[:250], idx, FUZZY) is False, "the hard cut ends in a fragment"

    verbatim = _digested_quote(LONG)
    assert LONG.startswith(verbatim) and LONG[len(verbatim)].isspace()
    assert _norm_quote(verbatim) in idx.text and quote_verified(verbatim, idx, FUZZY) is True, (
        "a fragment-free cut quote verifies by exact substring, as before")
    assert _digested_quote("x" * 300) == "x" * 250, "a single token longer than the limit keeps the hard cut"
