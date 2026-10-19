"""Anchored quote verification (v4 spec §2.8, B4): normalised exact
substring, else word 4-gram anchors → SequenceMatcher on windows around the
top 3 anchors → ``≥ quote_verify_fuzzy``. The bound: ≤ 50 ms per 250-char
quote on a 60 000-char document."""
from __future__ import annotations

import random
import time

import pytest

from argus.graph.research import QuoteIndex, quote_verified

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
    s = DOC.index(" ", start) + 1
    return DOC[s:s + length].strip()


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
    idx = QuoteIndex("He said “the interrupt() call pauses the graph” — and resumes later.")
    assert quote_verified('"The INTERRUPT() call   pauses the graph"', idx, FUZZY) is True
    assert quote_verified("the graph pauses forever", idx, FUZZY) is False


def test_threshold_is_the_knob():
    idx = QuoteIndex(DOC)
    words = _quote_at(5_000).split()
    for i in range(0, len(words), 5):        # one word in five replaced: anchors survive, similarity ≈ 0.8
        words[i] = "q" * len(words[i])
    mangled = " ".join(words)
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
