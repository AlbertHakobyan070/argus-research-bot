"""Verbatim quote verification (v4 spec §2.8, B4): the document and the quote
are normalised alike — case, whitespace, quote marks, dashes, markdown links and
images (their text), footnote markers, ordered-list and blockquote markers at
line starts, ``<=`` ``>=`` ``!=`` as ``≤`` ``≥`` ``≠`` — and split into tokens: whole numbers
(sign, ``.``/``,``/``:`` between digits, ``%``), the symbols ``< > ≤ ≥ ≠ ± =`` and
``\\w+`` words. A quote verifies when its tokens occur contiguously and in order in
the document's, the first segment not right after a negator; an ellipsis splits it
into at most 8 segments that must each occur so, in order, each starting within
3 000 chars of the previous one's end. Any substituted, added or dropped word,
number or symbol fails. The bounds: ≤ 50 ms per 250-char quote on a 60 000-char
document; indexing stays linear."""
from __future__ import annotations

import json
import random
import time

import pytest

from argus.graph.research import QuoteIndex, _digest_reply, quote_verified


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
    """About ``length`` chars of whole words from ``DOC``."""
    s = DOC.index(" ", start) + 1
    return DOC[s:s + length + 12].rsplit(" ", 1)[0]


def _miss() -> str:
    return _doc(400, seed=99)[:250]


def test_exact_miss_and_no_quote():
    idx = QuoteIndex(DOC)
    exact = _quote_at(31_000)
    assert quote_verified(exact, idx) is True
    assert quote_verified(exact.upper(), idx) is True, "case is normalised"
    words = exact.split()
    assert quote_verified(" ".join(words[:5] + ["zzz"] + words[5:]), idx) is False, "an added word"
    assert quote_verified(" ".join(words[:5] + words[6:]), idx) is False, "a dropped word"
    assert quote_verified(_miss(), idx) is False
    for none in ("", "   ", "…", "— ..."):
        assert quote_verified(none, idx) is None, (none, "no words → not checked")


# ---------------------------------------------------------------------------
# the review rounds' cases: every altered quote fails, every legitimate one passes
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
    "State is _plain_ data: the checkpointer stores every superstep[12] so a crash loses nothing.\n"
    "To deploy:\n"
    "1. Install the package.\n"
    "2) Start the server with the token.\n"
    "In this release the scheduler does not retry failed jobs automatically after a timeout. "
    "The model does support streaming responses for every request, but it does not support batching. "
    "Revenue rose from 5% of sales in 2023 to 12% of sales in 2024 across the whole group. "
    "In the benchmark the OpenAI model outperformed the Google model on every reasoning task we ran. "
    "A request is retried at most three times, and at least one attempt is always logged for audit. "
    # WP-B fix round 4: numbers, symbols and negators
    "The latency bound is <= 20 ms on every route, and p < 0.05 for every arm of the trial. "
    "Profit fell to -5% in 2024, while growth of 0.5% of users came from referrals. "
    "We served 1,000 users and 1,000 requests per second in 2024. "
    "COVID-19 cases rose between 2019–2020 in the region; the ratio was 3.14 at its peak. "
    "The drug is not effective in children. The update ships without breaking changes. "
    "Users don't need an account to read the docs. "
    "Version 2 does not support webhooks; version 3 does support webhooks. "
    # WP-B fix round 5: blockquote markers at line starts; a mid-line ">" stays a symbol
    "Throughput must stay >= 300 requests per second at peak. The first arm passes when x > 5 holds for every sample.\n"
    "> **Note**: Python 3.11 or newer is required for the Functional API, for interrupt(), and for streaming\n"
    "> from a node running in the Workflow. On older versions the plugin loads but emits a warning,\n"
    "> and those features will not work.\n"
    ">> A nested reply: the checkpointer stores every superstep\n"
    "> > so that a restart loses nothing at all.\n"
    "> 1. Clone the repository.\n"
    "> 2. Run the installer.\n"
    # WP-B closeout: a number glued to letters is one word token
    "The startup raised $50M in 2024. The model has 70B parameters and is 10x faster on a CPU at 2.5GHz; its "
    "latency is 100ms, training used 8k GPUs, and it was the 2nd place finisher in the benchmark. "
)
PROSE_IDX = QuoteIndex(_doc(30_000, seed=3) + " " + PROSE + _doc(30_000, seed=5))
PRESETS = ("Presets are the shipped defaults; the bench tunes them, and every knob outside a preset keeps "
           "the value the model declares, so an override is recorded in overrides_applied and survives a rebuild.")


@pytest.mark.parametrize("label,quote", [
    # WP-B review (round 1, quote_probe.py)
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
    ("a reordered paraphrase", "an override survives a rebuild and is recorded in overrides_applied"),
    ("a fabricated claim sharing four words", "the run is checkpointed so the vendor guarantees zero data loss"),
    # WP-B re-review (round 2, p3_quote_extra.py): the replacement sits nearby
    ("a dropped 'not'", "In this release the scheduler does retry failed jobs automatically after a timeout."),
    ("an inserted 'not' that occurs nearby", "The model does not support streaming responses for every request"),
    ("numbers swapped within the sentence",
     "Revenue rose from 12% of sales in 2023 to 5% of sales in 2024 across the whole group."),
    ("entities swapped within the sentence",
     "In the benchmark the Google model outperformed the OpenAI model on every reasoning task we ran."),
    ("'at most' → 'at least', 'least' nearby",
     "A request is retried at least three times, and at least one attempt is always logged for audit."),
    ("a new number", "Revenue rose from 7% of sales in 2023 to 12% of sales in 2024 across the whole group."),
    # dropped and added words
    ("a dropped word", "nothing is lost while the graph waits for a human to answer question"),
    ("an added word", "nothing is ever lost while the graph waits for a human to answer the question"),
    # WP-B fix round 4 (the third re-review's residuals): numbers and symbols are whole tokens
    ("≤ → ≥", "Performance bound (test): ≥ 50 ms per 250-char quote on a 60 000-char document."),
    ("<= in the source, >= in the quote", "The latency bound is >= 20 ms on every route"),
    ("< → >", "p > 0.05 for every arm of the trial"),
    ("-5% → 5%", "Profit fell to 5% in 2024"),
    ("0.5% quoted as 5%", "5% of users came from referrals"),
    ("1,000 → 1.000", "1.000 requests per second in 2024"),
    # ... and an excerpt may not cut a negation off
    ("starts right after 'not'", "effective in children"),
    ("starts right after 'without'", "breaking changes"),
    ("starts right after 'don't'", "need an account to read the docs"),
    ("an ellipsis first, then right after 'not'", "… effective in children"),
    # WP-B fix round 5: stripping blockquote markers loosens nothing else
    ("a mid-line > → <", "passes when x < 5 holds for every sample"),
    (">= → <= at the start of a quote", "<= 300 requests per second at peak"),
    ("a blockquoted figure changed", "Python 3.12 or newer is required for the Functional API"),
    # WP-B closeout (the rounds 4–5 re-review): a quote may not drop a number's suffix
    ("$50M quoted as $50", "raised $50"),
    ("70B quoted as 70", "has 70"),
    ("10x quoted as 10", "is 10"),
    ("2.5GHz quoted as 2.5", "at 2.5"),
    ("100ms quoted as 100", "is 100"),
    ("8k quoted as 8", "used 8"),
    ("2nd quoted as nd", "nd place finisher"),
    # final review M17: a decimal glued to its unit is one token, never split at its separator
    ("2.5GHz quoted as 2", "faster on a CPU at 2"),
    ("2.5GHz quoted from its fraction", "5GHz; its latency is 100ms"),
])
def test_an_altered_quote_never_verifies(label, quote):
    assert quote_verified(quote, PROSE_IDX) is False, label


@pytest.mark.parametrize("label,quote", [
    ("exact", PRESETS),
    ("markup in the source: a link and bold",
     "The interrupt function pauses the graph at a node and surfaces a value to the caller"),
    ("markup in the source: bold and code",
     "a timeout is retried at most timeout_retry_max times before the attempt fails"),
    ("markup in the source: italics", "State is plain data"),
    ("quote marks, dashes, punctuation and whitespace",
     "“It's checkpointed - nothing is lost,  while the graph waits for a human to answer the question”"),
    ("a footnote marker in the source", "the checkpointer stores every superstep so a crash loses nothing"),
    ("ordered-list markers in the source", "Install the package. Start the server with the token."),
    ("ellipsis segments in order", "Presets are the shipped defaults … an override is recorded in overrides_applied"),
    ("a word elided with an ellipsis", "nothing is lost while the graph ... for a human to answer the question"),
    # WP-B fix round 4: number and symbol tokens
    ("a year before a full stop", "1,000 requests per second in 2024"),
    ("decimals", "p < 0.05 for every arm of the trial"),
    ("a decimal and a percentage", "growth of 0.5% of users came from referrals"),
    ("a signed percentage", "Profit fell to -5% in 2024"),
    ("a thousands separator", "We served 1,000 users"),
    ("COVID-19", "COVID-19 cases rose"),
    ("a year range, en dash in the source", "between 2019-2020 in the region; the ratio was 3.14 at its peak"),
    ("<= in the source, ≤ in the quote", "The latency bound is ≤ 20 ms on every route"),
    ("a quote that starts with the negator", "not effective in children"),
    ("another occurrence not after a negator", "support webhooks"),
    # WP-B fix round 5: blockquote markers at line starts are markup, not ">" tokens
    ("a sentence the source wraps across two blockquoted lines",
     "Python 3.11 or newer is required for the Functional API, for interrupt(), and for streaming from a node "
     "running in the Workflow."),
    ("... across three blockquoted lines", "streaming from a node running in the Workflow. On older versions the "
                                           "plugin loads but emits a warning, and those features will not work."),
    ("nested blockquote markers (>> and > >)",
     "A nested reply: the checkpointer stores every superstep so that a restart loses nothing at all."),
    ("a quote that keeps the blockquote marker", "> **Note**: Python 3.11 or newer is required for the Functional API"),
    ("a blockquoted ordered list", "Clone the repository. Run the installer."),
    ("a mid-line > in the source", "passes when x > 5 holds for every sample"),
    (">= at the start of a quote", ">= 300 requests per second at peak"),
    # WP-B closeout: a number glued to letters, quoted whole
    ("an amount with its magnitude suffix", "raised $50M"),
    ("an ordinal", "2nd place finisher"),
    ("a count with its unit", "70B parameters"),
    ("a decimal glued to its unit", "10x faster on a CPU at 2.5GHz; its latency is 100ms"),         # final M17
])
def test_a_legitimate_quote_verifies(label, quote):
    assert quote_verified(quote, PROSE_IDX) is True, label


NEGATORS = ["not", "never", "cannot", "can't", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "won't",
            "wouldn't", "shouldn't", "couldn't", "without",
            # WP-B closeout: the rest of the n't contractions (spec §2.8)
            "didn't", "hasn't", "haven't", "hadn't", "ain't", "mustn't", "needn't", "mightn't", "shan't", "oughtn't"]


@pytest.mark.parametrize("negator", NEGATORS)
def test_an_excerpt_never_cuts_a_negation_off(negator):
    quote = "retry failed jobs after a timeout."
    negated = f"In this release the scheduler {negator} retry failed jobs after a timeout."
    assert quote_verified(quote, QuoteIndex(negated)) is False
    later = f"{negated} The next release will retry failed jobs after a timeout."
    assert quote_verified(quote, QuoteIndex(later)) is True, "a later occurrence no negator precedes still matches"


def test_ellipsis_segments_must_be_in_order_and_near():
    assert quote_verified("an override is recorded in overrides_applied … Presets are the shipped defaults",
                          PROSE_IDX) is False, "segments out of order"
    head, tail = "the dispatcher accepts a job", "and the worker acknowledges it"
    near = QuoteIndex(f"{head} {_doc(2_000, seed=21)} {tail}")
    far = QuoteIndex(f"{head} {_doc(4_000, seed=21)} {tail}")
    assert quote_verified(f"{head} … {tail}", near) is True, "the next segment starts within 3 000 chars"
    assert quote_verified(f"{head} … {tail}", far) is False, "more than 3 000 chars apart"


def test_the_ellipsis_gap_boundary_is_exactly_3000_chars():
    """A segment's first character may sit at most 3 000 chars (of the normalised
    token text) after the previous segment's end. (At 1d69139, 3 001 passed.)"""
    head, tail = "alpha beta gamma", "omega sigma tau"
    for last, gap, want in (("abc", 3000, True), ("abcd", 3001, False)):
        idx = QuoteIndex(f"{head} {' '.join(['wxyz'] * 599)} {last} {tail}")
        end, start = idx.words.index(" gamma ") + len(" gamma"), idx.words.index(" omega ") + 1
        assert start - end == gap
        assert quote_verified(f"{head} … {tail}", idx) is want, gap


def test_a_quote_of_more_than_8_segments_never_verifies():
    idx = QuoteIndex(" ".join(f"w{i}" for i in range(20)))
    assert quote_verified(" … ".join(f"w{i}" for i in range(8)), idx) is True
    assert quote_verified(" … ".join(f"w{i}" for i in range(9)), idx) is False
    rep = QuoteIndex("a " * 30_000)                  # the third re-review's degenerate case: 2 s at 1d69139
    t0 = time.perf_counter()
    assert quote_verified("…".join(["a"] * 125), rep) is False
    assert time.perf_counter() - t0 <= 0.050


ADVERSARIAL = {
    "20 000 unclosed [": ("[" * 20_000 + " word" * 8_000)[:60_000],
    "30 000 unclosed ![": "![" * 30_000,
    "15 000 links without a closing paren": "[a](" * 15_000,
    "15 000 footnotes without a closing paren": "[1](" * 15_000,
}


@pytest.mark.parametrize("label", list(ADVERSARIAL))     # ids stay short: the node id lands in an env var
def test_indexing_stays_linear(label):
    """The link, image and footnote patterns never rescan the document from every
    bracket. (At 1d69139, 20 000 unclosed "[" took seconds to index.)"""
    doc = ADVERSARIAL[label]
    assert len(doc) == 60_000
    best = float("inf")
    for _ in range(3):                       # best of three, as for the verification bound
        t0 = time.perf_counter()
        QuoteIndex(doc)
        best = min(best, time.perf_counter() - t0)
    assert best <= 0.050, f"{label}: {best * 1000:.1f} ms to index"


@pytest.mark.parametrize("case", ["exact", "altered", "miss"])
def test_verification_bound(case):
    exact = _quote_at(31_000)
    quote = {"exact": exact, "altered": exact.replace(" ", " zzz ", 1), "miss": _miss()}[case]
    assert len(DOC) == 60_000 and len(quote) >= 240
    best = float("inf")
    for _ in range(3):                       # best of three: the bound is on the work, not on a noisy timer
        t0 = time.perf_counter()
        result = quote_verified(quote, QuoteIndex(DOC))
        best = min(best, time.perf_counter() - t0)
    assert result is (case == "exact")
    assert best <= 0.050, f"{case}: {best * 1000:.1f} ms for one 250-char quote incl. indexing the document"


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
    its limit, so a quote it shortened never ends in a word fragment."""
    idx = QuoteIndex(_doc(20_000, seed=11) + " " + LONG + " " + _doc(20_000, seed=13))
    edited = LONG.replace(",", "").replace(";", " -")              # a digest LLM's punctuation edits
    assert edited[249].isalpha() and edited[250].isalpha(), "a hard cut at 250 would split a word"
    cut = _digested_quote(edited)
    assert edited.startswith(cut) and len(cut) <= 250 and edited[len(cut)].isspace(), cut
    assert quote_verified(cut, idx) is True
    assert quote_verified(edited[:250], idx) is False, "the hard cut ends in a fragment"

    verbatim = _digested_quote(LONG)
    assert LONG.startswith(verbatim) and LONG[len(verbatim)].isspace()
    assert quote_verified(verbatim, idx) is True
    assert _digested_quote("x" * 300) == "x" * 250, "a single token longer than the limit keeps the hard cut"
