"""Probe engines providers with live calls (v4 WP-A, spec §3).

Per provider whose key is set: list its models (when it has a ``/models``
endpoint), then call one model through the driver directly — no facade
retries or fallbacks, so each check is exactly one request:

  plain       a 20-token completion that must return text
  json_mode   a completion with response_format=json_object
  temperature a temperature=0.2 call (sent even where the config says unsupported)
  cap16       max_tokens=16 on a long answer: is the cap honoured?
  json200     a 200-token JSON call (json_mode as the facade would pick it)
              that must return non-empty content
  routed_via  captured from any of the answers?

Prints a Markdown table and suggested flag changes for argus_engines.json.
Never prints keys. Exits 1 when a call failed, a check did not hold, or
nothing could be probed (no key set).

  PYTHONPATH="" PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe scripts/engines_probe.py --provider freellmapi
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Clean Python path so the argus venv isn't poisoned by a parent venv.
os.environ.pop("PYTHONPATH", None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import openai  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from argus.engines import registry  # noqa: E402
from argus.engines.llm_openai import OpenAICompatDriver  # noqa: E402
from argus.engines.registry import EngineConfigError  # noqa: E402
from argus.engines.schema import EngineSpec, EnginesConfig, ProviderSpec  # noqa: E402
from argus.llm import strip_reasoning  # noqa: E402

TIERS = ("cheap", "strong", "judge")
CAP = 16
PLAIN = [SystemMessage(content="Answer in one short sentence."),
         HumanMessage(content="What is the capital of France?")]
COUNT = [HumanMessage(content="Count from 1 to 200, separated by spaces.")]
JSON_ASK = [SystemMessage(content='Reply with only a JSON object of the form '
                                  '{"city": "...", "country": "..."}.'),
            HumanMessage(content="The capital of France.")]
REJECTED = (openai.BadRequestError, openai.UnprocessableEntityError)   # the request shape was refused

_driver = OpenAICompatDriver()


def pick_model(cfg: EnginesConfig, prov: ProviderSpec, listed: list[str] | None) -> str | None:
    """A model the config already names for this provider: a fixed tier
    model (cheapest tier first), else the first preference it serves."""
    for tier in TIERS:
        for p in cfg.profiles.values():
            spec = p.tiers[tier]
            if spec.provider == prov.name and spec.model:
                return spec.model
    for tier in TIERS:
        for m in prov.model_preferences.get(tier, []):
            if listed is None or m in listed:
                return m
    return None


def _parses(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


class Probe:
    def __init__(self, prov: ProviderSpec, model: str):
        self.prov, self.model = prov, model
        self.engine = EngineSpec(provider=prov.name, model=model)
        self.key = os.environ.get(prov.api_key_env) if prov.api_key_env else None
        self.rows: list[list[str]] = []
        self.suggestions: list[str] = []
        self.failed = False
        self.routed: set[str] = set()

    def redact(self, text: str) -> str:
        return text.replace(self.key, "***") if self.key else text

    def row(self, check: str, status: str, resp=None, ms: int | None = None, detail: str = "") -> None:
        self.failed |= status == "FAIL"
        tokens = (f"{resp.usage['completion_tokens']}/{resp.usage['reasoning_tokens']}" if resp else "")
        self.rows.append([self.prov.name, resp.served_model if resp else self.model, check, status,
                          resp.finish_reason if resp else "", tokens, "" if ms is None else str(ms),
                          detail.replace("|", "/").replace("\n", " ")[:110]])

    def call(self, messages, *, max_tokens: int, json_mode: bool = False, temperature: float | None = None,
             engine: EngineSpec | None = None):
        t0 = time.monotonic()
        try:
            resp = _driver.complete(engine or self.engine, self.prov, messages, temperature=temperature,
                                    max_tokens=max_tokens, json_mode=json_mode)
        except (openai.APIError, EngineConfigError) as e:
            return None, e, int((time.monotonic() - t0) * 1000)
        if resp.routed_via:
            self.routed.add(resp.routed_via)
        return resp, None, int((time.monotonic() - t0) * 1000)

    def error(self, check: str, err: Exception, ms: int, *, expected: bool = True) -> None:
        self.row(check, "FAIL" if expected else "info", ms=ms,
                 detail=self.redact(f"{type(err).__name__}: {err}"))

    def thinking_hint(self, check: str, resp) -> None:
        if not strip_reasoning(resp.text) and resp.finish_reason == "length":
            self.suggestions.append(
                f"{self.prov.name}/{self.model}: {check} came back empty at its budget "
                f"(finish_reason=length): give this model a smaller thinking budget through the "
                f"EngineSpec extra_body (the field is provider-specific; see the provider's docs)")

    def run(self) -> None:
        p = self.prov
        resp, err, ms = self.call(PLAIN, max_tokens=20)
        if err:
            self.error("plain", err, ms)
        else:
            text = strip_reasoning(resp.text)
            self.row("plain", "ok" if text else "FAIL", resp, ms, repr(text[:60]))
            self.thinking_hint("plain", resp)

        resp, err, ms = self.call(JSON_ASK, max_tokens=400, json_mode=True)
        if err:
            self.error("json_mode", err, ms, expected=p.supports_json_mode or not isinstance(err, REJECTED))
            if isinstance(err, REJECTED) and p.supports_json_mode:
                self.suggestions.append(f'{p.name}: response_format was refused: set "supports_json_mode": false')
        else:
            good = _parses(strip_reasoning(resp.text))
            self.row("json_mode", "ok" if good else ("FAIL" if p.supports_json_mode else "info"), resp, ms,
                     ("parses as a JSON object" if good else "not a JSON object: ")
                     + ("" if good else repr(resp.text[:60])))
            if good and not p.supports_json_mode:
                self.suggestions.append(f'{p.name}: response_format accepted on {self.model}: '
                                        f'"supports_json_mode": true is possible (config says false)')

        accepts = p.supports_temperature if self.engine.supports_temperature is None else self.engine.supports_temperature
        forced = self.engine.model_copy(update={"supports_temperature": True})
        resp, err, ms = self.call(PLAIN, max_tokens=20, temperature=0.2, engine=forced)
        if err:
            self.error("temperature", err, ms, expected=accepts or not isinstance(err, REJECTED))
            if isinstance(err, REJECTED) and accepts:
                self.suggestions.append(f'{p.name}/{self.model}: temperature was refused: '
                                        f'set "supports_temperature": false')
        else:
            self.row("temperature", "ok" if accepts else "info", resp, ms, "temperature=0.2 accepted")
            if not accepts:
                self.suggestions.append(f'{p.name}: temperature accepted on {self.model}: '
                                        f'"supports_temperature": true is possible (config says false)')

        resp, err, ms = self.call(COUNT, max_tokens=CAP)
        if err:
            self.error("cap16", err, ms)
        else:
            n = resp.usage["completion_tokens"]
            honoured = n <= CAP if n else resp.finish_reason == "length"   # no usage reported → trust finish_reason
            self.row("cap16", "ok" if honoured else "FAIL", resp, ms,
                     f"{p.token_param}={CAP}: " + ("honoured" if honoured else f"ignored ({n} tokens)"))
            if not honoured:
                other = "max_tokens" if p.token_param == "max_completion_tokens" else "max_completion_tokens"
                self.suggestions.append(f'{p.name}: the {p.token_param} cap was ignored: try "token_param": "{other}"')

        resp, err, ms = self.call(JSON_ASK, max_tokens=200, json_mode=p.supports_json_mode)
        if err:
            self.error("json200", err, ms)
        else:
            text = strip_reasoning(resp.text)
            self.row("json200", "ok" if text else "FAIL", resp, ms,
                     ("parses; " if _parses(text) else "not JSON; ") + repr(text[:60]) if text
                     else "empty content")
            self.thinking_hint("json200", resp)

        self.row("routed_via", "ok" if self.routed else "info",
                 detail=", ".join(sorted(self.routed)) or "not captured (no x-routed-via header or body field)")


def probe_provider(cfg: EnginesConfig, prov: ProviderSpec, model: str | None) -> Probe:
    listed: list[str] | None = None
    listing_row: tuple[str, str] | None = None
    if prov.models_endpoint:
        try:
            listed = registry.live_models(prov.name)
            listing_row = ("ok", f"{len(listed)} listed")
        except EngineConfigError as e:
            listing_row = ("FAIL", str(e))
    chosen = model or pick_model(cfg, prov, listed)
    pr = Probe(prov, chosen or "-")
    if listing_row:
        status, detail = listing_row
        if chosen and listed is not None and chosen not in listed:
            status, detail = "FAIL", f"{detail}; {chosen} is not among them"
        pr.row("models", status, detail=pr.redact(detail))
    if chosen is None:
        pr.row("model", "FAIL", detail="the config names no model for this provider: "
                                        "pass --provider NAME --model ID")
        return pr
    pr.run()
    return pr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--provider", help="probe this provider (keyless local servers included)")
    which.add_argument("--all", action="store_true", help="probe every provider whose key is set")
    ap.add_argument("--model", help="model id to call (with --provider; default: one the config names)")
    args = ap.parse_args()
    if args.model and not args.provider:
        ap.error("--model needs --provider")

    cfg = asyncio.run(registry.load_config())
    if args.provider:
        provs = [registry.provider(args.provider)]
    else:
        provs = list(cfg.providers.values())

    probes, skipped = [], []
    for prov in provs:
        if args.all and prov.api_key_env is None:
            skipped.append(f"{prov.name}: keyless local server (probe it with --provider {prov.name})")
            continue
        if prov.api_key_env and not os.environ.get(prov.api_key_env):
            skipped.append(f"{prov.name}: {prov.api_key_env} is not set")
            continue
        probes.append(probe_provider(cfg, prov, args.model))

    if not probes:
        for s in skipped:
            print(f"skipped — {s}")
        print("Nothing was probed: no provider has its key set. Set a key in .env, "
              "or probe a local server with --provider NAME.")
        return 1

    print("| provider | model | check | status | finish | tokens (completion/reasoning) | ms | detail |")
    print("|---|---|---|---|---|---|---|---|")
    for pr in probes:
        for r in pr.rows:
            print("| " + " | ".join(r) + " |")
    for s in skipped:
        print(f"\nskipped — {s}")
    suggestions = [s for pr in probes for s in pr.suggestions]
    print("\nSuggested flag changes:" if suggestions else "\nSuggested flag changes: none")
    for s in suggestions:
        print(f"- {s}")
    return 1 if any(pr.failed for pr in probes) else 0


if __name__ == "__main__":
    sys.exit(main())
