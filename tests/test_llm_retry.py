"""Retry logic for LLM calls — Phase 6b. Hermetic (no network)."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from argus import llm


class _Chat:
    def __init__(self, fail_times: int, exc: BaseException):
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return SimpleNamespace(content="ok")


def test_retries_transient_then_succeeds(monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *_: None)
    chat = _Chat(fail_times=2, exc=httpx.ConnectError("boom"))
    resp = llm.invoke_with_retry(chat, ["m"], attempts=3)
    assert resp.content == "ok"
    assert chat.calls == 3, "should retry twice then succeed on the third"


def test_gives_up_after_attempts(monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *_: None)
    chat = _Chat(fail_times=99, exc=httpx.ConnectTimeout("nope"))
    with pytest.raises(httpx.ConnectTimeout):
        llm.invoke_with_retry(chat, ["m"], attempts=3)
    assert chat.calls == 3, "exactly `attempts` tries, no more"


def test_does_not_retry_non_transient(monkeypatch):
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *_: None)
    chat = _Chat(fail_times=99, exc=ValueError("bad request / parse"))
    with pytest.raises(ValueError):
        llm.invoke_with_retry(chat, ["m"], attempts=3)
    assert chat.calls == 1, "a non-transient error must fail fast (no retry)"


async def test_ainvoke_with_retry_transient(monkeypatch):
    import asyncio

    class _AChat:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("boom")
            return SimpleNamespace(content="async ok")

    chat = _AChat()
    resp = await llm.ainvoke_with_retry(chat, ["m"], attempts=3)
    assert resp.content == "async ok"
    assert chat.calls == 2
