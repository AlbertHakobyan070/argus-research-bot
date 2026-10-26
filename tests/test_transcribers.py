"""Transcriber engines (v4 WP-F): the block is read and checked at use time,
the ``openai_audio`` request shape and chunk concatenation (a fake httpx
transport), the error surfaces, the ``faster_whisper`` mapping, and
``media.extract_audio_chunks`` on a generated clip."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from argus import media
from argus.engines import registry, transcribers
from argus.engines.registry import EngineConfigError
from argus.engines.schema import EnginesConfig, TranscriberConfig
from argus.engines.transcribers import TranscribeError

FW = {"model": "small", "device": "cpu", "compute_type": "int8"}
OA = {"provider": "aud", "model": "whisper-x", "chunk_seconds": 600, "language": None}


def _cfg(**transcriber) -> EnginesConfig:
    tiers = {t: {"provider": "aud", "model": "m"} for t in ("cheap", "strong", "judge")}
    return EnginesConfig.model_validate({
        "active_profile": "p",
        "providers": {"aud": {"name": "aud", "base_url": "http://aud.test/v1", "api_key_env": "T_AUD_KEY",
                              "headers": {"X-App": "argus"}, "timeout_s": 12.0},
                      "local": {"name": "local", "base_url": "http://127.0.0.1:9/v1"}},
        "profiles": {"p": {"name": "p", "tiers": tiers}},
        "transcriber": transcriber})


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("T_AUD_KEY", "sk-aud-secret")


# ---------------------------------------------------------------------------
# selection and the block, checked at use time
# ---------------------------------------------------------------------------

def test_default_config_carries_no_block_and_needs_none_at_startup():
    cfg = _cfg()                          # a config without the block loads (startup never checks it)
    assert cfg.transcriber.asr_engine == "faster_whisper" and cfg.transcriber.faster_whisper == {}
    with pytest.raises(EngineConfigError, match=r"transcriber\.faster_whisper.*model"):
        transcribers.asr_engine(cfg.transcriber, cfg)


def test_selection_follows_asr_engine(monkeypatch):
    monkeypatch.setattr(transcribers, "asr_available", lambda: True)
    cfg = _cfg(faster_whisper=FW, openai_audio=OA)
    fw = transcribers.asr_engine(cfg.transcriber, cfg)
    assert isinstance(fw, transcribers.FasterWhisper) and fw.backend == "faster_whisper:small"
    t = cfg.transcriber.model_copy(update={"asr_engine": "openai_audio"})
    oa = transcribers.asr_engine(t, cfg)
    assert isinstance(oa, transcribers.OpenAIAudio) and oa.backend == "openai_audio:aud/whisper-x"


def test_openai_audio_without_its_key_names_the_env_var(monkeypatch):
    monkeypatch.delenv("T_AUD_KEY")
    cfg = _cfg(asr_engine="openai_audio", openai_audio=OA)
    with pytest.raises(EngineConfigError, match="T_AUD_KEY is not set"):
        transcribers.asr_engine(cfg.transcriber, cfg)


@pytest.mark.parametrize("block, needle", [
    ({k: v for k, v in OA.items() if k != "chunk_seconds"}, "chunk_seconds"),
    ({**OA, "provider": "nope"}, "provider 'nope'"),
    ({**OA, "chunk_seconds": 0}, "chunk_seconds"),
])
def test_openai_audio_block_errors_are_explicit(block, needle):
    cfg = _cfg(asr_engine="openai_audio", openai_audio=block)
    with pytest.raises(EngineConfigError, match=needle):
        transcribers.asr_engine(cfg.transcriber, cfg)


def test_faster_whisper_missing_package_is_a_config_error(monkeypatch):
    monkeypatch.setattr(transcribers, "asr_available", lambda: False)
    cfg = _cfg(faster_whisper=FW)
    with pytest.raises(EngineConfigError, match="faster-whisper"):
        transcribers.asr_engine(cfg.transcriber, cfg)


# ---------------------------------------------------------------------------
# openai_audio: request shape, chunking, concatenation, errors
# ---------------------------------------------------------------------------

def _fake_chunks(monkeypatch, sizes: list[int]):
    calls = {}

    async def fake_extract(src, out_dir, *, chunk_seconds, ffmpeg=None, **kw):
        calls.update(src=src, chunk_seconds=chunk_seconds)
        paths = []
        for i, n in enumerate(sizes):
            p = Path(out_dir) / f"chunk_{i:03d}.mp3"
            p.write_bytes(bytes([65 + i]) * n)
            paths.append(p)
        return paths
    monkeypatch.setattr(transcribers, "extract_audio_chunks", fake_extract)
    return calls


def _transport(monkeypatch, handler):
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        request.read()
        seen.append(request)
        return handler(request, len(seen) - 1)

    def client(timeout_s):
        TIMEOUTS.append(timeout_s)
        return httpx.AsyncClient(transport=httpx.MockTransport(wrapped), timeout=timeout_s)
    TIMEOUTS.clear()
    monkeypatch.setattr(transcribers, "_client", client)
    return seen


TIMEOUTS: list[float] = []      # the timeout_s each _client call got, per _transport


def _engine(**block) -> transcribers.OpenAIAudio:
    cfg = _cfg(asr_engine="openai_audio", openai_audio={**OA, **block})
    return transcribers.asr_engine(cfg.transcriber, cfg)


async def test_openai_audio_request_shape_and_chunks_are_concatenated(tmp_path, monkeypatch):
    calls = _fake_chunks(monkeypatch, [10, 20, 30])
    seen = _transport(monkeypatch, lambda req, i: httpx.Response(200, json={"text": f" part {i} "}))
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"v")
    text, language = await _engine(language="en").transcribe(src)

    assert text == "part 0\npart 1\npart 2", "chunks are transcribed in order and joined"
    assert language == "en"
    assert calls == {"src": src, "chunk_seconds": 600}
    assert len(seen) == 3
    req = seen[0]
    assert req.method == "POST" and str(req.url) == "http://aud.test/v1/audio/transcriptions"
    assert req.headers["authorization"] == "Bearer sk-aud-secret"
    assert req.headers["x-app"] == "argus"
    assert req.headers["content-type"].startswith("multipart/form-data")
    body = req.content
    for field, value in (("model", b"whisper-x"), ("response_format", b"json"), ("language", b"en")):
        assert f'name="{field}"'.encode() in body and value in body
    assert b'name="file"; filename="chunk_000.mp3"' in body and b"A" * 10 in body
    assert b"B" * 20 in seen[1].content and b"C" * 30 in seen[2].content
    assert TIMEOUTS == [12.0], "the provider's timeout_s, one client for every chunk"


def test_client_carries_the_timeout_into_httpx():
    assert transcribers._client(12.0).timeout == httpx.Timeout(12.0)


async def test_no_key_or_authorization_header_in_any_log(tmp_path, monkeypatch, caplog):
    """A failing and a succeeding transcription, through transcribe_url, with
    every logger at DEBUG (httpx and httpcore included)."""
    import logging
    from argus import transcribe as tr_mod
    from argus.media import MediaResult
    caplog.set_level(logging.DEBUG)
    media_file = tmp_path / "m" / "x" / "clip.mp4"

    async def fake_download(url, *, dest_root, quality="auto", **kw):
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(b"v")
        return MediaResult(ok=True, url=url, platform="x", path=str(media_file), size_bytes=1)
    monkeypatch.setattr(tr_mod, "download_media", fake_download)
    _fake_chunks(monkeypatch, [5, 5])
    cfg = _cfg(asr_engine="openai_audio", openai_audio=OA)
    for status in (401, 200):
        _transport(monkeypatch, lambda req, i, st=status: httpx.Response(
            st, json={"text": "words"} if st == 200 else {"error": "bad key sk-aud-secret"}))
        r = await tr_mod.transcribe_url("https://x.com/u/status/9", transcripts_root=tmp_path / "t",
                                        media_root=tmp_path / "m", engines=cfg)
        assert r.ok is (status == 200)
    assert caplog.records, "the run logged something, so the check is not vacuous"
    logged = caplog.text + " ".join(str(r.args) for r in caplog.records)
    assert "sk-aud-secret" not in logged
    import re
    assert not re.search(r"authorization['\"]?\s*[:=]", logged, re.I), "no header name/value pair"
    assert "bearer " not in logged.lower()


async def test_transport_error_keeps_no_context_holding_the_key(tmp_path, monkeypatch):
    _fake_chunks(monkeypatch, [5])

    def boom(req, i):
        raise httpx.ConnectError("refused sk-aud-secret")
    _transport(monkeypatch, boom)
    with pytest.raises(TranscribeError) as ei:
        await _engine().transcribe(tmp_path / "clip.mp4")
    assert "sk-aud-secret" not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__context__ is None


async def test_openai_audio_omits_a_null_language(tmp_path, monkeypatch):
    _fake_chunks(monkeypatch, [5])
    seen = _transport(monkeypatch, lambda req, i: httpx.Response(200, json={"text": "hi"}))
    text, language = await _engine().transcribe(tmp_path / "clip.mp4")
    assert (text, language) == ("hi", "")
    assert b'name="language"' not in seen[0].content


async def test_openai_audio_http_error_raises_without_the_key(tmp_path, monkeypatch):
    _fake_chunks(monkeypatch, [5, 5])
    _transport(monkeypatch, lambda req, i: (httpx.Response(200, json={"text": "ok"}) if i == 0 else
                                            httpx.Response(429, text="rate limited sk-aud-secret")))
    with pytest.raises(TranscribeError) as ei:
        await _engine().transcribe(tmp_path / "clip.mp4")
    msg = str(ei.value)
    assert "429" in msg and "chunk 2/2" in msg and "rate limited" in msg
    assert "sk-aud-secret" not in msg and "***" in msg


async def test_openai_audio_oversized_chunk_is_refused_before_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribers, "MAX_UPLOAD_BYTES", 15)
    _fake_chunks(monkeypatch, [10, 20])
    seen = _transport(monkeypatch, lambda req, i: httpx.Response(200, json={"text": "ok"}))
    with pytest.raises(TranscribeError, match=r"chunk 2/2 .*20 bytes.*chunk_seconds"):
        await _engine().transcribe(tmp_path / "clip.mp4")
    assert len(seen) == 0, "nothing is uploaded once a chunk is known to be too large"


async def test_openai_audio_413_is_a_size_error(tmp_path, monkeypatch):
    _fake_chunks(monkeypatch, [5])
    _transport(monkeypatch, lambda req, i: httpx.Response(413, text="Request Entity Too Large"))
    with pytest.raises(TranscribeError, match="413.*chunk_seconds"):
        await _engine().transcribe(tmp_path / "clip.mp4")


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>not json</html>"),
    httpx.Response(200, json={"error": {"message": "bad"}}),
    httpx.Response(200, json={"text": ["list"]}),
])
async def test_openai_audio_unusable_answer_raises(tmp_path, monkeypatch, response):
    _fake_chunks(monkeypatch, [5])
    _transport(monkeypatch, lambda req, i: response)
    with pytest.raises(TranscribeError, match="chunk 1/1"):
        await _engine().transcribe(tmp_path / "clip.mp4")


async def test_openai_audio_connection_error_raises(tmp_path, monkeypatch):
    _fake_chunks(monkeypatch, [5])

    def boom(req, i):
        raise httpx.ConnectError("refused")
    _transport(monkeypatch, boom)
    with pytest.raises(TranscribeError, match="ConnectError"):
        await _engine().transcribe(tmp_path / "clip.mp4")


async def test_openai_audio_all_empty_is_no_silent_empty_transcript(tmp_path, monkeypatch):
    _fake_chunks(monkeypatch, [5, 5])
    _transport(monkeypatch, lambda req, i: httpx.Response(200, json={"text": "  "}))
    with pytest.raises(TranscribeError, match="no speech"):
        await _engine().transcribe(tmp_path / "clip.mp4")


async def test_openai_audio_extraction_failure_is_a_transcribe_error(tmp_path, monkeypatch):
    async def fail(*a, **k):
        raise RuntimeError("ffmpeg failed (1): Invalid data")
    monkeypatch.setattr(transcribers, "extract_audio_chunks", fail)
    with pytest.raises(TranscribeError, match="audio extraction failed.*Invalid data"):
        await _engine().transcribe(tmp_path / "clip.mp4")


# ---------------------------------------------------------------------------
# faster_whisper: the block maps onto WhisperModel
# ---------------------------------------------------------------------------

async def test_faster_whisper_maps_the_block_onto_whisper_model(tmp_path, monkeypatch):
    import sys
    import types
    made: list[tuple] = []

    class FakeModel:
        def __init__(self, model, *, device, compute_type):
            made.append((model, device, compute_type))

        def transcribe(self, path, **kw):
            assert kw == {"vad_filter": True}
            segs = [type("S", (), {"text": t}) for t in (" hello ", "", "world")]
            return iter(segs), type("I", (), {"language": "en"})

    # A stand-in module, so the test needs no optional `asr` extra installed.
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(transcribers, "_ASR_MODEL", None)
    monkeypatch.setattr(transcribers, "asr_available", lambda: True)
    cfg = _cfg(faster_whisper={"model": "tiny", "device": "cuda", "compute_type": "float16"})
    eng = transcribers.asr_engine(cfg.transcriber, cfg)
    assert await eng.transcribe(tmp_path / "a.mp4") == ("hello\nworld", "en")
    await eng.transcribe(tmp_path / "b.mp4")
    assert made == [("tiny", "cuda", "float16")], "one model per (model, device, compute_type)"
    cfg2 = _cfg(faster_whisper=FW)
    await transcribers.asr_engine(cfg2.transcriber, cfg2).transcribe(tmp_path / "c.mp4")
    assert made[-1] == ("small", "cpu", "int8"), "a changed block loads its own model"


async def test_faster_whisper_all_silent_is_no_silent_empty_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribers, "asr_available", lambda: True)
    monkeypatch.setattr(transcribers, "_whisper_file", lambda path, key, language, timestamps: ("", ""))
    cfg = _cfg(faster_whisper=FW)
    with pytest.raises(TranscribeError, match="no speech"):
        await transcribers.asr_engine(cfg.transcriber, cfg).transcribe(tmp_path / "a.mp4")


# ---------------------------------------------------------------------------
# media.extract_audio_chunks on a generated clip (real ffmpeg, local only)
# ---------------------------------------------------------------------------

def _ffmpeg() -> str:
    ffmpeg = media.resolve_ffmpeg()
    if ffmpeg is None:
        pytest.skip("no ffmpeg")
    return ffmpeg


def _decoded_seconds(ffmpeg: str, p: Path) -> float:
    """What a server decoding this file on its own gets: 16 kHz mono samples / rate."""
    pcm = subprocess.run([ffmpeg, "-v", "error", "-i", str(p), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return len(pcm) / 2 / 16000


async def test_extract_audio_chunks_are_contiguous_with_no_lost_audio(tmp_path):
    """Review Minor 1: each chunk decoded on its own, the durations sum to the
    source's (the old segment muxer lost about 0.1 s per boundary)."""
    ffmpeg = _ffmpeg()
    d = tmp_path / "dir with space é"
    d.mkdir()
    src = d / "tone ü.wav"
    subprocess.run([ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=7.3", str(src)], check=True)
    out = d / "chunks"
    chunks = await media.extract_audio_chunks(src, out, chunk_seconds=2, ffmpeg=ffmpeg)
    assert [p.name for p in chunks] == ["chunk_000.mp3", "chunk_001.mp3", "chunk_002.mp3", "chunk_003.mp3"]
    assert all(p.parent == out for p in chunks)
    per = [_decoded_seconds(ffmpeg, c) for c in chunks]
    assert per[:3] == pytest.approx([2.0, 2.0, 2.0], abs=0.01)
    assert sum(per) == pytest.approx(_decoded_seconds(ffmpeg, src), abs=0.02)
    assert not list(out.glob("*.pcm")), "the intermediate PCM is removed"


async def test_extract_audio_chunks_errors_are_explicit(tmp_path, monkeypatch):
    real = media.resolve_ffmpeg()
    monkeypatch.setattr(media, "resolve_ffmpeg", lambda: None)
    with pytest.raises(RuntimeError, match="needs ffmpeg"):
        await media.extract_audio_chunks(tmp_path / "x.mp4", tmp_path / "o", chunk_seconds=10)
    if real is None:
        pytest.skip("no ffmpeg")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not media")
    with pytest.raises(RuntimeError, match=r"ffmpeg failed"):
        await media.extract_audio_chunks(bad, tmp_path / "o2", chunk_seconds=10, ffmpeg=real)


def _slow_children(monkeypatch) -> list:
    """Every subprocess the media module starts becomes a 60 s sleep; the
    started processes are returned for inspection."""
    import asyncio
    import sys
    real = asyncio.create_subprocess_exec
    procs: list = []

    async def slow(*cmd, **kw):
        proc = await real(sys.executable, "-c", "import time; time.sleep(60)", **kw)
        procs.append(proc)
        return proc
    monkeypatch.setattr(media.asyncio, "create_subprocess_exec", slow)
    return procs


async def test_cancelled_extraction_kills_and_awaits_ffmpeg(tmp_path, monkeypatch):
    """Review Minor 2: a cancel leaves no ffmpeg running, and the chunk
    folder can be removed at once (Windows holds open files)."""
    import asyncio
    import shutil
    procs = _slow_children(monkeypatch)
    out = tmp_path / "o"
    task = asyncio.create_task(media.extract_audio_chunks(tmp_path / "x.mp4", out, chunk_seconds=2, ffmpeg="ff"))
    for _ in range(100):
        if procs:
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert procs and procs[0].returncode is not None, "the child was killed and awaited"
    shutil.rmtree(out)


async def test_extraction_timeout_kills_ffmpeg_and_says_so(tmp_path, monkeypatch):
    procs = _slow_children(monkeypatch)
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        await media.extract_audio_chunks(tmp_path / "x.mp4", tmp_path / "o", chunk_seconds=2,
                                         ffmpeg="ff", timeout_s=1)
    assert procs[0].returncode is not None


async def test_cancelled_openai_audio_leaves_no_temp_folder(tmp_path, monkeypatch):
    import asyncio
    import tempfile
    procs = _slow_children(monkeypatch)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    task = asyncio.create_task(_engine().transcribe(tmp_path / "clip.mp4"))
    for _ in range(100):
        if procs:
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):    # not a PermissionError from the cleanup
        await task
    assert procs[0].returncode is not None
    assert not list(tmp_path.glob("argus_audio_*"))


def test_asr_available_does_not_import_faster_whisper(tmp_path):
    """Review Minor 3: the check runs on the event loop, so it must not pay
    the package import. An empty stand-in package on PYTHONPATH makes it
    findable whether or not the optional `asr` extra is installed."""
    import sys
    (tmp_path / "faster_whisper").mkdir()
    (tmp_path / "faster_whisper" / "__init__.py").write_text("", encoding="utf-8")
    code = ("import sys; from argus.engines import transcribers; "
            "print(transcribers.asr_available(), 'faster_whisper' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "PYTHONPATH": str(tmp_path)}).stdout.split()
    assert out == ["True", "False"]


# ---------------------------------------------------------------------------
# persistence: engines.transcriber round-trips through the library
# ---------------------------------------------------------------------------

def _write_cfg(p: Path, **transcriber) -> None:
    p.write_text(json.dumps(_cfg(**{"faster_whisper": FW, "openai_audio": OA, **transcriber}).model_dump()),
                 encoding="utf-8")


@pytest.fixture
async def plib(tmp_path, monkeypatch):
    from argus.library import Library
    monkeypatch.setattr(registry, "_state", registry._State())
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    yield lib
    await lib.close()


async def _reload(p: Path, lib, monkeypatch) -> EnginesConfig:
    monkeypatch.setattr(registry, "_state", registry._State())
    return await registry.load_config([str(p)], library=lib)


async def test_transcriber_persists_and_reloads(tmp_path, plib, monkeypatch):
    p = tmp_path / "e.json"
    _write_cfg(p)
    await registry.load_config([str(p)], library=plib)
    t = registry.current_config().transcriber
    await registry.set_transcriber(t.model_copy(update={
        "asr_engine": "openai_audio", "openai_audio": {**OA, "model": "whisper-y"}}))
    assert json.loads(await plib.get_setting("engines.transcriber")) == {
        "asr_engine": "openai_audio", "openai_audio": {"model": "whisper-y"}}, "only what differs from the file"
    cfg = await _reload(p, plib, monkeypatch)
    assert cfg.transcriber.asr_engine == "openai_audio"
    assert cfg.transcriber.openai_audio == {**OA, "model": "whisper-y"}
    assert cfg.transcriber.faster_whisper == FW
    assert transcribers.asr_engine(cfg.transcriber, cfg).backend == "openai_audio:aud/whisper-y"


async def test_a_stored_choice_never_shadows_later_file_edits(tmp_path, plib, monkeypatch):
    """Review Important 1 (the reviewer's probe): after a stored choice, a
    file edit of any key the choice did not change takes effect."""
    p = tmp_path / "e.json"
    _write_cfg(p)
    await registry.load_config([str(p)], library=plib)
    t = registry.current_config().transcriber
    await registry.set_transcriber(t.model_copy(update={"asr_engine": "faster_whisper"}))  # /engines transcriber faster_whisper
    _write_cfg(p, faster_whisper={**FW, "model": "tiny"})
    cfg = await _reload(p, plib, monkeypatch)
    assert cfg.transcriber.faster_whisper["model"] == "tiny"

    await registry.set_transcriber(cfg.transcriber.model_copy(update={
        "asr_engine": "openai_audio", "openai_audio": {**OA, "provider": "local"}}))
    _write_cfg(p, faster_whisper={**FW, "model": "tiny"}, openai_audio={**OA, "model": "whisper-fixed",
                                                                         "chunk_seconds": 300})
    cfg = await _reload(p, plib, monkeypatch)
    assert cfg.transcriber.asr_engine == "openai_audio"
    assert cfg.transcriber.openai_audio == {"provider": "local", "model": "whisper-fixed",
                                            "chunk_seconds": 300, "language": None}, (
        "the stored provider stays, the file's corrected model and chunk_seconds arrive")


async def test_setting_a_value_back_to_the_file_drops_it_from_the_library(tmp_path, plib, monkeypatch):
    p = tmp_path / "e.json"
    _write_cfg(p)
    await registry.load_config([str(p)], library=plib)
    t = registry.current_config().transcriber
    await registry.set_transcriber(t.model_copy(update={"asr_engine": "openai_audio"}))
    await registry.set_transcriber(t)
    assert json.loads(await plib.get_setting("engines.transcriber")) == {}


async def test_a_pre_fix_whole_block_row_still_loads(tmp_path, plib, monkeypatch):
    """A row written before this fix holds the whole block. It loads; only
    the fields a choice could set (asr_engine, and openai_audio's provider
    and model where they differ from the file) are kept, so the copied rest
    follows the file again."""
    p = tmp_path / "e.json"
    old_row = {"youtube": ["captions", "asr"], "asr_engine": "openai_audio",
               "faster_whisper": FW, "openai_audio": {**OA, "model": "whisper-mine"}}
    await plib.set_setting("engines.transcriber", json.dumps(old_row))
    _write_cfg(p, youtube=["asr"], faster_whisper={**FW, "model": "tiny"},
               openai_audio={**OA, "chunk_seconds": 300})
    cfg = await _reload(p, plib, monkeypatch)
    t = cfg.transcriber
    assert t.asr_engine == "openai_audio" and t.openai_audio["model"] == "whisper-mine"
    assert t.youtube == ["asr"] and t.faster_whisper["model"] == "tiny"
    assert t.openai_audio["chunk_seconds"] == 300 and t.openai_audio["provider"] == "aud"


async def test_removed_whisper_knob_warns_at_startup(monkeypatch, tmp_path, caplog):
    """Review Important 2: ARGUS_WHISPER_MODEL is no longer read; startup
    names it and where the setting lives now."""
    import logging
    from test_engines_cmd import CFG, _post_init
    monkeypatch.setattr(registry, "_state", registry._State())
    monkeypatch.setattr(registry, "_get_json", lambda *a, **k: pytest.fail("network I/O attempted"))
    caplog.set_level(logging.WARNING, logger="argus.bot")
    monkeypatch.setenv("ARGUS_WHISPER_MODEL", "tiny")
    await _post_init(monkeypatch, tmp_path, json.dumps(CFG))
    hits = [r.getMessage() for r in caplog.records if "ARGUS_WHISPER_MODEL" in r.getMessage()]
    assert len(hits) == 1
    assert "transcriber.faster_whisper.model" in hits[0] and "/engines transcriber" in hits[0]


def test_shipped_block_matches_the_spec():
    shipped = json.loads((registry._ROOT / "argus_engines.json").read_text(encoding="utf-8"))
    t = TranscriberConfig.model_validate(shipped["transcriber"])
    assert t.youtube == ["captions", "asr"] and t.asr_engine == "faster_whisper"
    assert t.faster_whisper == {"model": "small", "device": "cpu", "compute_type": "int8"}
    assert t.openai_audio == {"provider": "groq", "model": "whisper-large-v3-turbo",
                              "chunk_seconds": 600, "language": None}
    assert t.openai_audio["provider"] in shipped["providers"]


# ---------------------------------------------------------------------------
# /engines transcriber <faster_whisper|openai_audio> [provider/model]
# ---------------------------------------------------------------------------

@pytest.fixture
async def cmd_lib(tmp_path, monkeypatch):
    from argus.bot import engines_cmd
    from argus.library import Library
    monkeypatch.setattr(registry, "_state", registry._State())
    monkeypatch.setattr(engines_cmd, "_allowed", lambda _s, _uid: True)
    monkeypatch.setattr(engines_cmd, "get_settings", lambda: None)
    p = tmp_path / "e.json"
    p.write_text(json.dumps(_cfg(faster_whisper=FW, openai_audio=OA).model_dump()), encoding="utf-8")
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    await registry.load_config([str(p)], library=lib)
    yield lib
    await lib.close()


async def _say(lib, *args) -> str:
    from unittest.mock import AsyncMock, MagicMock
    from argus.bot import engines_cmd
    u = MagicMock()
    u.message.reply_text = AsyncMock()
    ctx = MagicMock()
    ctx.args = list(args)
    ctx.application.bot_data = {"library": lib}
    await engines_cmd.engines_cmd(u, ctx)
    return u.message.reply_text.await_args.args[0]


async def test_engines_transcriber_switches_and_persists(cmd_lib):
    text = await _say(cmd_lib, "transcriber", "openai_audio", "aud/whisper-z")
    assert "openai_audio:aud/whisper-z" in text and "⚠" not in text
    stored = json.loads(await cmd_lib.get_setting("engines.transcriber"))
    assert stored == {"asr_engine": "openai_audio", "openai_audio": {"model": "whisper-z"}}, (
        "only what the command changed; the provider equals the file's, the rest stays the file's")
    assert registry.current_config().transcriber.openai_audio == {**OA, "model": "whisper-z"}

    text = await _say(cmd_lib, "transcriber", "faster_whisper")
    assert "faster_whisper:small" in text
    assert registry.current_config().transcriber.asr_engine == "faster_whisper"
    assert json.loads(await cmd_lib.get_setting("engines.transcriber")) == {
        "openai_audio": {"model": "whisper-z"}}, "back to the file's engine: nothing stored for it"


async def test_engines_transcriber_shows_and_warns(cmd_lib, monkeypatch):
    text = await _say(cmd_lib, "transcriber")
    assert "faster_whisper:small" in text and "captions → asr" in text
    monkeypatch.delenv("T_AUD_KEY")
    text = await _say(cmd_lib, "transcriber", "openai_audio")
    assert "openai_audio:aud/whisper-x" in text and "T_AUD_KEY is not set" in text, (
        "saved, with the missing key named: it fails only when a transcription runs")


@pytest.mark.parametrize("args, needle", [
    (("transcriber", "mlx"), "faster_whisper or openai_audio"),
    (("transcriber", "openai_audio", "nope/m"), "provider 'nope'"),
    (("transcriber", "openai_audio", "aud"), "provider/model"),
    (("transcriber", "faster_whisper", "aud/m"), "faster_whisper takes no provider/model"),
])
async def test_engines_transcriber_refusals_persist_nothing(cmd_lib, args, needle):
    text = await _say(cmd_lib, *args)
    assert text.startswith("⚠️") and needle in text
    assert await cmd_lib.get_setting("engines.transcriber") is None
