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

    def fake_extract(src, out_dir, *, chunk_seconds, ffmpeg=None, **kw):
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
    monkeypatch.setattr(transcribers, "_client", lambda timeout_s: httpx.AsyncClient(
        transport=httpx.MockTransport(wrapped), timeout=timeout_s))
    return seen


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
    def fail(*a, **k):
        raise RuntimeError("ffmpeg failed (1): Invalid data")
    monkeypatch.setattr(transcribers, "extract_audio_chunks", fail)
    with pytest.raises(TranscribeError, match="audio extraction failed.*Invalid data"):
        await _engine().transcribe(tmp_path / "clip.mp4")


# ---------------------------------------------------------------------------
# faster_whisper: the block maps onto WhisperModel
# ---------------------------------------------------------------------------

async def test_faster_whisper_maps_the_block_onto_whisper_model(tmp_path, monkeypatch):
    import faster_whisper
    made: list[tuple] = []

    class FakeModel:
        def __init__(self, model, *, device, compute_type):
            made.append((model, device, compute_type))

        def transcribe(self, path, **kw):
            assert kw == {"vad_filter": True}
            segs = [type("S", (), {"text": t}) for t in (" hello ", "", "world")]
            return iter(segs), type("I", (), {"language": "en"})

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
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
    monkeypatch.setattr(transcribers, "_whisper_file", lambda path, key: ("", ""))
    cfg = _cfg(faster_whisper=FW)
    with pytest.raises(TranscribeError, match="no speech"):
        await transcribers.asr_engine(cfg.transcriber, cfg).transcribe(tmp_path / "a.mp4")


# ---------------------------------------------------------------------------
# media.extract_audio_chunks on a generated clip (real ffmpeg, local only)
# ---------------------------------------------------------------------------

def test_extract_audio_chunks_splits_a_clip(tmp_path):
    ffmpeg = media.resolve_ffmpeg()
    if ffmpeg is None:
        pytest.skip("no ffmpeg")
    src = tmp_path / "tone.wav"
    subprocess.run([ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=5", str(src)], check=True)
    out = tmp_path / "chunks"
    chunks = media.extract_audio_chunks(src, out, chunk_seconds=2, ffmpeg=ffmpeg)
    assert [p.name for p in chunks] == ["chunk_000.mp3", "chunk_001.mp3", "chunk_002.mp3"]
    assert all(p.stat().st_size > 0 and p.parent == out for p in chunks)


def test_extract_audio_chunks_errors_are_explicit(tmp_path, monkeypatch):
    real = media.resolve_ffmpeg()
    monkeypatch.setattr(media, "resolve_ffmpeg", lambda: None)
    with pytest.raises(RuntimeError, match="needs ffmpeg"):
        media.extract_audio_chunks(tmp_path / "x.mp4", tmp_path / "o", chunk_seconds=10)
    if real is None:
        pytest.skip("no ffmpeg")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not media")
    with pytest.raises(RuntimeError, match=r"ffmpeg failed"):
        media.extract_audio_chunks(bad, tmp_path / "o2", chunk_seconds=10, ffmpeg=real)


# ---------------------------------------------------------------------------
# persistence: engines.transcriber round-trips through the library
# ---------------------------------------------------------------------------

async def test_transcriber_persists_and_reloads(tmp_path, monkeypatch):
    from argus.library import Library
    monkeypatch.setattr(registry, "_state", registry._State())
    p = tmp_path / "e.json"
    p.write_text(json.dumps(_cfg(faster_whisper=FW, openai_audio=OA).model_dump()), encoding="utf-8")
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    try:
        await registry.load_config([str(p)], library=lib)
        t = registry.current_config().transcriber
        await registry.set_transcriber(t.model_copy(update={
            "asr_engine": "openai_audio", "openai_audio": {**OA, "model": "whisper-y"}}))
        monkeypatch.setattr(registry, "_state", registry._State())
        cfg = await registry.load_config([str(p)], library=lib)
    finally:
        await lib.close()
    assert cfg.transcriber.asr_engine == "openai_audio"
    assert cfg.transcriber.openai_audio["model"] == "whisper-y"
    assert cfg.transcriber.faster_whisper == FW
    assert transcribers.asr_engine(cfg.transcriber, cfg).backend == "openai_audio:aud/whisper-y"


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
    assert stored["asr_engine"] == "openai_audio"
    assert stored["openai_audio"] == {**OA, "model": "whisper-z"}, "provider/model replace, the rest stays"
    assert registry.current_config().transcriber.openai_audio["model"] == "whisper-z"

    text = await _say(cmd_lib, "transcriber", "faster_whisper")
    assert "faster_whisper:small" in text
    assert json.loads(await cmd_lib.get_setting("engines.transcriber"))["asr_engine"] == "faster_whisper"


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
