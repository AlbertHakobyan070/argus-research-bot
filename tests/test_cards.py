"""The video card (bot/cards.py + its flow in bot/media.py): rendering of
every phase, the pick → download → done / cancelled flow on a fake bot,
the widget form and its fallback when Telegram refuses widgets, widget
queries, /quality buttons and /widgets. Hermetic: no network, no yt-dlp."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from telegram import Update
from telegram.error import BadRequest, TimedOut

from argus import media as media_mod
from argus.bot import cards, tgx
from argus.bot import media as bot_media
from argus.config import Settings

URL = "https://youtu.be/abc123DEF45"
PROBE = {"url": URL, "platform": "youtube", "title": "Fast <inference> & you", "uploader": "Fireship",
         "duration_s": 754, "heights": [144, 360, 480, 720, 1080, 1440, 2160, 4320], "audio_kbps": 160,
         "sizes": {1080: 88_000_000, 720: 41_000_000}, "audio_bytes": 9_000_000}


class _Lib:
    def __init__(self, **kv):
        self.kv = dict(kv)

    async def get_setting(self, key, default=None):
        return self.kv.get(key, default)

    async def set_setting(self, key, value):
        self.kv[key] = value

    async def get_json(self, key, default=None):
        return default

    async def add_asset(self, **kw):
        self.asset = {"asset_id": 7, "kind": "media", "meta": {}, **kw}
        return 7

    async def get_asset(self, asset_id):
        return self.asset


class _Bot:
    """Records every call; do_api_request answers like Telegram, or raises
    ``refuse`` for the methods named in ``refuse_methods``."""

    def __init__(self, refuse: Exception | None = None, refuse_methods=("sendWidget",)):
        self.calls: list[tuple[str, dict]] = []
        self.refuse, self.refuse_methods, self._ids = refuse, refuse_methods, iter(range(500, 600))

    def named(self, name: str) -> list[dict]:
        return [kw for n, kw in self.calls if n == name]

    async def send_message(self, **kw):
        self.calls.append(("send_message", kw))
        return SimpleNamespace(message_id=next(self._ids))

    async def edit_message_text(self, **kw):
        self.calls.append(("edit_message_text", kw))

    async def edit_message_caption(self, **kw):
        self.calls.append(("edit_message_caption", kw))

    async def edit_message_reply_markup(self, **kw):
        self.calls.append(("edit_message_reply_markup", kw))

    async def send_video(self, **kw):
        self.calls.append(("send_video", kw))

    async def send_document(self, **kw):
        self.calls.append(("send_document", kw))

    async def do_api_request(self, endpoint, api_kwargs=None, **_):
        self.calls.append((endpoint, dict(api_kwargs or {})))
        if self.refuse is not None and endpoint in self.refuse_methods:
            raise self.refuse
        return {"message_id": next(self._ids)} if endpoint == "sendWidget" else True


def _app(bot, lib):
    tasks: list[asyncio.Task] = []

    def create_task(coro):
        t = asyncio.ensure_future(coro)
        tasks.append(t)
        return t
    return SimpleNamespace(bot=bot, bot_data={"library": lib}, create_task=create_task, tasks=tasks)


async def _drain(app):
    while any(not t.done() for t in app.tasks):
        await asyncio.gather(*app.tasks, return_exceptions=True)


def _datas(markup):
    return [b.callback_data for row in markup.inline_keyboard for b in row]


def _btns(markup):
    return [b for row in markup.inline_keyboard for b in row]


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    cards._cards.clear()
    tgx._noticed.clear()
    s = Settings(freellmapi_base_url="http://127.0.0.1:9/v1", freellmapi_api_key="", telegram_bot_token="123:test",
                 telegram_allowed_user_id=1, reports_root=tmp_path, checkpoint_db=tmp_path / "cp.sqlite",
                 library_db=tmp_path / "lib.sqlite", vault_root=tmp_path, media_root=tmp_path / "media",
                 transcripts_root=tmp_path / "tr", history_root=tmp_path / "h", ffmpeg_path=None)
    monkeypatch.setattr(bot_media, "get_settings", lambda: s)
    monkeypatch.setattr(bot_media, "_allowed", lambda _s, uid: uid == 1)

    async def no_autotag(*a, **kw):
        return None
    monkeypatch.setattr(bot_media, "_autotag", no_autotag)
    monkeypatch.setattr(cards, "widget_html", lambda: "<p>card</p>")
    yield
    cards._cards.clear()
    tgx._noticed.clear()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _picked(**kw) -> cards.Card:
    c = cards.Card(chat_id=1, url=URL, platform="youtube", **kw)
    cards.fill(c, PROBE, "auto")
    c.phase = "pick"
    return c


def test_the_pick_step_offers_the_links_own_heights_and_the_settings_default():
    c = _picked()
    assert c.video_opts == ["4320", "2160", "1440", "1080", "720", "480"], "highest six, highest first"
    assert c.quality == "1080", "auto caps at 1080"
    kb = cards.keyboard(c)
    rows = kb.inline_keyboard
    assert [b.text for b in rows[0]] == ["🎬 Video ✓", "🎵 MP3"]
    chosen = [b for row in rows[1:-1] for b in row if b.callback_data == "vc:q:1080"][0]
    assert chosen.text == "1080p ✓" and chosen.style == "primary"
    assert [(b.text, b.callback_data, b.style) for b in rows[-1]] == [
        ("⬇️ Download", "vc:ok:_", "success"), ("✖ Cancel", "vc:x:_", "danger")]
    for d in _datas(kb):
        assert len(d.encode()) <= 64 and len(d.split(":")) == 3, d
    text = cards.render_text(c)
    assert "Fast &lt;inference&gt; &amp; you" in text and "<inference>" not in text
    assert "Video · 1080p" in text and "≈ 83.9 MB" in text
    assert "Fireship · 12:34 · youtube" in text


@pytest.mark.parametrize("setting,expect", [("max", "4320"), ("min", "480"), ("720", "720"),
                                            ("600", "480"), ("100", "480"), ("auto", "1080")])
def test_the_quality_setting_maps_onto_the_offered_heights(setting, expect):
    assert cards.default_quality(setting, ["4320", "2160", "1440", "1080", "720", "480"]) == expect
    assert cards.default_quality(setting, ["auto"]) == "auto"


def test_format_and_quality_taps_change_only_a_card_at_its_pick_step():
    c = _picked()
    assert cards.apply(c, "f", "a") and c.fmt == "a"
    assert "MP3 · 320 kbps" in cards.render_text(c)
    assert [b.callback_data for b in cards.keyboard(c).inline_keyboard[1]] == ["vc:a:320", "vc:a:v0", "vc:a:192"]
    assert cards.apply(c, "q", "720") and (c.fmt, c.quality) == ("v", "720")
    assert not cards.apply(c, "q", "721"), "only offered heights"
    assert not cards.apply(c, "q", "720"), "no change, no redraw"
    c.phase = "dl"
    assert not cards.apply(c, "q", "1080") and c.quality == "720"


def test_the_progress_and_end_phases_render():
    c = _picked()
    c.phase, c.stage, c.pct, c.done_bytes, c.total_bytes, c.eta = "dl", "video", 50.0, 44_040_192, 88_080_384, "0:31"
    text = cards.render_text(c)
    assert "<code>██████░░░░░░</code> 50%" in text
    assert "Video stream · 42.0 MB / 84.0 MB · ETA 0:31" in text
    assert _datas(cards.keyboard(c)) == ["vc:x:_"]
    c.phase, c.error = "fail", "Download failed (youtube): <403>"
    assert "&lt;403&gt;" in cards.render_text(c) and _datas(cards.keyboard(c)) == ["vc:retry:_"]
    c.phase = "x"
    assert "Cancelled" in cards.render_text(c) and cards.keyboard(c) is None


def test_the_widget_state_fits_1024_bytes_with_a_long_cyrillic_title():
    c = _picked()
    c.title, c.meta = "Очень длинное название видео " * 12, "Канал с длинным именем " * 5
    for phase in ("read", "pick", "queue", "dl", "done", "fail", "x"):
        c.phase, c.error, c.summary = phase, "ошибка " * 80, "1080p MP4 · 84.1 MB"
        s = cards.widget_state(c)
        assert len(s.encode("utf-8")) <= 1024, phase
        assert json.loads(s)["ph"] == phase


# ---------------------------------------------------------------------------
# the flow on a fake bot
# ---------------------------------------------------------------------------

async def _open(monkeypatch, bot, lib, *, fmt="v"):
    async def probe(url, **kw):
        return dict(PROBE)
    monkeypatch.setattr(bot_media, "probe_url", probe)
    app = _app(bot, lib)
    card = await bot_media.open_card(app, 1, URL, fmt=fmt)
    await _drain(app)
    return app, card


async def test_pick_download_done_on_one_message(monkeypatch, tmp_path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"z" * 2048)
    seen = {}

    async def download(url, **kw):
        seen.update(kw)
        kw["on_status"]({"stage": "video", "pct": 40.0, "eta": "0:10", "done_bytes": 800, "total_bytes": 2048})
        await asyncio.sleep(0)
        return media_mod.MediaResult(ok=True, url=url, platform="youtube", path=str(vid), title="Clip",
                                     size_bytes=2048, duration_s=754)
    monkeypatch.setattr(bot_media, "download_media", download)
    bot, lib = _Bot(), _Lib()
    app, card = await _open(monkeypatch, bot, lib)
    assert card.phase == "pick" and card.message_id == 500
    first = bot.named("send_message")[0]
    assert first["link_preview_options"].url == URL and first["link_preview_options"].show_above_text

    await bot_media.card_action(app, 1, 500, "q", "720")
    assert "720p ✓" in [b.text for b in _btns(bot.named("edit_message_text")[-1]["reply_markup"])]
    await bot_media.card_action(app, 1, 500, "ok", "_")
    await _drain(app)
    assert seen["quality"] == "720" and seen["audio"] is None
    final = bot.named("edit_message_text")[-1]
    assert final["message_id"] == 500, "the same message from start to finish"
    assert final["text"].startswith("✅") and "720p MP4 · 2 KB · 12:34 · asset:7" in final["text"]
    assert "mx:tag:7" in _datas(final["reply_markup"]), "the item buttons move onto the card"
    assert len(bot.named("send_message")) == 1, "no separate Saved / title messages"
    assert bot.named("send_video"), "the file still comes"


async def test_cancel_stops_the_download_and_ends_the_card(monkeypatch):
    started = asyncio.Event()

    async def download(url, **kw):
        started.set()
        await asyncio.sleep(3600)
    monkeypatch.setattr(bot_media, "download_media", download)
    bot = _Bot()
    app, card = await _open(monkeypatch, bot, _Lib())
    await bot_media.card_action(app, 1, card.message_id, "ok", "_")
    await started.wait()
    await bot_media.card_action(app, 1, card.message_id, "x", "_")
    await _drain(app)
    assert card.phase == "x" and card.task.cancelled()
    last = bot.named("edit_message_text")[-1]
    assert "Cancelled" in last["text"] and last["reply_markup"] is None


async def test_a_failed_download_offers_try_again_which_returns_to_the_pick(monkeypatch):
    async def download(url, **kw):
        return media_mod.MediaResult(ok=False, url=url, platform="youtube", error="HTTP Error 403: Forbidden")
    monkeypatch.setattr(bot_media, "download_media", download)
    bot = _Bot()
    app, card = await _open(monkeypatch, bot, _Lib())
    await bot_media.card_action(app, 1, card.message_id, "ok", "_")
    await _drain(app)
    assert card.phase == "fail" and "403" in bot.named("edit_message_text")[-1]["text"]
    await bot_media.card_action(app, 1, card.message_id, "retry", "_")
    assert card.phase == "pick" and "Pick a format" in bot.named("edit_message_text")[-1]["text"]


async def test_a_link_that_cannot_be_read_says_why_on_the_card(monkeypatch):
    async def probe(url, **kw):
        raise RuntimeError("Private video. Sign in if you've been granted access")
    monkeypatch.setattr(bot_media, "probe_url", probe)
    bot = _Bot()
    app = _app(bot, _Lib())
    card = await bot_media.open_card(app, 1, URL)
    await _drain(app)
    assert card.phase == "fail" and card.failed_at == "probe"
    assert "Private video" in bot.named("edit_message_text")[-1]["text"]


class _Recorder:
    def __init__(self):
        self.kwargs = None

    async def __call__(self, **kw):
        self.kwargs = kw


async def test_an_expired_card_says_so_and_drops_only_its_own_buttons():
    from telegram import InlineKeyboardButton as B, InlineKeyboardMarkup as M
    bot = _Bot()
    done = M([[B("🎵 MP3", callback_data="mx:mp3:7"), B("🏷 Tags", callback_data="mx:tag:7")],
              [B("🔘 Regular buttons", callback_data="vc:plain:_")]])
    q = SimpleNamespace(message=SimpleNamespace(chat=SimpleNamespace(id=1), message_id=999, reply_markup=done),
                        edit_message_reply_markup=_Recorder())
    update = SimpleNamespace(callback_query=q, effective_chat=SimpleNamespace(id=1))
    ctx = SimpleNamespace(application=_app(bot, _Lib()))
    await bot_media.handle_card_callback(update, ctx, "plain", "_")
    assert _datas(q.edit_message_reply_markup.kwargs["reply_markup"]) == ["mx:mp3:7", "mx:tag:7"], (
        "the item buttons still work after a restart")
    assert "expired" in bot.named("send_message")[-1]["text"]
    q.message.reply_markup = M([[B("✖ Cancel", callback_data="vc:x:_")]])
    await bot_media.handle_card_callback(update, ctx, "x", "_")
    assert q.edit_message_reply_markup.kwargs == {"reply_markup": None}


class _FlakyBot(_Bot):
    """edit_message_text raises ``fail`` for the next ``n`` edits, each edit
    taking ``latency`` seconds; records how many edits overlap."""

    def __init__(self, fail=None, n=0, latency=0.0):
        super().__init__()
        self.fail, self.n, self.latency, self.inflight, self.overlap = fail, n, latency, 0, 0

    async def edit_message_text(self, **kw):
        self.inflight += 1
        self.overlap = max(self.overlap, self.inflight)
        try:
            await asyncio.sleep(self.latency)
            if self.n:
                self.n -= 1
                raise self.fail
            self.calls.append(("edit_message_text", kw))
        finally:
            self.inflight -= 1

    async def send_audio(self, **kw):
        self.calls.append(("send_audio", kw))


async def test_a_card_that_cannot_be_edited_is_sent_again_and_its_buttons_still_work(monkeypatch):
    bot = _FlakyBot(fail=TimedOut("Timed out"), n=1)          # the pick redraw times out
    app, card = await _open(monkeypatch, bot, _Lib())
    assert card.phase == "pick" and card.message_id == 501, "the pick step came as a new message"
    resent = bot.named("send_message")[-1]
    assert "vc:ok:_" in _datas(resent["reply_markup"])
    assert cards.get(1, 500) is None and cards.get(1, 501) is card, "the card moved with it"
    assert await bot_media.card_action(app, 1, 501, "f", "a", answer=None) is card and card.fmt == "a"


async def test_an_outcome_whose_card_was_deleted_comes_as_a_new_message_that_works(monkeypatch):
    async def download(url, **kw):
        return media_mod.MediaResult(ok=False, url=url, platform="youtube", error="HTTP Error 403: Forbidden")
    monkeypatch.setattr(bot_media, "download_media", download)
    bot = _FlakyBot(fail=BadRequest("Message to edit not found"))
    app, card = await _open(monkeypatch, bot, _Lib())
    bot.n = 9                                                 # the user deleted the card
    await bot_media.card_action(app, 1, card.message_id, "ok", "_")
    await _drain(app)
    out = bot.named("send_message")[-1]
    assert "403" in out["text"] and _datas(out["reply_markup"]) == ["vc:retry:_"]
    assert cards.get(1, card.message_id) is card and not card.gone
    bot.n = 0
    await bot_media.card_action(app, 1, card.message_id, "retry", "_")
    assert card.phase == "pick", "Try again on the new message works"


async def test_a_change_during_a_redraw_is_drawn_after_it(monkeypatch):
    monkeypatch.setattr(cards, "_PROGRESS_INTERVAL_S", 0.01)
    bot = _FlakyBot(latency=0.05)
    c = _picked()
    c.message_id, c.phase, c.stage = 500, "dl", "audio"
    cards.kick(bot, c)
    await asyncio.sleep(0.02)                                 # the first redraw is on the wire
    c.stage = "merging"
    cards.kick(bot, c)
    await c.pending
    assert "Merging video and audio" in bot.named("edit_message_text")[-1]["text"]


async def test_redraws_of_one_card_never_overlap_and_the_last_state_wins():
    bot = _FlakyBot(latency=0.05)
    c = _picked()
    c.message_id, c.phase = 500, "queue"
    first = asyncio.ensure_future(cards.settle(bot, c))
    await asyncio.sleep(0.01)
    c.phase, c.error = "fail", "Download failed (youtube): no ffmpeg"
    await asyncio.gather(first, cards.settle(bot, c))
    assert bot.overlap == 1
    assert "no ffmpeg" in bot.named("edit_message_text")[-1]["text"]


async def test_cancel_after_the_file_arrived_is_too_late_and_the_card_ends_done(monkeypatch, tmp_path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"z" * 10)
    gate = asyncio.Event()

    class SlowLib(_Lib):
        async def add_asset(self, **kw):
            await gate.wait()
            return await super().add_asset(**kw)

    async def download(url, **kw):
        return media_mod.MediaResult(ok=True, url=url, platform="youtube", path=str(vid), title="Clip",
                                     size_bytes=10, duration_s=3)
    monkeypatch.setattr(bot_media, "download_media", download)
    bot = _Bot()
    app, card = await _open(monkeypatch, bot, SlowLib())
    await bot_media.card_action(app, 1, card.message_id, "ok", "_")
    await asyncio.sleep(0.01)                                 # registering the asset
    await bot_media.card_action(app, 1, card.message_id, "x", "_")
    gate.set()
    await _drain(app)
    assert card.phase == "done" and "asset:7" in card.summary


async def test_a_new_stage_and_a_new_attempt_drop_the_old_numbers(monkeypatch):
    seen = []

    async def download(url, **kw):
        kw["on_status"]({"stage": "video", "pct": 100.0, "eta": "00:31", "done_bytes": 12, "total_bytes": 80})
        kw["on_status"]({"stage": "merging", "pct": None, "eta": None, "done_bytes": None, "total_bytes": None})
        seen.append(cards._progress_line(card))
        return media_mod.MediaResult(ok=False, url=url, platform="youtube", error="boom")
    monkeypatch.setattr(bot_media, "download_media", download)
    bot = _Bot()
    app, card = await _open(monkeypatch, bot, _Lib())
    await bot_media.card_action(app, 1, card.message_id, "ok", "_")
    await _drain(app)
    assert seen == ["Merging video and audio"]
    card.pct, card.eta, card.done_bytes, card.total_bytes = 12.0, "00:31", 12, 80
    await bot_media.card_action(app, 1, card.message_id, "retry", "_")
    await bot_media.card_action(app, 1, card.message_id, "ok", "_")
    assert cards._progress_line(card) == "Starting"
    await _drain(app)


def test_quality_words_are_labelled_for_what_they_do():
    assert [cards._qlabel(q) for q in ("720", "auto", "max", "min")] == ["720p", "Auto (≤1080p)", "Max", "Min"]


async def test_an_mp3_the_library_could_not_take_is_still_sent_as_audio(monkeypatch, tmp_path):
    mp3 = tmp_path / "song.mp3"
    mp3.write_bytes(b"z" * 10)

    async def download(url, **kw):
        return media_mod.MediaResult(ok=True, url=url, platform="youtube", path=str(mp3), title="Song",
                                     size_bytes=10, duration_s=3)
    monkeypatch.setattr(bot_media, "download_media", download)
    bot = _FlakyBot()
    app, card = await _open(monkeypatch, bot, None, fmt="a")
    await bot_media.card_action(app, 1, card.message_id, "ok", "_")
    await _drain(app)
    assert bot.named("send_audio") and not bot.named("send_video")


# ---------------------------------------------------------------------------
# the widget form and its fallback
# ---------------------------------------------------------------------------

async def test_a_refused_widget_switches_widgets_off_tells_once_and_sends_the_card(monkeypatch):
    bot = _Bot(refuse=BadRequest("INPUT_MEDIA_WIDGET_NOT_ALLOWED"))
    lib = _Lib(**{cards.WIDGETS_KEY: "on"})
    app, card = await _open(monkeypatch, bot, lib)
    assert [n for n, _ in bot.calls][:1] == ["sendWidget"]
    assert lib.kv[cards.WIDGETS_KEY] == "off"
    notices = [kw["text"] for kw in bot.named("send_message") if kw["text"].startswith("ℹ️")]
    assert len(notices) == 1 and "INPUT_MEDIA_WIDGET_NOT_ALLOWED" in notices[0] and "/widgets on" in notices[0]
    assert not card.widget and card.message_id is not None
    assert "Reading the link" in [kw["text"] for kw in bot.named("send_message")][-1], "the regular card follows"

    lib.kv[cards.WIDGETS_KEY] = "on"                     # turned on again by hand, refused again
    await _open(monkeypatch, bot, lib)
    notices = [kw["text"] for kw in bot.named("send_message") if kw["text"].startswith("ℹ️")]
    assert len(notices) == 1, "one notice per chat, not one per card"


async def test_a_passing_widget_error_falls_back_without_turning_widgets_off(monkeypatch):
    bot = _Bot(refuse=TimedOut("Timed out"))
    lib = _Lib(**{cards.WIDGETS_KEY: "on"})
    _app_, card = await _open(monkeypatch, bot, lib)
    assert lib.kv[cards.WIDGETS_KEY] == "on" and not card.widget
    assert any("could not be sent" in kw["text"] for kw in bot.named("send_message"))


async def test_the_widget_card_and_its_queries(monkeypatch):
    bot = _Bot()
    lib = _Lib(**{cards.WIDGETS_KEY: "on"})
    app, card = await _open(monkeypatch, bot, lib)
    sent = bot.named("sendWidget")[0]
    assert card.widget and sent["allow_queries"] is True and sent["parse_mode"] == "HTML"
    assert sent["caption"].startswith("🎬") and len(sent["caption"]) <= 1024
    assert _datas(sent["reply_markup"]) == ["vc:plain:_"], "apps without widgets get a way back"
    pushed = json.loads(bot.named("editMessageWidgetData")[-1]["chat_data"])
    assert pushed["ph"] == "pick" and pushed["q"] == "1080"

    update = Update.de_json({"update_id": 9, "widget_query": {
        "id": "wq1", "from": {"id": 1, "is_bot": False, "first_name": "A"},
        "chat": {"id": 1, "type": "private"}, "message_id": card.message_id,
        "chat_instance": "c", "data": "q:720"}}, None)
    assert cards.WidgetQueryHandler(bot_media.handle_widget_query).check_update(update)
    await bot_media.handle_widget_query(update, SimpleNamespace(application=app))
    names = [n for n, _ in bot.calls]
    answer_at = names.index("answerWidgetQuery")
    assert names.index("editMessageWidgetData", answer_at) > answer_at, "answer first, then push the state"
    answer = bot.named("answerWidgetQuery")[-1]
    assert answer["widget_query_id"] == "wq1" and json.loads(answer["data"])["q"] == "720"


async def test_a_stranger_widget_query_is_answered_and_ignored(monkeypatch):
    bot = _Bot()
    app, card = await _open(monkeypatch, bot, _Lib(**{cards.WIDGETS_KEY: "on"}))
    update = Update.de_json({"update_id": 9, "widget_query": {
        "id": "wq2", "from": {"id": 666, "is_bot": False, "first_name": "M"},
        "chat": {"id": 1, "type": "private"}, "message_id": card.message_id,
        "chat_instance": "c", "data": "ok"}}, None)
    await bot_media.handle_widget_query(update, SimpleNamespace(application=app))
    assert json.loads(bot.named("answerWidgetQuery")[-1]["data"])["err"] == "Unauthorized."
    assert card.phase == "pick"


async def test_a_widget_caption_follows_the_phase_only(monkeypatch):
    bot = _Bot()
    app, card = await _open(monkeypatch, bot, _Lib(**{cards.WIDGETS_KEY: "on"}))
    caps = bot.named("editMessageCaption")
    assert len(caps) == 1 and "Choose a format" in caps[0]["caption"], "read → pick"
    await bot_media.card_action(app, 1, card.message_id, "f", "a")
    assert len(bot.named("editMessageCaption")) == 1, "a choice redraws the widget only"
    assert "MP3" not in cards.render_text(card, short=True), "so the caption never shows a stale choice"
    card.phase, card.pct = "dl", None
    assert "p" not in json.loads(cards.widget_state(card)), "an unknown percent is not drawn as 0%"


async def test_regular_buttons_move_the_card_and_switch_the_old_widget_off(monkeypatch):
    bot = _Bot()
    app, card = await _open(monkeypatch, bot, _Lib(**{cards.WIDGETS_KEY: "on"}))
    old = card.message_id
    await bot_media.card_action(app, 1, old, "plain", "_")
    assert not card.widget and card.message_id != old and cards.get(1, card.message_id) is card
    off = [json.loads(kw["chat_data"]) for kw in bot.named("editMessageWidgetData") if kw["message_id"] == old][-1]
    assert off["ph"] == "off" and "Moved" in off["err"], "no buttons left that act on the old widget"


def test_widget_states_carry_a_rising_revision():
    c = _picked()
    a, b = json.loads(cards.widget_state(c)), json.loads(cards.off_state("Moved.", c))
    assert b["n"] > a["n"], "the widget keeps the newer of an answer and a push"
    assert "n" not in json.loads(cards.off_state("This card expired.")), "no card, no revision"


async def test_a_flood_wait_on_a_raw_call_waits_and_tries_once_more(monkeypatch):
    from telegram.error import RetryAfter
    waits = []

    async def sleep(s):
        waits.append(s)
    monkeypatch.setattr(tgx, "_sleep", sleep)

    class Once(_Bot):
        async def do_api_request(self, endpoint, api_kwargs=None, **_):
            self.calls.append((endpoint, dict(api_kwargs or {})))
            if len(self.calls) == 1:
                raise RetryAfter(3)
            return True
    bot = Once()
    assert await tgx.edit_widget_data(bot, 1, 5, "{}") is True
    assert waits == [3] and len(bot.calls) == 2


async def test_a_working_widget_rearms_the_failure_notice(monkeypatch):
    tgx._noticed.add((1, "widget-error"))
    await _open(monkeypatch, _Bot(), _Lib(**{cards.WIDGETS_KEY: "on"}))
    assert (1, "widget-error") not in tgx._noticed, "the next failure is said again"


async def test_a_notice_that_could_not_be_sent_is_tried_again():
    class Down(_Bot):
        fails = 1

        async def send_message(self, **kw):
            if self.fails:
                self.fails -= 1
                raise TimedOut("Timed out")
            return await super().send_message(**kw)
    bot = Down()
    await tgx.notice_once(bot, 1, "k", "hello")
    await tgx.notice_once(bot, 1, "k", "hello")
    await tgx.notice_once(bot, 1, "k", "hello")
    assert [kw["text"] for kw in bot.named("send_message")] == ["hello"]


def test_polling_asks_for_widget_queries():
    import inspect

    from argus.bot import app as bot_app
    assert '"widget_query"' in inspect.getsource(bot_app.main)


# ---------------------------------------------------------------------------
# /quality buttons and /widgets
# ---------------------------------------------------------------------------

class _Replies:
    def __init__(self):
        self.calls = []

    async def __call__(self, text, **kw):
        self.calls.append((text, kw))


def _cmd(args, lib, bot=None):
    update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_chat=SimpleNamespace(id=1),
                             message=SimpleNamespace(reply_text=_Replies()))
    ctx = SimpleNamespace(args=args, application=SimpleNamespace(bot=bot or _Bot(), bot_data={"library": lib}))
    return update, ctx


async def test_quality_buttons_mark_the_current_value_and_set_a_new_one():
    lib = _Lib(media_quality="720")
    update, ctx = _cmd([], lib)
    await bot_media.quality_cmd(update, ctx)
    text, kw = update.message.reply_text.calls[-1]
    marked = [b for b in _btns(kw["reply_markup"]) if b.style == "primary"]
    assert [(b.text, b.callback_data) for b in marked] == [("720p ✓", "qs:set:720")]

    q = SimpleNamespace(message=SimpleNamespace(chat=SimpleNamespace(id=1), message_id=33))
    await bot_media.handle_quality_callback(SimpleNamespace(callback_query=q), ctx, "set", "1080")
    assert lib.kv["media_quality"] == "1080"
    edit = ctx.application.bot.named("edit_message_text")[-1]
    assert "Set to 1080" in edit["text"] and "1080p ✓" in [b.text for b in _btns(edit["reply_markup"])]


async def test_widgets_command_switches_and_rearms_the_notice():
    lib = _Lib()
    tgx._noticed.update({(1, "widget"), (1, "widget-error")})
    update, ctx = _cmd(["on"], lib)
    await bot_media.widgets_cmd(update, ctx)
    assert lib.kv[cards.WIDGETS_KEY] == "on" and not tgx._noticed
    assert update.message.reply_text.calls[-1][0].startswith("🧩 HTML widgets: on")
    update, ctx = _cmd(["sideways"], lib)
    await bot_media.widgets_cmd(update, ctx)
    assert update.message.reply_text.calls[-1][0].startswith("Usage")


# ---------------------------------------------------------------------------
# tgx on the wire (a real telegram.Bot over a mock transport)
# ---------------------------------------------------------------------------

async def test_send_widget_and_edit_widget_data_on_the_wire():
    import urllib.parse

    import httpx
    import telegram
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.request import HTTPXRequest

    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = {k: v[0] for k, v in urllib.parse.parse_qs(req.content.decode()).items()}
        seen.append((req.url.path.rsplit("/", 1)[-1], body))
        if seen[-1][0] == "editMessageWidgetData":
            return httpx.Response(400, json={"ok": False, "error_code": 400,
                                             "description": "Bad Request: message is not modified"})
        return httpx.Response(200, json={"ok": True, "result": {
            "message_id": 77, "date": 0, "chat": {"id": 1, "type": "private"}}})

    bot = telegram.Bot("123:test", request=HTTPXRequest(httpx_kwargs={"transport": httpx.MockTransport(handler)}))
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔘 Regular buttons", callback_data="vc:plain:_")]])
    mid = await tgx.send_widget(bot, 1, "<p>x</p>", chat_data='{"v":1}', caption="<b>c</b>", reply_markup=kb)
    assert mid == 77
    name, body = seen[-1]
    assert name == "sendWidget" and body["html"] == "<p>x</p>" and body["chat_data"] == '{"v":1}'
    assert body["allow_queries"] == "true" and body["parse_mode"] == "HTML"
    assert json.loads(body["reply_markup"])["inline_keyboard"][0][0]["callback_data"] == "vc:plain:_"
    assert await tgx.edit_widget_data(bot, 1, 77, '{"v":1}') is True, "not modified counts as shown"
