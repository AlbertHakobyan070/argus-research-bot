"""Bot API methods newer than python-telegram-bot 22.8 (which stops at Bot API
10.0): rich messages (10.1) and HTML widgets (10.4). They go out raw through
``Bot.do_api_request``, so PTB's ``Defaults`` do not apply and every
parameter is passed explicitly.

A refusal is never silent: the caller falls back to the plain message and
:func:`notice_once` says so in the chat, once per chat and reason for the
life of the process.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

from telegram.error import BadRequest, EndPointNotFound, Forbidden, RetryAfter, TelegramError

logger = logging.getLogger("argus.bot")

_noticed: set[tuple[int, str]] = set()
_sleep = asyncio.sleep   # the RetryAfter wait (tests patch it)


async def _call(bot, method: str, params: dict[str, Any]) -> Any:
    """One raw call. A ``RetryAfter`` (nothing was done) waits its
    ``retry_after`` and tries once more, as progress.edit_text does; a
    second one propagates."""
    kw = {k: v for k, v in params.items() if v is not None}
    try:
        return await bot.do_api_request(method, api_kwargs=kw)
    except RetryAfter as e:
        wait = e.retry_after                     # int, or a timedelta under PTB_TIMEDELTA
        await _sleep(wait.total_seconds() if isinstance(wait, timedelta) else wait)
    return await bot.do_api_request(method, api_kwargs=kw)


async def send_rich(bot, chat_id: int, html: str, *, reply_markup=None) -> int:
    """sendRichMessage with ``html`` content (Bot API 10.1); the new
    message's id. Links are explicit ``<a>`` tags, so automatic entity
    detection stays off (research text is full of $, /, @ and numbers).
    Any TelegramError propagates: the caller sends the plain message."""
    msg = await _call(bot, "sendRichMessage", {
        "chat_id": chat_id, "reply_markup": reply_markup,
        "rich_message": {"html": html, "skip_entity_detection": True}})
    return int(msg["message_id"])


async def send_widget(bot, chat_id: int, html: str, *, chat_data: str, caption: str,
                      reply_markup=None) -> int:
    """sendWidget (Bot API 10.4) with queries allowed and an HTML caption
    (what apps without widgets show); the new message's id."""
    msg = await _call(bot, "sendWidget", {
        "chat_id": chat_id, "html": html, "chat_data": chat_data, "allow_queries": True,
        "caption": caption, "parse_mode": "HTML", "reply_markup": reply_markup})
    return int(msg["message_id"])


async def edit_widget_data(bot, chat_id: int, message_id: int, chat_data: str) -> bool:
    """editMessageWidgetData: True when the widget shows ``chat_data``
    ("message is not modified" included), False when the message is gone.
    Other errors propagate."""
    try:
        await _call(bot, "editMessageWidgetData",
                    {"chat_id": chat_id, "message_id": message_id, "chat_data": chat_data})
    except BadRequest as e:
        reason = str(e).lower()
        if "not modified" in reason:
            return True
        if "not found" in reason or "can't be edited" in reason:
            return False
        raise
    return True


async def edit_caption(bot, chat_id: int, message_id: int, caption: str, *, reply_markup=None) -> None:
    """editMessageCaption (HTML) with the RetryAfter wait; "message is not
    modified" counts as done, other errors propagate."""
    try:
        await _call(bot, "editMessageCaption", {
            "chat_id": chat_id, "message_id": message_id, "caption": caption, "parse_mode": "HTML",
            "reply_markup": reply_markup})
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            raise


async def answer_widget_query(bot, query_id: str, data: str | None = None) -> None:
    await _call(bot, "answerWidgetQuery", {"widget_query_id": query_id, "data": data})


def refused(e: TelegramError) -> bool:
    """True when Telegram refuses the method or its content (a bad request,
    a server without the method, a forbidden chat; production answers
    sendWidget with ``INPUT_MEDIA_WIDGET_NOT_ALLOWED`` since 2026-11-13), as
    against a passing network, timeout or flood error."""
    return isinstance(e, (BadRequest, EndPointNotFound, Forbidden))


async def notice_once(bot, chat_id: int, key: str, text: str) -> None:
    """``text`` in the chat the first time ``key`` happens there; later
    calls do nothing. A notice that cannot be sent is logged, and the next
    occurrence tries again."""
    if (chat_id, key) in _noticed:
        return
    _noticed.add((chat_id, key))
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except TelegramError:
        _noticed.discard((chat_id, key))
        logger.exception("could not send the %s notice to chat %s", key, chat_id)
