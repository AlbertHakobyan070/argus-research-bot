"""Telegram Mini App ``initData`` (v4 spec §2.7): validated on every
``/api/*`` request (``Authorization: tma <initData>``); signed only by the
dev URL script. An ``AuthError`` carries its reason code and the *names* of
the fields present, the only things the server logs — never a value."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode

CODES = ("missing_field", "hash_mismatch", "expired", "wrong_user")


class AuthError(Exception):
    def __init__(self, code: str, fields: tuple[str, ...] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.fields = fields

    @property
    def status(self) -> int:
        return 403 if self.code == "wrong_user" else 401


@dataclass(frozen=True)
class AuthUser:
    id: int
    first_name: str


def _hash(pairs: list[tuple[str, str]], bot_token: str) -> str:
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(pairs) if k != "hash")
    return hmac.new(secret, data_check_string.encode(), hashlib.sha256).hexdigest()


def validate_init_data(init_data: str, bot_token: str, allowed_user_id: int, max_age_s: float, *,
                       now: float | None = None) -> AuthUser:
    """The signed user, or ``AuthError``: ``missing_field`` (no ``hash`` /
    ``auth_date``, or an unreadable ``user``), ``hash_mismatch``, ``expired``
    (older than ``max_age_s``), ``wrong_user`` (403)."""
    pairs = parse_qsl(init_data or "", keep_blank_values=True)
    fields = tuple(k for k, _ in pairs)
    data = dict(pairs)
    if "hash" not in data or "auth_date" not in data:
        raise AuthError("missing_field", fields)
    if not hmac.compare_digest(_hash(pairs, bot_token).encode(), data["hash"].encode()):
        raise AuthError("hash_mismatch", fields)
    try:
        auth_date = int(data["auth_date"])
        user = json.loads(data["user"])
        user_id = int(user["id"])
    except (KeyError, ValueError, TypeError):
        raise AuthError("missing_field", fields) from None
    if (time.time() if now is None else now) - auth_date > max_age_s:
        raise AuthError("expired", fields)
    if user_id != allowed_user_id:
        raise AuthError("wrong_user", fields)
    return AuthUser(id=user_id, first_name=str(user.get("first_name") or ""))


def sign_init_data(fields: dict, bot_token: str) -> str:
    """initData for ``fields`` (a dict value is JSON-encoded, as Telegram's
    ``user`` is), signed like Telegram signs it. For the dev URL script only."""
    pairs = [(k, json.dumps(v, separators=(",", ":")) if isinstance(v, dict) else str(v)) for k, v in fields.items()]
    return urlencode([*pairs, ("hash", _hash(pairs, bot_token))])
