"""Print a local console URL signed for the allowed user (v4 spec §3 WP-E2).

    ./venv/Scripts/python.exe scripts/webapp_dev_url.py [--port 8787]

Signs initData (``user``, ``auth_date``, ``hash``) with the bot token from
.env through ``webapp.auth.sign_init_data`` and prints
``http://127.0.0.1:<port>/#tgWebAppData=…&tgWebAppVersion=9.0&…``. The
initData rides in the URL fragment, which a browser never sends to the
server; the console's auth accepts it for ``ARGUS_WEBAPP_INITDATA_MAX_AGE_S``
(default 1 h). Never prints the token. Exits non-zero when the console
server is not reachable on that port.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.parse import quote

import httpx

from argus.config import get_settings
from argus.webapp.auth import sign_init_data

THEME = {"bg_color": "#ffffff", "text_color": "#000000", "hint_color": "#999999", "link_color": "#2481cc",
         "button_color": "#2481cc", "button_text_color": "#ffffff", "secondary_bg_color": "#f1f1f1"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=None, help="the console port (default ARGUS_WEBAPP_PORT)")
    args = parser.parse_args(argv)
    s = get_settings()
    base = f"http://127.0.0.1:{args.port or s.webapp_port}"
    try:
        httpx.get(base + "/", timeout=5.0, trust_env=False)      # any answer: the server is up
    except httpx.HTTPError as e:
        print(f"the console server is not reachable at {base} ({type(e).__name__}) — is the bot running?",
              file=sys.stderr)
        return 1
    init_data = sign_init_data({"user": {"id": s.telegram_allowed_user_id, "first_name": "Albert"},
                                "auth_date": int(time.time())}, s.telegram_bot_token)
    theme = quote(json.dumps(THEME, separators=(",", ":")), safe="")
    print(f"{base}/#tgWebAppData={quote(init_data, safe='')}&tgWebAppVersion=9.0&tgWebAppPlatform=weba"
          f"&tgWebAppThemeParams={theme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
