"""The Mini App console (v4 spec §2.7): an aiohttp server on 127.0.0.1 over
the run driver (``bot/runs.py``), with Telegram initData auth, sanitised
report HTML and a supervised cloudflared quick tunnel. Optional: a failure
here is visible (log, DM, ``/status``) and never affects the bot."""
