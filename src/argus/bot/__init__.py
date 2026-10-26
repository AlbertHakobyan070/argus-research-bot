"""Argus Telegram bot layer.

python-telegram-bot v22 async, long-polling. Each handler module registers
its own commands and callbacks through ``register(app)``; ``app.py`` lists
the modules in ``HANDLER_MODULES`` and wires the lifecycle hooks
(``POST_INIT_HOOKS``, ``POST_STOP_HOOKS``, ``POST_SHUTDOWN_HOOKS``).
``common.HELP_TEXT`` is the user-facing command list, ``docs/help.md`` the
full guide.

  research.py    /research, /ask, the plan and report buttons, Edit / Revise replies
  runs.py        the run driver: one live drive per run, cancel by drain,
                 attach on action, /continue, the plan card's data
  progress.py    the run event bus and the live Telegram progress card
  media.py       /find, /video, /fetch, /quality, /transcript(s), /media, pasted links, sent files
  history.py     /runs, /status, /append, /continue, /cancel, /help
  delete.py      /delete
  engines_cmd.py /engines (profiles, roles, tiers, the transcriber)
  app.py         startup (engines check, restart reconcile), shutdown, dispatch
  common.py      shared helpers and state

The bot is single-user (TELEGRAM_ALLOWED_USER_ID) for safety. Runs live in
per-run checkpoint threads ``tg:<chat>:<run8>`` and in the Library, so any
run can be re-attached after a restart; callback data is run-scoped, so a
button on an old message never acts on the wrong run. The Mini App console
(``argus.webapp``) is started by the same hooks and calls the same driver.
"""
from .app import main  # the entry point; nothing else is re-exported
