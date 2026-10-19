"""Argus Telegram bot layer.

python-telegram-bot v22 async, long-polling. Commands (registered in
``build_application``; ``HELP_TEXT`` is the user-facing list):
  research  /research [/length <m>] <topic> (both HITL gates) · /ask <q>
            (one ungrounded cheap-tier answer — no search)
  media     /find [yt|shorts|reddit] <q> · /video [shorts] <q> · /fetch <url…>
            · /quality · /transcript <indices> · /transcripts · pasted links
  history   /runs · /status · /append <id> <ref…> · /continue <id>
            · /delete · /cancel [id|all]
  /help [full]

The bot is single-user (TELEGRAM_ALLOWED_USER_ID) for safety.

v2 architecture (Phase 1)
-------------------------
- ONE long-lived AsyncSqliteSaver + ONE compiled graph per flavour
  (deep/quick), opened in PTB ``post_init`` and stored on
  ``application.bot_data``. Runs are isolated by per-run thread ids
  ``tg:<chat>:<run8>`` — multiple concurrent runs per chat are legal,
  and any run can be re-attached after a bot restart.
- Every run is registered in the Library (SQLite) so /status lists
  history and later phases can /append + /continue + /delete.
- Callback data is run-scoped (``plan:approve:<run8>``) so buttons on
  old messages can never act on the wrong run.

T7 additions
------------
- 5-button length selector at plan approval (TLDR / Short / Medium /
  Long / Lecture). The chosen mode is pushed back into the graph
  checkpoint via ``graph.aupdate_state`` before ``Command(resume=)``,
  so the compose + report_builder nodes see the user's HITL choice.
- /length flag in /research for users who want to skip the keyboard.
- Validation summary on title page + per-section confidence in the
  markdown + redesigned PDF.
"""
from .app import main  # the entry point; nothing else is re-exported
