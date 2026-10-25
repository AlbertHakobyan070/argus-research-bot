"""/engines — show and change the engine bindings (v4 WP-A, spec §2.2/§2.3),
and the error handler that answers an ``EngineConfigError`` raised by any
drive entry with its text (spec §2.2: the run is refused with that text).

Changes go through the registry, which validates before it persists:
``use`` switches the active profile; ``tier``, ``set``, ``fallback`` and
``reset`` edit the active profile's library-layer overrides (``reset`` takes
any profile). Provider definitions are file-only and never change here."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from pathlib import Path

from telegram import Update
from telegram.ext import Application, ApplicationHandlerStop, CommandHandler, ContextTypes

from ..config import get_settings
from ..engines import registry, transcribers
from ..engines.registry import EngineConfigError
from ..engines.roles import ROLES
from ..engines.schema import EngineSpec, EnginesConfig, ProfileOverrides
from .common import _allowed, _get_library, _safe_send

logger = logging.getLogger("argus.bot")

USAGE = """\
/engines — profile, bindings, validation, last run
/engines use <profile>
/engines set <role> <provider>/<model>
/engines tier <cheap|strong|judge> <provider>/<model>
/engines fallback <role> <provider>/<model>
/engines reset [profile]
/engines models <provider>
/engines transcriber [faster_whisper|openai_audio [provider/model]]"""
_TIERS = ("cheap", "strong", "judge")


async def engines_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    try:
        text = await _dispatch(list(ctx.args or []), _get_library(ctx), update.effective_chat.id)
    except (EngineConfigError, ValueError) as e:
        text = f"⚠️ {e}"
    await update.message.reply_text(text[:4000])


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Send an ``EngineConfigError``'s text to the chat, and stop the generic
    error handler from logging an expected refusal as an unhandled error.
    Other errors pass through untouched."""
    err = getattr(ctx, "error", None)
    if not isinstance(err, EngineConfigError):
        return
    logger.warning("engines: refused: %s", err)
    chat = getattr(update, "effective_chat", None)
    if chat is not None:
        await _safe_send(ctx.bot, chat.id, f"⚠️ {err}", parse_mode=None)
    raise ApplicationHandlerStop


def register(app: Application) -> None:
    app.add_handler(CommandHandler("engines", engines_cmd))
    app.add_error_handler(on_error)


def _spec(arg: str) -> EngineSpec:
    provider, sep, model = arg.partition("/")
    if not (sep and provider and model):
        raise ValueError(f"expected provider/model, got {arg!r}")
    return EngineSpec(provider=provider, model=model)


async def _dispatch(args: list[str], lib, chat_id: int) -> str:
    sub, rest = (args[0].lower(), args[1:]) if args else ("", [])
    if not sub:
        return await _render(lib, chat_id)
    if sub == "use" and len(rest) == 1:
        snap = await registry.set_active_profile(rest[0])
        return f"✅ active profile: {snap.profile}"
    if sub == "tier" and len(rest) == 2:
        if rest[0] not in _TIERS:
            raise ValueError(f"tier must be one of {', '.join(_TIERS)}")
        profile = registry.current_config().active_profile
        ov = registry.library_overrides(profile)
        await registry.set_overrides(profile, ov.model_copy(
            update={"tiers": {**ov.tiers, rest[0]: _spec(rest[1])}}))
        return f"✅ {profile}: tier {rest[0]} → {rest[1]}"
    if sub in ("set", "fallback") and len(rest) == 2:
        role, spec = rest[0], _spec(rest[1])
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        cfg = registry.current_config()
        profile = cfg.active_profile
        b = registry.configured_binding(cfg, profile, role)
        b = b.model_copy(update={"primary": spec} if sub == "set"
                         else {"fallbacks": [*b.fallbacks, spec]})
        ov = registry.library_overrides(profile)
        await registry.set_overrides(profile, ov.model_copy(update={"roles": {**ov.roles, role: b}}))
        return f"✅ {profile}: {role} {'→' if sub == 'set' else 'fallback +'} {rest[1]}"
    if sub == "reset" and len(rest) <= 1:
        profile = rest[0] if rest else registry.current_config().active_profile
        await registry.set_overrides(profile, ProfileOverrides())
        return f"✅ {profile}: overrides cleared"
    if sub == "models" and len(rest) == 1:
        ids = await asyncio.to_thread(registry.live_models, rest[0])
        return f"{rest[0]}: {len(ids)} model(s)\n" + "\n".join(ids)
    if sub == "transcriber" and len(rest) <= 2:
        return await _transcriber(rest, lib)
    return USAGE


_ASR_ENGINES = ("faster_whisper", "openai_audio")


async def _transcriber(rest: list[str], lib) -> str:
    """Show, or switch the ASR engine (and openai_audio's provider/model),
    persisted under ``engines.transcriber``. The block is checked for use
    here only to warn: a transcription is what fails on it."""
    try:
        cfg = registry.current_config()
    except EngineConfigError:
        cfg = await registry.load_config(library=lib)     # raises the real problem, shown to the user
    t = cfg.transcriber
    if rest:
        engine = rest[0]
        if engine not in _ASR_ENGINES:
            raise ValueError(f"the ASR engine is faster_whisper or openai_audio, got {engine!r}")
        update: dict = {"asr_engine": engine}
        if len(rest) == 2:
            if engine == "faster_whisper":
                raise ValueError("faster_whisper takes no provider/model "
                                 "(its model is transcriber.faster_whisper.model in argus_engines.json)")
            spec = _spec(rest[1])
            if spec.provider not in cfg.providers:
                raise ValueError(f"provider {spec.provider!r} is not configured "
                                 f"(known: {', '.join(cfg.providers)})")
            update["openai_audio"] = {**t.openai_audio, "provider": spec.provider, "model": spec.model}
        t = t.model_copy(update=update)
        await registry.set_transcriber(t)
        cfg = registry.current_config()
    try:
        backend, problem = transcribers.asr_engine(t, cfg).backend, ""
    except EngineConfigError as e:
        backend, problem = transcribers.asr_label(t), f"\n⚠ {e} — transcriptions fail until it is fixed"
    head = "✅ transcriber" if rest else "🎙 Transcriber"
    return f"{head}: {backend}\nYouTube: {' → '.join(t.youtube) or '(no steps)'}{problem}"


def _fmt(spec: EngineSpec) -> str:
    if spec.model:
        return f"{spec.provider}/{spec.model}"
    if spec.model_from_preferences:
        return f"{spec.provider}/<{spec.model_from_preferences} preference>"
    return f"{spec.provider}/<choose a model>"


async def _render(lib, chat_id: int) -> str:
    try:
        cfg = registry.current_config()
    except EngineConfigError:
        cfg = await registry.load_config(library=lib)     # raises the real problem, shown to the user
    p = cfg.profiles[cfg.active_profile]
    lines = [f"⚙️ Engines — profile {p.name}" + (f": {p.description}" if p.description else "")]
    try:
        snap = await registry.active_snapshot()
        lines.append(f"Validation: ✅ resolved {snap.resolved_at}")
        rows = {r: (b.primary, b.fallbacks, b.on_exhausted) for r, b in snap.roles.items()}
    except EngineConfigError as e:
        snap = None
        lines.append(f"Validation: ❌ {e}")
        rows = {r: (b.primary, b.fallbacks, b.on_exhausted)
                for r, (_t, b) in registry.bindings(cfg, cfg.active_profile).items()}
    lines += [f"  {role}: {_fmt(primary)}{''.join(' → ' + _fmt(f) for f in fbs)} [{policy}]"
              for role, (primary, fbs, policy) in rows.items()]
    if snap is not None:
        lines += [f"⚠ {w}" for w in snap.warnings]
    last = await _last_run(lib, chat_id)
    if last is not None:
        lines.append(last[0])
    lines += _diversity(cfg, snap, last[1] if last else {})
    mtd = getattr(lib, "exa_usd_month_to_date", None) if lib is not None else None
    lines.append(f"Exa this month: ${await mtd():.2f}" if mtd else "Exa this month: n/a (not recorded yet)")
    lines.append("Profiles: " + ", ".join(cfg.profiles))
    return "\n".join(lines)


def _diversity(cfg: EnginesConfig, snap, last_summary: dict) -> list[str]:
    """The judge-diversity line, with the command that clears a ⚠."""
    writer = snap.roles["compose.section"].primary if snap else None
    judge = snap.roles["panel.grounding"].primary if snap else None
    bound_same = bool(writer and judge) and registry.family(writer.model) == registry.family(judge.model)
    served_same = last_summary.get("judge_writer_same_family") is True
    if not (bound_same or served_same):
        return ["judge diversity ok"] if snap or last_summary else []
    why = (f"judge {judge.model} and writer {writer.model} are one model family" if bound_same
           else "the last run's judge and writer were served by one model family")
    taken = {s.provider for s in (writer, judge) if s}
    keyed = [q for q in cfg.providers.values() if q.api_key_env and q.name not in taken]
    ready = [q for q in keyed if os.environ.get(q.api_key_env)]
    if ready:
        fix = f"/engines tier judge {ready[0].name}/<model> (list: /engines models {ready[0].name})"
    elif keyed:
        fix = (f"set {keyed[0].api_key_env} in .env and restart, then /engines tier judge "
               f"{keyed[0].name}/<model>")
    else:
        fix = "/engines tier judge <provider>/<model> with a provider of another family"
    return [f"judge diversity ⚠ — {why}; fix: {fix}"]


async def _last_run(lib, chat_id: int) -> tuple[str, dict] | None:
    """The newest run of this chat with a metadata.json: its served tally."""
    if lib is None:
        return None
    for run in await lib.list_runs(chat_id=chat_id, limit=10):
        folder = run.get("report_dir")
        meta_path = Path(folder) / "metadata.json" if folder else None
        if meta_path is None or not meta_path.is_file():
            continue
        try:
            meta = json.loads(await asyncio.to_thread(meta_path.read_text, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return f"Last run {run['run_id']}: metadata.json unreadable ({e})", {}
        summary = meta.get("engines") or registry.summarize_calls(meta.get("model_calls") or [])
        served: Counter = Counter()
        for row in summary["by_role"].values():
            served.update(row.get("served_models") or {})
        tally = ", ".join(f"{m} ×{n}" for m, n in served.most_common(5)) or "no served calls"
        rate = summary.get("reroute_rate")
        return f"Last run {run['run_id']}: {tally}" + (f" · reroute {rate:.0%}" if rate is not None else ""), summary
    return None
