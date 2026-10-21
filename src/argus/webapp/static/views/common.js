// Shared view helpers: escaping, labels, the problem panel, Telegram buttons, a confirm dialog.
import { ApiError } from "../api.js";

export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
export const tg = () => window.Telegram?.WebApp;
const tgAtLeast = (v) => Boolean(tg()?.isVersionAtLeast?.(v));

export const LENGTHS = { tldr: "TL;DR", short: "Short", medium: "Medium", long: "Long", lecture: "Lecture" };
export const RECENCY = { "": "Auto (the brief decides)", d: "Past day", w: "Past week", m: "Past month", y: "Past year", any: "Any date" };

export function when(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** The Exa upper bound for a depth (RunConfig.exa_cost_estimate_usd): searches at the search price plus
 *  /contents pages at the page price; dotted overrides of the two caps apply. null without prices. */
export function exaEstimate(presets, depth, overrides, prices) {
  if (!prices) return null;
  const exa = presets.presets[depth]?.search?.exa || {};
  const searches = overrides?.["search.exa.max_searches"] ?? exa.max_searches ?? 0;
  const pages = overrides?.["search.exa.max_contents_pages"] ?? exa.max_contents_pages ?? 0;
  return searches * prices.price_per_search_usd + pages * prices.price_per_page_usd;
}

export const usd = (x, digits = 3) => (x === null || x === undefined ? "n/a" : `$${Number(x).toFixed(digits)}`);

/** A run's status as a kit `.status` (shape + label, never colour alone). */
export function runStatus(run) {
  if (run.in_flight) return { s: "warning", label: `Running${run.stage ? ` · ${run.stage}` : ""}` };
  if (run.awaiting === "plan") return { s: "serious", label: "Plan waiting for you" };
  if (run.awaiting === "report") return { s: "serious", label: "Report waiting for you" };
  const s = run.status || "";
  if (s === "done") return { s: "good", label: "Done" };
  if (s === "error") return { s: "critical", label: "Error" };
  if (s === "cancelled") return { s: "idle", label: "Cancelled" };
  return { s: "idle", label: s.replace(/_/g, " ") || "unknown" };
}
export const statusHtml = (run) => { const st = runStatus(run); return `<span class="status" data-s="${st.s}">${esc(st.label)}</span>`; };

const PROBLEMS = {
  expired: ["Session", "Session expired"],
  missing_field: ["Sign-in", "Open the console from Telegram"],
  hash_mismatch: ["Sign-in", "This sign-in could not be verified"],
  wrong_user: ["Sign-in", "This console belongs to someone else"],
  busy: ["Busy", "Another run is executing"],
  already_running: ["Already running", "This run is being driven right now"],
  refused: ["Refused", "The bot refused this action"],
  not_found: ["Not found", "Nothing here"],
  engine_config: ["Engines", "The engines configuration refused this"],
  bad_request: ["Request", "The console sent something the server refused"],
  send_failed: ["Send to chat", "Sending to the chat failed"],
  internal: ["Server", "The server hit an error"],
};

/** The explicit panel for an API failure: what went wrong, the server's own message. */
export function problemHtml(err, { dismiss = true } = {}) {
  const [kicker, title] = PROBLEMS[err.code] || [`Error ${err.status ?? ""}`.trim(), err.code || "Request failed"];
  return `<aside class="callout problem" data-kind="trap" role="alert"><h4>${esc(kicker)}<i>${esc(title)}</i></h4>`
    + `<p>${esc(err.message)}</p><p class="meta">${esc(err.code)}${err.status ? ` · HTTP ${esc(err.status)}` : ""}</p>`
    + (dismiss ? '<div class="btn-row"><button class="btn btn-s" type="button" data-dismiss>Dismiss</button></div>' : "")
    + "</aside>";
}

/** Show `err` in `slot` (an action's inline problem panel). Auth failures are the app's (it swaps the view). */
export function showProblem(slot, err) {
  if (!(err instanceof ApiError)) throw err;
  if (err.status === 401 || err.status === 403) {
    window.dispatchEvent(new CustomEvent("argus:auth", { detail: err }));
    return;
  }
  slot.innerHTML = problemHtml(err);
  slot.hidden = false;
  $("[data-dismiss]", slot).addEventListener("click", () => { slot.hidden = true; slot.innerHTML = ""; });
  slot.scrollIntoView({ block: "nearest" });
}

/** Run an action: disables `button` meanwhile; a failure goes to `slot`. Returns the result, or null. */
export async function act(button, slot, fn) {
  if (button) button.disabled = true;
  slot.hidden = true;
  try {
    return await fn();
  } catch (e) {
    showProblem(slot, e);
    return null;
  } finally {
    if (button) button.disabled = false;
  }
}

export const toast = (message, kind = "cost", title) => window.UI.toast(message, { kind, title });

// ---- Telegram buttons (guarded: the SDK may be absent, e.g. in a local preview) -------------------
let mainHandler = null;
export function mainButton(text, handler) {
  clearMainButton();
  const mb = tg()?.MainButton;
  if (!mb || !tgAtLeast("6.0")) return;
  mainHandler = handler;
  mb.setText(text);
  mb.onClick(mainHandler);
  mb.show();
}
export function clearMainButton() {
  const mb = tg()?.MainButton;
  if (mb && mainHandler) { mb.offClick(mainHandler); mb.hide(); }
  mainHandler = null;
}
export function backButton(visible, handler) {
  const bb = tg()?.BackButton;
  if (!bb || !tgAtLeast("6.1")) return;
  bb.offClick(handler);
  if (visible) { bb.onClick(handler); bb.show(); } else bb.hide();
}
/** Links open through Telegram when it is there (external browser), else in a new tab. */
export function openLink(href) {
  if (tg()?.openLink) tg().openLink(href); else window.open(href, "_blank", "noopener,noreferrer");
}

// ---- confirm dialog (the kit's .dlg; no window.confirm) -------------------------------------------
export function confirmDialog({ title, body, ok, danger = true }) {
  const d = document.createElement("dialog");
  d.className = `dlg${danger ? " danger" : ""}`;
  d.innerHTML = `<form method="dialog"><div class="dlg-h"><span class="kicker">Confirm</span><h2>${esc(title)}</h2></div>`
    + `<div class="dlg-b"><p>${esc(body)}</p></div><div class="dlg-f"><button class="btn" value="cancel">Keep it</button>`
    + `<button class="btn ${danger ? "btn-danger" : "btn-a"}" value="ok">${esc(ok)}</button></div></form>`;
  document.body.append(d);
  window.UI.init(d);
  return new Promise((resolve) => {
    d.addEventListener("close", () => { resolve(d.returnValue === "ok"); d.remove(); }, { once: true });
    d.showModal();
  });
}
