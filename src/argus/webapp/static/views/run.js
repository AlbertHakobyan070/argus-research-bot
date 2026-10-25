// #/run/<id> — one run: progress (events polled every 2 s while a drive runs, the snapshot re-fetched
// on stage end), the plan gate, the report gate, the report reader with tap-to-verify, Send to chat.
import { get, post } from "../api.js";
import {
  $, $$, act, clearMainButton, confirmDialog, esc, exaEstimate, LENGTHS, mainButton, openLink, RECENCY,
  showProblem, statusHtml, toast, usd, when,
} from "./common.js";

const POLL_MS = 2000;
const GRACE_TICKS = 3;          // after an action, poll this long for the new drive to show up
const SHOWN_EVENTS = 30;
const MAX_EVENTS = 500;          // the bus keeps as many per run

export async function render(el, [id], ctx) {
  ctx.crumbs([["#/runs", "Runs"], ["", id]]);
  const [data, presets] = await Promise.all([get.run(id), get.presets()]);
  const s = { id, data, presets, events: [], lastSeq: 0, between: false, grace: 0, ticking: false,
              timer: null, stopped: false, reportKey: null, evidence: null };
  await moreEvents(s);
  el.innerHTML = `<div id="run-head"></div><div id="notice" hidden></div>
    <section class="sec" id="gate" hidden></section><section class="sec" id="report" hidden></section>
    <section class="sec" id="progress"></section>`;
  s.el = el;
  s.notice = $("#notice", el);
  paint(s);
  schedule(s);
  return () => { s.stopped = true; clearTimeout(s.timer); clearMainButton(); hidePop(); };
}

// ---- polling -------------------------------------------------------------------------------------
// One chain at a time: a tick reschedules itself when it ends, so schedule() during a tick does nothing.
function schedule(s) {
  if (s.stopped || s.ticking) return;
  clearTimeout(s.timer);
  if (s.data.run.in_flight || s.between || s.grace > 0) s.timer = setTimeout(() => tick(s), POLL_MS);
}

/** The events after s.lastSeq: the server answers in pages, so read on until its last_seq. */
async function moreEvents(s) {
  const fresh = [];
  for (;;) {
    const r = await get.events(s.id, s.lastSeq);
    if (!r.events.length) break;
    fresh.push(...r.events);
    s.lastSeq = r.events[r.events.length - 1].seq;
    if (s.lastSeq >= r.last_seq) break;
  }
  s.events = s.events.concat(fresh).slice(-MAX_EVENTS);
  return fresh;
}

const ends = (e) => (e.kind === "stage" && e.data?.status !== "start") || e.kind === "gate";

/** One poll: new events; the snapshot is re-read when any of them ends a step (a stage end, error,
 *  cancel or drain, or a gate), and on each later tick until the next stage starts or the drive is
 *  over (the last stage end can come before the driver releases the run). */
export async function tick(s) {
  if (s.stopped || s.ticking) return;    // one chain at a time
  s.ticking = true;
  try {
    const grace = s.grace > 0;            // after an action: re-read until the new drive shows up
    if (grace) s.grace -= 1;
    const fresh = await moreEvents(s);
    if (s.stopped) return;
    const last = fresh.filter((e) => e.kind === "stage" || e.kind === "gate").pop();
    if (last) s.between = ends(last);
    if (fresh.some(ends) || s.between || grace) {
      const data = await get.run(s.id);
      if (s.stopped) return;
      s.data = data;
      if (!data.run.in_flight) s.between = false;
      else s.grace = 0;
      paint(s);
    } else if (fresh.length) paintProgress(s);
  } catch (e) {
    showProblem(s.notice, e);              // polling stops; the top bar refresh starts it again
    return;
  } finally {
    s.ticking = false;
  }
  schedule(s);
}

/** After a successful action: re-read the run. A failure is shown, never an unhandled rejection; polling
 *  stops, as a failed tick does, and the top bar refresh starts it again. */
export async function refresh(s) {
  try {
    const data = await get.run(s.id);
    if (s.stopped) return;
    s.data = data;
    s.grace = GRACE_TICKS;
    paint(s);
    schedule(s);
  } catch (e) {
    showProblem(s.notice, e);
  }
}

// ---- painting ------------------------------------------------------------------------------------
function paint(s) {
  paintHead(s);
  paintProgress(s);
  paintGate(s);
  paintReport(s);
  window.UI.init(s.el);
}

function paintHead(s) {
  const { run } = s.data;
  const canCancel = run.in_flight || run.awaiting;
  const facts = [LENGTHS[run.length] || run.length, run.depth, run.archetype, `origin ${run.origin || "—"}`,
                 `started ${when(run.created_at)}`].filter(Boolean);
  $("#run-head", s.el).innerHTML = `<div class="page-head">
      <div><span class="kicker">Run ${esc(run.run_id)}</span><h1>${esc(run.topic || run.run_id)}</h1>
        <p>${statusHtml(run)} <span class="meta head-facts">${esc(facts.join(" · "))}</span></p></div>
      ${canCancel ? '<div class="btn-row"><button class="btn btn-danger" type="button" id="cancel-run">Cancel run…</button></div>' : ""}
    </div>`;
  const cancel = $("#cancel-run", s.el);
  if (cancel) cancel.addEventListener("click", () => cancelRun(s, cancel));
}

async function cancelRun(s, button) {
  const live = s.data.run.in_flight;
  const ok = await confirmDialog({
    title: `Cancel run ${s.id}?`,
    body: live ? "The run finishes its current step, keeps everything it fetched, and stops. /continue resumes it."
      : "The run leaves its gate and stops. /continue resumes it.",
    ok: "Cancel run",
  });
  if (!ok) return;
  const r = await act(button, s.notice, () => post.cancel(s.id));
  if (r) { toast(r.message || `Run ${s.id}: ${r.outcome}.`, "why", "Cancel"); await refresh(s); }
}

const count = (v) => (Array.isArray(v) ? v.length : v ?? "?");
/** A link only for http(s) URLs (search results are untrusted); anything else is plain text. */
const link = (url, text) => (/^https?:\/\//i.test(url || "") ? `<a href="${esc(url)}" data-ext>${esc(text)}</a>` : esc(text));
const host = (url) => { try { return new URL(url).host; } catch { return url || "?"; } };

function describe(e) {
  const d = e.data || {};
  switch (e.kind) {
    case "stage": return [["error", "cancelled", "draining", "paused"].includes(d.status) ? "trap" : "",
      `${d.status}${d.summary ? ` — ${d.summary}` : ""}`];
    case "search": return ["", `search wave ${d.wave} (${d.mode}): ${count(d.queries)} queries, ${count(d.hits)} hits`
      + (d.errors?.length ? `, ${d.errors.length} errors` : "")];
    case "fetch": return [d.ok ? "" : "trap", `${d.ok ? "fetched" : "failed"} ${host(d.url)} · ${d.fetcher || "?"} · ${d.ms ?? "?"} ms${d.pinned ? " · pinned" : ""}`];
    case "digest": return [d.ok ? "" : "trap", `digest source ${d.source_id}: ${d.ok ? `relevance ${d.relevance}` : "failed"}${d.cached ? " (cached)" : ""}`];
    case "rerank": return ["", `rerank wave ${d.wave}: kept ${count(d.kept)} of ${count(d.candidates)}`];
    case "section": return [d.ok ? "" : "trap", `section “${d.title}” ${d.ok ? `${d.chars} chars` : "failed"}${d.reused ? " (reused)" : ""}`];
    case "analysis": return ["", `analysis: ${count(d.takeaways)} takeaways${d.has_disagreement ? " · disagreement" : ""}${d.has_scenarios ? " · scenarios" : ""}`];
    case "judge": return [d.skipped ? "trap" : "", `judge ${d.name} batch ${d.batch}: ${d.skipped ? "skipped" : d.verdict}`];
    case "degrade": return ["trap", `degraded ${d.role}: ${d.degradation}${d.reason ? ` — ${d.reason}` : ""}`];
    case "gate": return ["", `${d.kind} gate acted on from ${d.actor}`];
    default: return ["", `${e.kind} ${JSON.stringify(d)}`];
  }
}

function paintProgress(s) {
  const { run, snapshot } = s.data;
  const sec = $("#progress", s.el);
  const c = snapshot.counts || {};
  const kpi = (k, v, sub = "") => `<div class="kpi"><span class="kpi-k">${esc(k)}</span><span class="kpi-v">${esc(v)}</span>${sub ? `<span class="kpi-s">${esc(sub)}</span>` : ""}</div>`;
  const kpis = `<div class="kpis">${kpi("Sources", c.sources ?? 0, "found")}${kpi("Fetched", c.fetched ?? 0)}`
    + `${kpi("Evidence", c.evidence ?? 0, "digested notes")}${kpi("Findings", c.findings ?? 0)}`
    + `${kpi("Gap waves", snapshot.gap_waves ?? 0)}${kpi("Exa", usd(run.exa_usd), "this run")}</div>`;
  const shown = s.events.slice(-SHOWN_EVENTS).reverse();
  const rows = shown.map((e) => {
    const [kind, text] = describe(e);
    return `<tr${kind ? ` data-kind="${kind}"` : ""}><td class="meta">${esc(new Date(e.ts * 1000).toLocaleTimeString())}</td>`
      + `<td><code>${esc(e.node || e.kind)}</code></td><td>${esc(text)}</td></tr>`;
  }).join("");
  const head = run.in_flight ? `Running — at ${run.stage || "start"}` : s.events.length ? "The last drive" : "Progress";
  const lede = run.in_flight ? "Updates every two seconds while the run works."
    : s.events.length ? "Events of this bot session (the log is kept for an hour after a drive)." : "No events in this bot session.";
  sec.innerHTML = `<div class="sec-head"><span class="kicker">Progress</span><h2>${esc(head)}</h2><p>${esc(lede)}</p></div>${kpis}`
    + (rows ? `<div class="tw timeline"><table><thead><tr><th>Time</th><th>Node</th><th>Event</th></tr></thead><tbody>${rows}</tbody></table></div>`
      + (s.events.length > SHOWN_EVENTS ? `<p class="meta">The newest ${SHOWN_EVENTS} of ${s.events.length} events.</p>` : "") : "");
}

// ---- the plan gate -------------------------------------------------------------------------------
function paintGate(s) {
  const sec = $("#gate", s.el);
  const { run, snapshot } = s.data;
  sec.hidden = run.awaiting !== "plan";
  if (sec.hidden) { if (sec.dataset.painted) { sec.innerHTML = ""; delete sec.dataset.painted; clearMainButton(); } return; }
  if (sec.dataset.painted === String(s.lastSeq)) return;          // keep the user's knob choices between polls
  sec.dataset.painted = String(s.lastSeq);
  const p = snapshot.plan;
  const head = `<div class="sec-head"><span class="kicker">Plan gate</span><h2>Review the research plan</h2>
    <p>Approve starts the research. Edit redrafts the plan from your notes. The chat card follows what you do here.</p></div>`;
  if (!p) {
    sec.innerHTML = `${head}<aside class="callout" data-kind="mech"><h4>mech<i>Plan not available</i></h4>
      <p>This run's checkpoint carries no plan data the console can show (for example a checkpoint from before v4). You can still approve, edit or cancel it.</p></aside>${planActions()}`;
  } else {
    sec.innerHTML = head + planBody(p) + planKnobs(s, p) + planActions();
    wireKnobs(s, p);
  }
  wirePlanActions(s, sec);
}

function planBody(p) {
  const plan = p.plan || {};
  const subs = plan.sub_questions || [];
  const intents = plan.planned_sources || [];
  const sources = p.sources || [];
  const shown = sources.slice(0, 12);
  return `${p.error ? `<aside class="callout" data-kind="trap"><h4>trap<i>The plan's settings do not fit</i></h4><p>${esc(p.error)}</p></aside>` : ""}
    ${plan.summary ? `<p class="lede plan-summary">${esc(plan.summary)}</p>` : ""}
    <div class="plan-grid">
      ${subs.length ? `<div><h3>Sub-questions</h3><ol>${subs.map((q) => `<li>${esc(q)}</li>`).join("")}</ol></div>` : ""}
      ${intents.length ? `<div><h3>Search intents</h3><ul>${intents.map((i) => `<li><code>${esc(i.kind || "search")}</code> ${esc(i.query || "live search")}</li>`).join("")}</ul></div>` : ""}
      <div><h3>Found sources</h3>${sources.length
        ? `<ol>${shown.map((x) => `<li>${link(x.url, (x.title || "").trim() || host(x.url))} <span class="meta">${esc(host(x.url))}</span></li>`).join("")}</ol>`
          + (sources.length > shown.length ? `<p class="meta">…and ${sources.length - shown.length} more.</p>` : "")
        : `<aside class="callout" data-kind="trap"><h4>trap<i>Live search found no sources</i></h4><p>Approving would fetch nothing. Edit the plan or cancel it.</p></aside>`}</div>
    </div>`;
}

function planKnobs(s, p) {
  const depths = Object.keys(s.presets.presets);
  const radios = (name, opts, current) => `<div class="seg seg-s" role="radiogroup" aria-label="${esc(name)}">${opts.map(([v, label]) => `<label><input type="radio" name="${esc(name)}" value="${esc(v)}"${v === current ? " checked" : ""}><span>${esc(label)}</span></label>`).join("")}</div>`;
  const recency = p.recency_window ? `${RECENCY[p.recency_window.window] || p.recency_window.window} (since ${p.recency_window.start_date})` : "no filter";
  return `<form class="set-panel plan-knobs" id="knobs">
      <div class="set-row"><span class="set-l"><b>Length</b><small>How long the report is.</small></span>
        ${radios("length", Object.entries(LENGTHS), p.length)}</div>
      <div class="set-row"><span class="set-l"><b>Depth</b><small id="depth-auto">Auto follows the length.</small></span>
        ${radios("depth", [["auto", "Auto"], ...depths.map((d) => [d, d[0].toUpperCase() + d.slice(1)])], p.depth_auto ? "auto" : p.depth)}</div>
      <div class="set-row"><label class="set-l" for="profile"><b>Engines profile</b><small>Which models run each role.</small>
        ${p.profile_problem ? `<small class="err">${esc(p.profile_problem)}</small>` : ""}</label>
        <select class="input" id="profile">${(p.profiles || []).map((n) => `<option${n === p.profile ? " selected" : ""}>${esc(n)}</option>`).join("")}</select></div>
      <div class="set-row"><span class="set-l"><b>Recency</b><small>Set by the brief or the chat card's recency button.</small></span>
        <span class="meta">${esc(p.recency === null || p.recency === undefined ? `auto — ${recency}` : recency)}</span></div>
      <div class="set-row"><span class="set-l"><b>Exa estimate</b><small>Upper bound for these settings${p.exa_enabled ? "" : " — Exa is off for this run"}.</small></span>
        <span class="mono" id="estimate"></span></div>
    </form>`;
}

function choices(s, p) {
  const form = $("#knobs", s.el);
  if (!form) return {};
  const pick = (name) => form.elements.namedItem(name).value;      // a RadioNodeList's checked value
  return { length: pick("length") || p.length, depth: pick("depth"), engines_profile: $("#profile", form).value };
}

function wireKnobs(s, p) {
  const form = $("#knobs", s.el);
  const update = () => {
    const c = choices(s, p);
    const depth = c.depth === "auto" ? s.presets.default_depth[c.length] : c.depth;
    $("#depth-auto", form).textContent = `Auto follows the length (${s.presets.default_depth[c.length]} for ${LENGTHS[c.length]}).`;
    const unchanged = c.length === p.length && depth === p.depth;
    const est = !p.exa_enabled ? null : unchanged ? p.exa_estimate_usd
      : exaEstimate(s.presets, depth, s.data.snapshot.run_config?.overrides_applied, s.presets.exa_prices);
    $("#estimate", form).textContent = p.exa_enabled ? (est === null ? "n/a" : `≈ ${usd(est)}`) : "off";
  };
  form.addEventListener("change", update);
  update();
}

function planActions() {
  return `<div class="gate-actions">
      <div class="btn-row"><button class="btn btn-a" type="button" data-plan="approve"><i data-icon="check"></i>Approve plan</button>
        <button class="btn" type="button" data-plan="edit-open"><i data-icon="pencil"></i>Edit…</button>
        <button class="btn btn-danger" type="button" data-plan="cancel">Cancel run…</button></div>
      <div class="field edit-box" hidden><label for="plan-feedback">What should change in the plan?</label>
        <textarea class="input" id="plan-feedback" placeholder="e.g. focus on 2026 data and add the EU angle"></textarea>
        <div class="btn-row"><button class="btn btn-a" type="button" data-plan="edit">Redraft the plan</button></div></div>
    </div>`;
}

function wirePlanActions(s, sec) {
  const approve = $('[data-plan="approve"]', sec);
  const doApprove = async () => {
    const body = { action: "approve", ...choices(s, s.data.snapshot.plan || {}) };
    if (!s.data.snapshot.plan) { delete body.length; delete body.depth; delete body.engines_profile; }
    const r = await act(approve, s.notice, () => post.plan(s.id, body));
    if (r) { toast(r.message || "Approved — the research is running.", "cost", "Plan approved"); await refresh(s); }
  };
  approve.addEventListener("click", doApprove);
  mainButton("Approve plan", doApprove);
  $('[data-plan="edit-open"]', sec).addEventListener("click", () => {
    const box = $(".edit-box", sec);
    box.hidden = !box.hidden;
    if (!box.hidden) $("#plan-feedback", sec).focus();
  });
  const edit = $('[data-plan="edit"]', sec);
  edit.addEventListener("click", async () => {
    const area = $("#plan-feedback", sec);
    const feedback = area.value.trim();
    area.setAttribute("aria-invalid", String(!feedback));
    if (!feedback) { area.focus(); return; }
    const r = await act(edit, s.notice, () => post.plan(s.id, { action: "edit", feedback }));
    if (r) { toast(r.message || "Redrafting the plan.", "cost", "Plan edit"); await refresh(s); }
  });
  $('[data-plan="cancel"]', sec).addEventListener("click", (e) => cancelRun(s, e.currentTarget));
}

// ---- the report ----------------------------------------------------------------------------------
function paintReport(s) {
  const sec = $("#report", s.el);
  const { run, snapshot } = s.data;
  const paths = snapshot.report_paths || {};
  sec.hidden = !paths.md;
  if (sec.hidden) return;
  const key = `${paths.md}|${snapshot.compose_runs}|${run.awaiting}|${run.in_flight}`;
  if (s.reportKey === key) return;
  s.reportKey = key;
  const files = ["md", "pdf", "evidence"].filter((k) => paths[k]);
  const atGate = run.awaiting === "report";
  sec.innerHTML = `<div class="sec-head"><span class="kicker">Report</span><h2>${atGate ? "The report is ready for review" : "The report"}</h2>
      <p>Tap a citation <span class="cite cite-demo">[n]</span> to see its source and the quotes behind it.</p></div>
    <div id="report-check"></div>
    <div class="set-panel send-panel"><div class="set-row"><span class="set-l"><b>Send to chat</b><small>The bot sends the files to your Telegram chat.</small></span>
      <div class="row">${files.map((k) => `<label class="check"><input type="checkbox" name="send" value="${k}"${k !== "evidence" ? " checked" : ""}>${k === "md" ? "Markdown" : k === "pdf" ? "PDF" : "Evidence"}</label>`).join("")}
      <button class="btn" type="button" id="send-chat"><i data-icon="share-2"></i>Send to chat</button></div></div></div>
    ${atGate ? reportActions() : ""}
    <article class="prose report" id="report-body"><span class="skel h"></span><span class="skel w80"></span><span class="skel w60"></span></article>`;
  const send = $("#send-chat", sec);
  send.addEventListener("click", async () => {
    const what = $$('input[name="send"]:checked', sec).map((i) => i.value);
    if (!what.length) { toast("Choose at least one file to send.", "trap", "Send to chat"); return; }
    const r = await act(send, s.notice, () => post.send(s.id, what));
    if (r) toast(r.message, "cost", "Sent to chat");
  });
  if (atGate) wireReportActions(s, sec);
  loadReport(s, $("#report-body", sec), $("#report-check", sec));
}

// The report's own trust banners (report_builder's title block: "> ⚠ Not fact-checked — …", "> ⚠ Degraded run — …").
export const BANNER = /^⚠ (Not fact-checked|Degraded run) — /;

/** The report header's fact-check line, from the report's banners (`quoted`: the text of each paragraph of
 *  the report's top-level quotes; the `BANNER` ones count) and its evidence.json, read together with the
 *  report: any banner, or an evidence.json that marks the run degraded, is the trap callout; only an
 *  evidence.json that marks the run not degraded, with every finding judged, earns "Fact-checked". No
 *  evidence (an older run, or one that failed to load) or no finding judged: no claim either way. */
export function trustHeader(quoted, ev) {
  const banners = quoted.filter((t) => BANNER.test(t));
  const ok = ev && !ev.missing && !ev.error;
  if (banners.length || (ok && ev.degraded === true)) {
    const lines = banners.length ? banners.map((b) => `<p>${esc(b.replace(/^⚠ /, ""))}</p>`).join("")
      : "<p>evidence.json marks this run degraded; the report's notes say why.</p>";
    return `<aside class="callout" data-kind="trap"><h4>trap<i>Not a fully fact-checked report</i></h4>${lines}</aside>`;
  }
  const pv = ok && ev.degraded === false ? ev.panel || {} : {};
  return pv.total && pv.judged === pv.total
    ? `<p class="meta">Fact-checked: all ${esc(pv.total)} findings judged · panel verdict ${esc(pv.verdict)}.</p>` : "";
}

function reportActions() {
  return `<div class="gate-actions"><div class="btn-row">
      <button class="btn btn-a" type="button" data-report="send"><i data-icon="check"></i>Deliver report</button>
      <button class="btn" type="button" data-report="extend"><i data-icon="scan-search"></i>Extend</button>
      <button class="btn" type="button" data-report="revise-open"><i data-icon="pencil"></i>Revise…</button>
      <button class="btn btn-danger" type="button" data-report="cancel">Cancel run…</button></div>
    <p class="hint">Deliver finishes the run. Extend researches more (pending /append sources included). Revise rewrites from your notes.</p>
    <div class="field revise-box" hidden><label for="revise-feedback">What should change in the report?</label>
      <textarea class="input" id="revise-feedback"></textarea>
      <div class="btn-row"><button class="btn btn-a" type="button" data-report="revise">Revise the report</button></div></div></div>`;
}

function wireReportActions(s, sec) {
  const run = async (button, body, title) => {
    const r = await act(button, s.notice, () => post.report(s.id, body));
    if (r) { toast(r.message || `${title}: ${r.outcome}.`, "cost", title); await refresh(s); }
  };
  const deliver = $('[data-report="send"]', sec);
  deliver.addEventListener("click", () => run(deliver, { action: "send" }, "Report delivered"));
  const extend = $('[data-report="extend"]', sec);
  extend.addEventListener("click", () => run(extend, { action: "extend" }, "Extend"));
  $('[data-report="revise-open"]', sec).addEventListener("click", () => {
    const box = $(".revise-box", sec);
    box.hidden = !box.hidden;
    if (!box.hidden) $("#revise-feedback", sec).focus();
  });
  const revise = $('[data-report="revise"]', sec);
  revise.addEventListener("click", () => {
    const area = $("#revise-feedback", sec);
    const feedback = area.value.trim();
    area.setAttribute("aria-invalid", String(!feedback));
    if (!feedback) { area.focus(); return; }
    run(revise, { action: "revise", feedback }, "Revise");
  });
  $('[data-report="cancel"]', sec).addEventListener("click", (e) => cancelRun(s, e.currentTarget));
}

async function loadReport(s, body, check) {
  const hasEvidence = Boolean(s.data.snapshot.report_paths?.evidence);   // older runs have none: not requested
  const [html, evidence] = await Promise.allSettled([get.report(s.id), hasEvidence ? get.evidence(s.id) : null]);
  if (s.stopped) return;
  if (html.status === "rejected") { body.innerHTML = ""; showProblem(body, html.reason); body.hidden = false; return; }
  body.innerHTML = html.value;               // sanitised server-side (allowlist); CSP blocks any script
  if (evidence.status === "rejected") {
    s.evidence = { error: evidence.reason };
    showProblem(s.notice, evidence.reason);  // the popovers say why too
  } else s.evidence = evidence.value || { missing: true };
  // the header says what the report's own banners say; nothing is claimed before both have loaded
  check.innerHTML = trustHeader($$(":scope > blockquote > p", body).map((p) => p.textContent.trim()), s.evidence);
  $$(".cite[data-n]", body).forEach((c) => {
    c.tabIndex = 0;
    c.setAttribute("role", "button");
    c.setAttribute("aria-label", `Source ${c.dataset.n}`);
  });
  body.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite[data-n]");
    if (cite) { showCite(s, cite); return; }
    const a = e.target.closest("a[href]");
    if (a && /^https?:/.test(a.getAttribute("href"))) { e.preventDefault(); openLink(a.href); }
  });
  body.addEventListener("keydown", (e) => {
    const cite = e.target.closest(".cite[data-n]");
    if (cite && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); showCite(s, cite); }
  });
}

// ---- tap-to-verify -------------------------------------------------------------------------------
const pop = () => document.getElementById("cite-pop");
function hidePop() { const p = pop(); if (p.matches(":popover-open")) p.hidePopover(); }

function showCite(s, cite) {
  const n = Number(cite.dataset.n);
  const ev = s.evidence || {};
  let inner;
  if (ev.missing || ev.error) {
    inner = `<span class="kicker">Source [${n}]</span><p>${esc(ev.missing
      ? "This run has no evidence file, so its sources cannot be shown here."
      : `The evidence file could not be read: ${ev.error.message}`)}</p>`;
  } else {
    const src = (ev.sources || []).find((x) => x.id === n);
    if (!src) {
      inner = `<span class="kicker">Source [${n}]</span><p>Source ${n} is not in this run's evidence.</p>`;
    } else {
      const claims = (ev.notes || []).filter((x) => x.source_id === n).flatMap((x) => x.claims || []).filter((c) => c.quote);
      const facts = [src.domain, src.published && `published ${String(src.published).slice(0, 10)}`, src.source_type,
                     src.pinned && "pinned by you"].filter(Boolean).join(" · ");
      inner = `<span class="kicker">Source [${n}]</span><h4>${esc(src.title || src.domain || src.url)}</h4>
        <p class="meta">${esc(facts)}</p>
        ${claims.slice(0, 3).map((c) => `<blockquote class="cite-quote"><p>${esc(c.quote)}</p>
          <span class="tag" data-kind="${c.quote_verified ? "cost" : "trap"}">${c.quote_verified ? "quote verified in the page" : "quote not verified"}</span></blockquote>`).join("")
          || '<p class="faint">No quoted claims were kept from this source.</p>'}
        ${/^https?:\/\//i.test(src.url || "") ? `<p><a href="${esc(src.url)}" data-ext>Open the source <i data-icon="external-link"></i></a></p>` : ""}`;
    }
  }
  const p = pop();
  p.innerHTML = inner;
  window.UI.init(p);
  const ext = $("a[data-ext]", p);
  if (ext) ext.addEventListener("click", (e) => { e.preventDefault(); openLink(ext.href); });
  if (!p.matches(":popover-open")) p.showPopover();
  window.UI.place(p, cite, "bottom", "start");
}

document.addEventListener("click", (e) => {        // plan-source links open through Telegram
  const a = e.target.closest("#gate a[data-ext]");
  if (a) { e.preventDefault(); openLink(a.href); }
});
