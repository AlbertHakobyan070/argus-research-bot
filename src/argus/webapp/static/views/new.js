// #/new — start a research run (POST /api/runs): topic, length, depth, recency, engines profile,
// dotted RunConfig overrides, and the Exa upper-bound estimate for those settings.
import { ApiError, get, post } from "../api.js";
import { $, act, esc, exaEstimate, LENGTHS, mainButton, RECENCY, toast, usd } from "./common.js";

const DEFAULT_LENGTH = "short";

/** "research.waves=2" lines → {"research.waves": 2}; a value is JSON when it parses, else a string. */
export function parseOverrides(text) {
  const out = {};
  for (const [i, raw] of text.split("\n").entries()) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 1) throw new Error(`line ${i + 1}: expected dotted.key=value, got “${line}”`);
    const value = line.slice(eq + 1).trim();
    let parsed;
    try { parsed = JSON.parse(value); } catch { parsed = value; }
    out[line.slice(0, eq).trim()] = parsed;
  }
  return out;
}

export async function render(el, _params, ctx) {
  ctx.crumbs([["", "New research"]]);
  const [presets, engines] = await Promise.all([get.presets(), get.engines().catch((e) => {
    if (e instanceof ApiError && e.code === "engine_config") return { problem: e.message };
    throw e;
  })]);
  const profiles = engines.config ? Object.keys(engines.config.profiles) : [];
  const depths = Object.keys(presets.presets);
  el.innerHTML = `<div class="page-head"><div><span class="kicker">New research</span><h1>Start a research run</h1>
      <p>The bot drafts a plan from live search first; nothing is fetched until you approve it. Progress shows here and in the chat.</p></div></div>
    <div id="notice" hidden></div>
    <section class="sec set-sec" id="new-run">
      <div class="set-intro"><h2>The question</h2><p>Ask it the way you would in the chat with <code>/research</code>. Length and depth can still change at the plan gate.</p></div>
      <form class="set-panel" id="new-form" novalidate>
        <div class="fields">
          <div class="field"><label for="topic">Topic</label>
            <textarea class="input" id="topic" name="topic" required placeholder="What should Argus research?"></textarea>
            <p class="err" id="topic-err" hidden>Write the topic first.</p></div>
        </div>
        <div class="set-row"><span class="set-l"><b>Length</b><small>How long the report is.</small></span>
          <div class="seg seg-s" role="radiogroup" aria-label="Length">${Object.entries(LENGTHS).map(([v, label]) => `<label><input type="radio" name="length" value="${v}"${v === DEFAULT_LENGTH ? " checked" : ""}><span>${esc(label)}</span></label>`).join("")}</div></div>
        <div class="set-row"><label class="set-l" for="depth"><b>Depth</b><small>How much it searches and fetches.</small></label>
          <select class="input" id="depth" name="depth"><option value="">Auto</option>${depths.map((d) => `<option value="${esc(d)}">${esc(d)}</option>`).join("")}</select></div>
        <div class="set-row"><label class="set-l" for="recency"><b>Recency</b><small>Limit sources to a recent window.</small></label>
          <select class="input" id="recency" name="recency">${Object.entries(RECENCY).map(([v, label]) => `<option value="${v}">${esc(label)}</option>`).join("")}</select></div>
        <div class="set-row"><label class="set-l" for="profile"><b>Engines profile</b><small>${engines.problem ? `<span class="err">${esc(engines.problem)}</span>` : "Which models run each role."}</small></label>
          <select class="input" id="profile" name="profile"${profiles.length ? "" : " disabled"}>${profiles.map((n) => `<option${n === engines.active_profile ? " selected" : ""}>${esc(n)}</option>`).join("")}</select></div>
        <div class="fields">
          <div class="field"><label for="overrides">Overrides <span class="faint">(optional)</span></label>
            <textarea class="input mono" id="overrides" name="overrides" placeholder="research.waves=2&#10;search.exa.max_searches=8"></textarea>
            <p class="hint">One <code>dotted.key=value</code> per line on the run configuration (research, search, fetch, compose, panel, recency).</p>
            <p class="err" id="overrides-err" hidden></p></div>
        </div>
        <div class="set-panel-f"><span class="hint" id="estimate"></span><button class="btn btn-a" type="submit"><i data-icon="play"></i>Start research</button></div>
      </form>
    </section>`;
  const form = $("#new-form", el);
  const notice = $("#notice", el);
  const read = () => {
    const length = form.elements.namedItem("length").value;
    const depth = form.depth.value || presets.default_depth[length];
    let overrides = {};
    const err = $("#overrides-err", form);
    try {
      overrides = parseOverrides(form.overrides.value);
      err.hidden = true;
      form.overrides.removeAttribute("aria-invalid");
    } catch (e) {
      err.textContent = e.message;
      err.hidden = false;
      form.overrides.setAttribute("aria-invalid", "true");
      overrides = null;
    }
    return { length, depth, overrides };
  };
  const estimate = () => {
    const { length, depth, overrides } = read();
    const est = exaEstimate(presets, depth, overrides || {}, presets.exa_prices);
    $("#estimate", form).textContent = est === null ? "Exa estimate: n/a (no Exa prices configured)"
      : `Exa estimate ≈ ${usd(est)} upper bound (${LENGTHS[length]}, ${depth}${presets.exa_prices.prices_as_of ? `, prices as of ${presets.exa_prices.prices_as_of}` : ""}) — when Exa is enabled`;
  };
  form.addEventListener("input", estimate);
  form.addEventListener("change", estimate);
  estimate();
  const submit = $('button[type="submit"]', form);
  const start = async () => {
    const topic = form.topic.value.trim();
    $("#topic-err", form).hidden = Boolean(topic);
    form.topic.setAttribute("aria-invalid", String(!topic));
    const { length, overrides } = read();
    if (!topic) { form.topic.focus(); return; }
    if (overrides === null) { form.overrides.focus(); return; }
    const body = { topic, length };
    if (form.depth.value) body.depth = form.depth.value;
    if (form.recency.value) body.recency = form.recency.value;
    if (profiles.length && form.profile.value !== engines.active_profile) body.engines_profile = form.profile.value;
    if (Object.keys(overrides).length) body.overrides = overrides;
    const r = await act(submit, notice, () => post.start(body));
    if (r) {
      toast(r.message || "The plan is being drafted.", "cost", `Run ${r.run_id} started`);
      location.hash = `#/run/${r.run_id}`;
    }
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); start(); });
  mainButton("Start research", start);
}
