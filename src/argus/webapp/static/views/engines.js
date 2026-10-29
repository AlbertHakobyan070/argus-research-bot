// #/engines — the engines console (GET/PUT /api/engines), laid out as the kit's console.html: the
// active profile with its validation and warnings, per-profile tier and role overrides (live model
// lists, fallback lists), the transcriber, the month-to-date Exa spend, and the provider definitions
// (read-only: they are file-only). The server takes one change per PUT; each control sends one.
import { ApiError, get, putEngines } from "../api.js";
import { $, $$, act, esc, showProblem, toast, usd, when } from "./common.js";

const TIERS = ["cheap", "strong", "judge"];
const POLICIES = ["raise", "skip_item", "degrade"];
const clone = (v) => JSON.parse(JSON.stringify(v ?? {}));

export async function render(el, _params, ctx) {
  ctx.crumbs([["", "Engines"]]);
  const s = { el, body: await get.engines(), profile: null, role: null, lists: {} };
  s.profile = s.body.active_profile;
  paint(s);
}

function paint(s) {
  const { body } = s;
  const cfg = body.config;
  s.providers = Object.keys(cfg.providers);
  s.el.innerHTML = `<div class="page-head"><div><span class="kicker">Engines</span>
      <h1>${esc(body.validation.ok ? `${body.active_profile} is active and valid` : `${body.active_profile} is active but does not validate`)}</h1>
      <p>Changes apply to the next run. Provider definitions and prices live in <code>argus_engines.json</code>.</p></div></div>
    ${profilesSec(s)}${overridesSec(s)}${transcriberSec(s)}${spendSec(s)}${providersSec(s)}
    ${s.providers.map((p, i) => `<datalist id="models-${i}"></datalist>`).join("")}`;
  wire(s);
  window.UI.init(s.el);
}

/** One PUT (one change); the answer is the new GET body. */
async function save(s, button, slot, change, done) {
  const r = await act(button, slot, () => putEngines(change));
  if (r) { s.body = r; paint(s); toast(done, "cost", "Engines saved"); }
}

const specText = (spec) => (spec ? `${spec.provider} / ${spec.model || (spec.model_from_preferences ? `preferences: ${spec.model_from_preferences}` : "no model")}` : "—");
const intro = (title, text) => `<div class="set-intro"><h2>${esc(title)}</h2><p>${text}</p><div class="sec-notice" hidden></div></div>`;

// ---- profiles ------------------------------------------------------------------------------------
function profilesSec(s) {
  const { body } = s;
  const snap = body.snapshot;
  const errors = body.validation.errors || [];
  const warnings = snap?.warnings || [];
  const rows = Object.values(body.config.profiles).map((p) => {
    const active = p.name === body.active_profile;
    return `<div class="set-row"><span class="set-l"><b>${esc(p.name)} ${active ? '<span class="tag" data-kind="cost">active</span>' : ""}</b>
        <small>${esc(p.description || "")}</small></span>
      <button class="btn btn-s" type="button" data-activate="${esc(p.name)}"${active ? " disabled" : ""}>${active ? "Active" : "Make active"}</button></div>`;
  }).join("");
  return `<section class="sec set-sec" id="profiles" data-kicker="Profiles">
      ${intro("Active profile", "The profile every new run uses. Switching validates it first; a profile that does not validate is refused and nothing changes.")}
      <div class="stack">
        ${errors.length ? `<aside class="callout" data-kind="trap"><h4>trap<i>${esc(body.active_profile)} does not validate</i></h4><ul>${errors.map((e) => `<li>${esc(e)}</li>`).join("")}</ul></aside>`
          : `<p class="meta">Validated${snap?.resolved_at ? ` · resolved ${esc(when(snap.resolved_at))}` : ""}.</p>`}
        ${warnings.length ? `<aside class="callout" data-kind="mech"><h4>mech<i>Warnings</i></h4><ul>${warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></aside>` : ""}
        <div class="set-panel">${rows}</div>
      </div></section>`;
}

// ---- overrides -----------------------------------------------------------------------------------
function enginePick(s, spec, attrs = "") {
  const prov = spec?.provider || s.providers[0];
  return `<div class="engine-pick"${attrs}>
      <select class="input" data-f="provider" aria-label="Provider">${s.providers.map((p) => `<option${p === prov ? " selected" : ""}>${esc(p)}</option>`).join("")}</select>
      <input class="input mono" data-f="model" aria-label="Model" list="models-${s.providers.indexOf(prov)}" value="${esc(spec?.model || "")}"
        placeholder="${esc(spec?.model_from_preferences ? `preferences: ${spec.model_from_preferences}` : "model id")}">
    </div>`;
}

function roleNames(s) {
  const cfg = s.body.config;
  const names = new Set([...Object.keys(cfg.profiles[s.profile]?.roles || {}), ...Object.keys(cfg.overrides?.[s.profile]?.roles || {})]);
  if (s.body.snapshot?.profile === s.profile) Object.keys(s.body.snapshot.roles || {}).forEach((r) => names.add(r));
  return [...names].sort();
}

function overridesSec(s) {
  const cfg = s.body.config;
  const profile = cfg.profiles[s.profile];
  const ov = cfg.overrides?.[s.profile] || {};
  const tiers = TIERS.map((t) => {
    const over = ov.tiers?.[t];
    const eff = over || profile.tiers[t];
    return `<div class="set-row stack" data-tier="${t}"><span class="set-l"><b>${t} tier</b>
        <small>${esc(specText(eff))} · ${over ? "console override" : "from the profile"}</small></span>
      <div class="row">${enginePick(s, eff)}
        <button class="btn btn-s" type="button" data-act="tier-save">Override</button>
        ${over ? '<button class="btn btn-s btn-ghost" type="button" data-act="tier-clear">Clear</button>' : ""}</div>
      <p class="hint" data-f="list"></p><p class="err" data-f="err" hidden></p></div>`;
  }).join("");
  const roles = roleNames(s);
  if (!roles.includes(s.role)) s.role = roles[0] || null;
  return `<section class="sec set-sec" id="overrides" data-kicker="Overrides">
      ${intro("Overrides", "Per-profile changes kept in the bot's library, over the profile in the engines file. A tier sets the default for its roles; a role sets its own engine and fallbacks.")}
      <div class="set-panel">
        <div class="set-row"><label class="set-l" for="ov-profile"><b>Profile</b><small>Whose overrides you edit.</small></label>
          <select class="input" id="ov-profile">${Object.keys(cfg.profiles).map((p) => `<option${p === s.profile ? " selected" : ""}>${esc(p)}</option>`).join("")}</select></div>
        ${tiers}
        ${roles.length ? `<div class="set-row"><label class="set-l" for="ov-role"><b>Role</b><small>A role's own binding: primary, fallbacks in order, what happens when all fail.</small></label>
          <select class="input" id="ov-role">${roles.map((r) => `<option${r === s.role ? " selected" : ""}>${esc(r)}</option>`).join("")}</select></div>
        ${roleEditor(s)}` : ""}
      </div></section>`;
}

function roleBinding(s) {
  const cfg = s.body.config;
  const over = cfg.overrides?.[s.profile]?.roles?.[s.role];
  const base = cfg.profiles[s.profile]?.roles?.[s.role];
  return { over, binding: over || base || null, source: over ? "console override" : base ? "from the profile" : "the tier's default" };
}

function roleEditor(s) {
  const { over, binding, source } = roleBinding(s);
  const resolved = s.body.snapshot?.profile === s.profile ? s.body.snapshot.roles?.[s.role] : null;
  const fallbacks = binding?.fallbacks || [];
  return `<div class="set-row stack" id="role-editor"><span class="set-l"><b>${esc(s.role)}</b>
      <small>${esc(source)}${resolved ? ` · resolves to ${esc(specText(resolved.primary))}${resolved.fallbacks?.length ? ` then ${resolved.fallbacks.map(specText).map(esc).join(", ")}` : ""}` : ""}</small></span>
    <div class="field"><span class="field-l">Primary</span>${enginePick(s, binding?.primary || null, ' data-slot="primary"')}</div>
    <div class="field"><span class="field-l">Fallbacks <span class="faint">(tried in order)</span></span>
      <ol class="fallbacks">${fallbacks.map((f, i) => `<li>${fallbackRow(s, f, i)}</li>`).join("")}</ol>
      <div class="btn-row"><button class="btn btn-s" type="button" data-act="fb-add"><i data-icon="plus"></i>Add fallback</button></div></div>
    <div class="field"><label for="on-exhausted">When every engine fails</label>
      <select class="input" id="on-exhausted">${POLICIES.map((p) => `<option${p === (binding?.on_exhausted || "raise") ? " selected" : ""}>${p}</option>`).join("")}</select>
      <p class="hint">raise stops the run; skip_item drops that item (skippable roles); degrade uses the role's named fallback behaviour (degradable roles).</p></div>
    <div class="btn-row"><button class="btn btn-a btn-s" type="button" data-act="role-save">Override role</button>
      ${over ? '<button class="btn btn-s btn-ghost" type="button" data-act="role-clear">Clear role override</button>' : ""}</div>
    <p class="hint" data-f="list"></p><p class="err" data-f="err" hidden></p></div>`;
}

const fallbackRow = (s, spec, i = null) => `<div class="row">${enginePick(s, spec, ` data-slot="fallback"${i === null ? "" : ` data-orig="${i}"`}`)}<button class="ib" type="button" data-act="fb-remove" aria-label="Remove this fallback"><i data-icon="x"></i></button></div>`;

/** The spec a picker describes. With the original's provider it starts from the original (per-model
 *  fields survive) and an empty model keeps its preferences tier; another provider is {provider, model}. */
function specOf(pick, original) {
  const provider = $('[data-f="provider"]', pick).value;
  const model = $('[data-f="model"]', pick).value.trim();
  const same = original?.provider === provider;   // same provider: keep the spec's per-model fields
  if (model) return same ? { ...original, model, model_from_preferences: null } : { provider, model };
  if (same && original.model_from_preferences) return { ...original };
  throw new Error(`choose a model for ${provider}`);
}

function overrideBody(s, mutate) {
  const ov = clone(s.body.config.overrides?.[s.profile]);
  ov.tiers = ov.tiers || {};
  ov.roles = ov.roles || {};
  mutate(ov);
  return { overrides: { [s.profile]: { tiers: ov.tiers, roles: ov.roles } } };
}

/** Fill a provider's datalist from its live model list (once per page). */
async function loadModels(s, provider, row) {
  const i = s.providers.indexOf(provider);
  const hint = $('[data-f="list"]', row);
  if (s.lists[provider]) { hint.textContent = s.lists[provider]; return; }
  hint.textContent = `Loading ${provider}'s models…`;
  try {
    const { models } = await get.models(provider);
    $(`#models-${i}`, s.el).innerHTML = models.map((m) => `<option value="${esc(m)}"></option>`).join("");
    s.lists[provider] = `${provider}: ${models.length} live models to choose from.`;
    hint.textContent = s.lists[provider];
  } catch (e) {
    if (!(e instanceof ApiError)) throw e;
    if (e.status === 401 || e.status === 403) { showProblem(null, e); return; }   // the app swaps to the sign-in view
    hint.textContent = "";
    const err = $('[data-f="err"]', row);
    err.textContent = `${provider}'s model list is unavailable — ${e.message}. You can still type a model id.`;
    err.hidden = false;
  }
}

// ---- transcriber, spend, providers ---------------------------------------------------------------
function transcriberSec(s) {
  const t = s.body.config.transcriber || {};
  const settings = t[t.asr_engine] || {};
  const keys = Object.keys(settings);
  return `<section class="sec set-sec" id="transcriber" data-kicker="Transcriber">
      ${intro("Transcriber", "Speech-to-text for media without captions. YouTube tries the sources in order.")}
      <form class="set-panel" id="tr-form">
        <div class="set-row"><span class="set-l"><b>Speech-to-text engine</b><small>Order for YouTube: ${esc((t.youtube || []).join(" → ") || "—")}</small></span>
          <div class="seg seg-s" role="radiogroup" aria-label="Speech-to-text engine">${["faster_whisper", "openai_audio"].map((e) => `<label><input type="radio" name="asr" value="${e}"${e === t.asr_engine ? " checked" : ""}><span>${e === "faster_whisper" ? "faster-whisper (local)" : "OpenAI-compatible audio"}</span></label>`).join("")}</div></div>
        <div class="set-row"><span class="set-l"><b>Its settings</b><small>From the engines file.</small></span>
          <span class="meta">${keys.length ? keys.map((k) => `${esc(k)}: ${esc(JSON.stringify(settings[k]))}`).join(" · ") : "none set (the engine's defaults)"}</span></div>
        <div class="set-panel-f"><button class="btn btn-a" type="button" id="tr-save">Save transcriber</button></div>
      </form></section>`;
}

function spendSec(s) {
  const exa = s.body.config.pricing?.exa;
  return `<section class="sec set-sec" id="spend" data-kicker="Spend">
      ${intro("Exa spend", "Search credit used this calendar month, summed from the runs in the library.")}
      <div class="kpis"><div class="kpi"><span class="kpi-k">Exa this month</span><span class="kpi-v">${esc(usd(s.body.exa_month_to_date_usd, 2))}</span>
        <span class="kpi-s">${exa ? `${esc(usd(exa.price_per_search_usd, 4))} per search · ${esc(usd(exa.price_per_page_usd, 4))} per page · prices as of ${esc(exa.prices_as_of)}` : "no Exa prices configured"}</span></div></div>
    </section>`;
}

function providersSec(s) {
  const rows = Object.values(s.body.config.providers).map((p) => {
    const key = p.api_key || {};
    const st = key.env === null ? '<span class="status" data-s="idle">keyless</span>'
      : key.present ? `<span class="status" data-s="good">${esc(key.env)} set</span>` : `<span class="status" data-s="critical">${esc(key.env)} missing</span>`;
    return `<tr class="prov"><td><b>${esc(p.name)}</b></td><td class="mono url">${esc(p.base_url)}</td>
      <td>${st}</td><td>${p.models_endpoint ? "live list" : `${(p.known_models || []).length} known`}</td></tr>${p.notes ? `<tr class="prov-note"><td colspan="4">${esc(p.notes)}</td></tr>` : ""}`;
  }).join("");
  return `<section class="sec set-sec" id="providers" data-kicker="Providers">
      ${intro("Providers", "Where each engine runs. Read-only here: edit <code>argus_engines.json</code> to change one. Key values are never shown.")}
      <div class="tw"><table><thead><tr><th>Provider</th><th>Base URL</th><th>API key</th><th>Models</th></tr></thead><tbody>${rows}</tbody></table></div>
    </section>`;
}

// ---- wiring --------------------------------------------------------------------------------------
function wire(s) {
  const el = s.el;
  const slot = (id) => $(`#${id} .sec-notice`, el);
  $$("[data-activate]", el).forEach((b) => b.addEventListener("click", () => save(s, b, slot("profiles"),
    { active_profile: b.dataset.activate }, `${b.dataset.activate} is now the active profile.`)));

  $("#ov-profile", el).addEventListener("change", (e) => { s.profile = e.target.value; s.role = null; paint(s); });
  const role = $("#ov-role", el);
  if (role) role.addEventListener("change", (e) => { s.role = e.target.value; paint(s); });

  // pickers: the model list follows the provider
  el.addEventListener("change", (e) => {
    const sel = e.target.closest('.engine-pick [data-f="provider"]');
    if (!sel) return;
    const pick = sel.closest(".engine-pick");
    $('[data-f="model"]', pick).setAttribute("list", `models-${s.providers.indexOf(sel.value)}`);
    $('[data-f="model"]', pick).value = "";
    loadModels(s, sel.value, pick.closest(".set-row"));
  });
  el.addEventListener("focusin", (e) => {
    const input = e.target.closest('.engine-pick [data-f="model"]');
    if (input) loadModels(s, $('[data-f="provider"]', input.closest(".engine-pick")).value, input.closest(".set-row"));
  });

  const rowError = (row, message) => { const err = $('[data-f="err"]', row); err.textContent = message; err.hidden = false; };
  $$("[data-tier]", el).forEach((row) => {
    const tier = row.dataset.tier;
    const original = s.body.config.overrides?.[s.profile]?.tiers?.[tier] || s.body.config.profiles[s.profile].tiers[tier];
    $('[data-act="tier-save"]', row).addEventListener("click", (e) => {
      let spec;
      try { spec = specOf($(".engine-pick", row), original); } catch (err) { rowError(row, err.message); return; }
      save(s, e.currentTarget, slot("overrides"), overrideBody(s, (ov) => { ov.tiers[tier] = spec; }),
        `${s.profile}: the ${tier} tier now uses ${specText(spec)}.`);
    });
    const clear = $('[data-act="tier-clear"]', row);
    if (clear) clear.addEventListener("click", () => save(s, clear, slot("overrides"),
      overrideBody(s, (ov) => { delete ov.tiers[tier]; }), `${s.profile}: the ${tier} tier follows the profile again.`));
  });

  $("#tr-save", el).addEventListener("click", (e) => {
    const asr = $("#tr-form", el).elements.namedItem("asr").value;
    save(s, e.currentTarget, slot("transcriber"), { transcriber: { ...s.body.config.transcriber, asr_engine: asr } },
      `Speech-to-text now uses ${asr}.`);
  });

  const editor = $("#role-editor", el);
  if (!editor) return;
  const { binding } = roleBinding(s);
  $('[data-act="fb-add"]', editor).addEventListener("click", () => {
    const li = document.createElement("li");
    li.innerHTML = fallbackRow(s, null);
    $(".fallbacks", editor).append(li);
    window.UI.init(li);
  });
  editor.addEventListener("click", (e) => {
    const rm = e.target.closest('[data-act="fb-remove"]');
    if (rm) rm.closest("li").remove();
  });
  $('[data-act="role-save"]', editor).addEventListener("click", (e) => {
    let next;
    try {
      const fbs = $$('[data-slot="fallback"]', editor);
      next = {
        primary: specOf($('[data-slot="primary"]', editor), binding?.primary),
        fallbacks: fbs.map((p) => specOf(p, p.dataset.orig === undefined ? null : binding?.fallbacks?.[Number(p.dataset.orig)])),
        on_exhausted: $("#on-exhausted", editor).value,
        retry_on_truncation: binding?.retry_on_truncation ?? true,
      };
    } catch (err) { rowError(editor, err.message); return; }
    save(s, e.currentTarget, slot("overrides"), overrideBody(s, (ov) => { ov.roles[s.role] = next; }),
      `${s.profile}: ${s.role} now runs on ${specText(next.primary)}${next.fallbacks.length ? ` with ${next.fallbacks.length} fallback(s)` : ""}.`);
  });
  const clearRole = $('[data-act="role-clear"]', editor);
  if (clearRole) clearRole.addEventListener("click", () => save(s, clearRole, slot("overrides"),
    overrideBody(s, (ov) => { delete ov.roles[s.role]; }), `${s.profile}: ${s.role} follows the profile again.`));
}
