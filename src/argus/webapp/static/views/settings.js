// #/settings — the console's look: follow Telegram's light/dark or pin one, and which light and which dark
// theme (the kit's themes). Saved on the bot (PUT /api/appearance); app.js applies it ("argus:appearance").
// Backend: the site-dependent packages (updates.py), their versions and an Update button each.
import { get, saveAppearance, updates } from "../api.js";
import { $, $$, act, esc, toast } from "./common.js";

const MODES = [["telegram", "Follow Telegram"], ["light", "Always light"], ["dark", "Always dark"]];
const LIGHT = [["ledger", "Ledger", "warm parchment, amber"], ["steel", "Steel", "cool grey, blue"],
  ["solar", "Solar", "cream, ocean blue"], ["paper", "Paper", "white, ink blue"]];
const DARK = [["dark", "Ledger dark", "espresso, amber"], ["graphite", "Graphite", "blue-black, steel"],
  ["slate", "Slate", "deep slate, teal"], ["contrast", "Contrast", "black, signal yellow"]];

// The imported Noetrix themes (scripts/import_noetrix_themes.py), appended once per page load.
const noetrix = fetch("/static/themes-noetrix.json").then((r) => {
  if (!r.ok) throw new Error(`themes-noetrix.json: HTTP ${r.status}`);
  return r.json();
}).then((list) => {
  for (const t of list) (t.mode === "light" ? LIGHT : DARK).push([t.id, t.name, "Noetrix"]);
});

export async function render(el, _params, ctx) {
  ctx.crumbs([["", "Settings"]]);
  await noetrix;
  let { appearance: a } = await get.me();
  const swatches = (key, label, themes) => `<div class="set-row stack"><span class="set-l"><b>${esc(label)}</b>
      <small id="${key}-name"></small></span>
      <div class="themes" role="radiogroup" aria-label="${esc(label)}">${themes.map(([id, name]) => `<button class="tsw" type="button" role="radio" data-key="${key}" data-theme-id="${id}" aria-label="${esc(name)}" data-tip="${esc(name)}"><i></i></button>`).join("")}</div></div>`;
  el.innerHTML = `<div class="page-head"><div><span class="kicker">Settings</span><h1>Look and backend</h1>
      <p>The theme is saved on the bot, so every device you open the console from looks the same.</p></div></div>
    <section class="sec set-sec" data-kicker="Theme">
      <div class="set-intro"><h2>Theme</h2><p>Follow Telegram switches between your light and dark theme with the app; or pin one.</p>
        <div class="sec-notice" hidden></div></div>
      <div class="set-panel">
        <div class="set-row"><span class="set-l"><b>Light or dark</b><small>Telegram's own setting, or always one.</small></span>
          <div class="seg seg-s" role="radiogroup" aria-label="Light or dark">${MODES.map(([v, label]) => `<label><input type="radio" name="mode" value="${v}"><span>${esc(label)}</span></label>`).join("")}</div></div>
        ${swatches("light", "Light theme", LIGHT)}
        ${swatches("dark", "Dark theme", DARK)}
      </div>
    </section>
    <section class="sec set-sec" data-kicker="Backend">
      <div class="set-intro"><h2>Backend</h2><p>The packages that break when sites change. Update installs the latest release into the bot's own environment.</p>
        <div class="sec-notice" id="upd-notice" hidden></div></div>
      <div class="set-panel" id="upd"><p class="meta">Checking versions…</p></div>
    </section>`;
  backend(el);
  const notice = $(".sec-notice", el);
  const paint = () => {
    $$("input[name=mode]", el).forEach((r) => { r.checked = r.value === a.mode; });
    $$(".tsw", el).forEach((b) => b.setAttribute("aria-checked", String(a[b.dataset.key] === b.dataset.themeId)));
    for (const [key, themes] of [["light", LIGHT], ["dark", DARK]]) {
      const [, name, note] = themes.find(([id]) => id === a[key]);
      $(`#${key}-name`, el).textContent = `${name} — ${note}`;
    }
    window.dispatchEvent(new CustomEvent("argus:appearance", { detail: a }));
  };
  const save = async (button, change) => {
    const r = await act(button, notice, () => saveAppearance(change));
    if (r) a = r.appearance;
    paint();                                     // a refused change repaints the saved state
  };
  $$("input[name=mode]", el).forEach((r) => r.addEventListener("change", () => save(null, { mode: r.value })));
  $$(".tsw", el).forEach((b) => b.addEventListener("click", () => {
    // picking a light theme while pinned dark (or the reverse) shows it: the pin follows the pick
    const change = { [b.dataset.key]: b.dataset.themeId };
    if (a.mode !== "telegram" && a.mode !== b.dataset.key) change.mode = b.dataset.key;
    save(b, change);
  }));
  paint();
}

function componentRow(c) {
  const versions = c.packages.map((p) => `${esc(p.name)} <span class="mono">${esc(p.installed || "not installed")}</span>`).join(" · ");
  const current = c.packages[0].installed === c.latest;
  const st = c.restart_pending ? '<span class="status" data-s="serious">restart to apply</span>'
    : c.latest_error ? '<span class="status" data-s="critical">latest unknown</span>'
      : current ? '<span class="status" data-s="good">up to date</span>'
        : `<span class="status" data-s="idle">${esc(c.latest)} available</span>`;
  return `<div class="set-row"><span class="set-l"><b>${esc(c.name)} ${st}</b>
      <small>${versions}${c.latest ? ` · latest <span class="mono">${esc(c.latest)}</span>` : ""}</small>
      <small>${esc(c.latest_error ? `PyPI: ${c.latest_error}` : c.note)}</small></span>
    <button class="btn btn-s" type="button" data-update="${esc(c.name)}"${current ? " disabled" : ""}>${current ? "Up to date" : "Update"}</button></div>`;
}

async function backend(el) {
  const panel = $("#upd", el), notice = $("#upd-notice", el);
  let last = [];
  const paint = (components) => {
    last = components;
    panel.innerHTML = components.map(componentRow).join("");
    $$("[data-update]", panel).forEach((b) => b.addEventListener("click", async () => {
      $$("[data-update]", panel).forEach((x) => { x.disabled = true; });
      b.textContent = "Updating…";
      const r = await act(b, notice, () => updates.run(b.dataset.update));
      if (r) {
        paint(r.components);
        const what = r.changes.filter((ch) => ch.op === "install").map((ch) => `${ch.package} ${ch.version}`).join(", ");
        toast(what ? `Installed ${what}.${r.restart_needed ? " Restart the bot to load it." : ""}` : "Already up to date.",
          "cost", `${r.name} update`);
      } else {
        paint(last);                               // a failed update repaints the state it started from
      }
    }));
  };
  const r = await act(null, notice, () => updates.list());
  if (r) paint(r.components);
  else panel.innerHTML = '<p class="meta">Versions could not be read.</p>';
}
