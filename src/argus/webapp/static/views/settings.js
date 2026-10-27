// #/settings — the console's look: follow Telegram's light/dark or pin one, and which light and which dark
// theme (the kit's themes). Saved on the bot (PUT /api/appearance); app.js applies it ("argus:appearance").
import { get, saveAppearance } from "../api.js";
import { $, $$, act, esc } from "./common.js";

const MODES = [["telegram", "Follow Telegram"], ["light", "Always light"], ["dark", "Always dark"]];
const LIGHT = [["ledger", "Ledger", "warm parchment, amber"], ["steel", "Steel", "cool grey, blue"],
  ["solar", "Solar", "cream, ocean blue"], ["paper", "Paper", "white, ink blue"]];
const DARK = [["dark", "Ledger dark", "espresso, amber"], ["graphite", "Graphite", "blue-black, steel"],
  ["slate", "Slate", "deep slate, teal"], ["contrast", "Contrast", "black, signal yellow"]];

export async function render(el, _params, ctx) {
  ctx.crumbs([["", "Settings"]]);
  let { appearance: a } = await get.me();
  const swatches = (key, label, themes) => `<div class="set-row stack"><span class="set-l"><b>${esc(label)}</b>
      <small id="${key}-name"></small></span>
      <div class="themes" role="radiogroup" aria-label="${esc(label)}">${themes.map(([id, name]) => `<button class="tsw" type="button" role="radio" data-key="${key}" data-theme-id="${id}" aria-label="${esc(name)}" data-tip="${esc(name)}"><i></i></button>`).join("")}</div></div>`;
  el.innerHTML = `<div class="page-head"><div><span class="kicker">Settings</span><h1>How the console looks</h1>
      <p>Saved on the bot, so every device you open the console from looks the same.</p></div></div>
    <section class="sec set-sec" data-kicker="Theme">
      <div class="set-intro"><h2>Theme</h2><p>Follow Telegram switches between your light and dark theme with the app; or pin one.</p>
        <div class="sec-notice" hidden></div></div>
      <div class="set-panel">
        <div class="set-row"><span class="set-l"><b>Light or dark</b><small>Telegram's own setting, or always one.</small></span>
          <div class="seg seg-s" role="radiogroup" aria-label="Light or dark">${MODES.map(([v, label]) => `<label><input type="radio" name="mode" value="${v}"><span>${esc(label)}</span></label>`).join("")}</div></div>
        ${swatches("light", "Light theme", LIGHT)}
        ${swatches("dark", "Dark theme", DARK)}
      </div>
    </section>`;
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
