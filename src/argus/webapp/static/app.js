// The console's hash router: #/runs, #/run/<id>, #/new, #/engines. Each view module exports
// render(el, params, ctx) and may return a cleanup function (polling, Telegram buttons).
import { ApiError, get } from "./api.js";
import { takeLaunchParams } from "./auth.js";
import { $, $$, backButton, clearMainButton, esc, problemHtml, tg } from "./views/common.js";
import * as engines from "./views/engines.js";
import * as media from "./views/media.js";
import * as mediaItem from "./views/media_item.js";
import * as newRun from "./views/new.js";
import * as run from "./views/run.js";
import * as runs from "./views/runs.js";
import * as settings from "./views/settings.js";

const ROUTES = [
  [/^\/runs$/, runs, "runs"],
  [/^\/run\/([A-Za-z0-9_-]+)$/, run, "runs"],
  [/^\/new$/, newRun, "new"],
  [/^\/engines$/, engines, "engines"],
  [/^\/media(?:\/(library|get|review|jobs|settings))?$/, media, "media"],
  [/^\/media\/item\/(\d+)$/, mediaItem, "media"],
  [/^\/settings$/, settings, "settings"],
];
const view = $("#view");
let cleanup = null;
let authFailed = false;
let shownPath = null;                      // a new view opens at the top; a refresh keeps the place

// ---- theme: the saved appearance (#/settings) — Telegram's light/dark or a pinned one, then which kit
// theme; Telegram's own header and background take the theme's ground colour
let appearance = { mode: "telegram", light: "ledger", dark: "dark" };
function applyTheme() {
  const webApp = tg();
  const dark = appearance.mode === "dark" || (appearance.mode === "telegram" && webApp?.colorScheme === "dark");
  document.documentElement.setAttribute("data-theme", dark ? appearance.dark : appearance.light);
  const ground = getComputedStyle(document.documentElement).getPropertyValue("--ground").trim();
  if (/^#[0-9a-f]{6}$/i.test(ground) && webApp?.isVersionAtLeast?.("6.9")) {
    webApp.setHeaderColor(ground);
    webApp.setBackgroundColor(ground);
  }
}
window.addEventListener("argus:appearance", (e) => { appearance = e.detail; applyTheme(); });

function crumbs(items) {
  $("#crumbs").innerHTML = [["#/runs", "Argus"], ...items].map(([href, label], i, all) => (i === all.length - 1
    ? `<li><span aria-current="page">${esc(label)}</span></li>`
    : `<li><a href="${esc(href)}">${esc(label)}</a></li>`)).join("");
}

function showAuth(err) {
  authFailed = true;
  if (cleanup) { cleanup(); cleanup = null; }
  clearMainButton();
  crumbs([["", "Sign-in"]]);
  view.innerHTML = `<div class="page-head"><div><span class="kicker">Argus console</span><h1>${esc(err.code === "expired"
    ? "Session expired — reopen from the menu button" : "The console could not sign you in")}</h1>`
    + "<p>Close this window and tap the <b>Argus</b> menu button in your chat with the bot.</p></div></div>"
    + `<section class="sec">${problemHtml(err, { dismiss: false })}</section>`;
  $("#who").textContent = "not signed in";
}
window.addEventListener("argus:auth", (e) => showAuth(e.detail));

/** A view (or the sign-in check) that could not load: the explicit problem view, with the API's message. */
function showLoadProblem(e) {
  crumbs([["", "Problem"]]);
  view.innerHTML = `<div class="page-head"><div><span class="kicker">Argus console</span><h1>This view could not load</h1>`
    + "<p>Use the refresh button above to try again.</p></div></div>"
    + `<section class="sec">${problemHtml(e, { dismiss: false })}<div class="btn-row"><a class="btn" href="#/runs">Back to runs</a></div></section>`;
}

const goBack = () => { if (history.length > 1) history.back(); else location.hash = "#/runs"; };

async function route() {
  if (authFailed) return;
  if (cleanup) { cleanup(); cleanup = null; }
  clearMainButton();
  const path = location.hash.replace(/^#/, "") || "/runs";
  const hit = ROUTES.map(([re, mod, nav]) => [path.match(re), mod, nav]).find(([m]) => m);
  if (!hit) { location.replace("#/runs"); return; }
  const [m, mod, nav] = hit;
  if (path !== shownPath) { window.scrollTo(0, 0); shownPath = path; }
  $$("#nav a").forEach((a) => {
    const on = a.dataset.route === nav;
    a.classList.toggle("on", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  backButton(path !== "/runs", goBack);
  view.setAttribute("aria-busy", "true");
  try {
    cleanup = (await mod.render(view, m.slice(1), { crumbs })) || null;
  } catch (e) {
    if (!(e instanceof ApiError)) throw e;
    if (e.status === 401 || e.status === 403) { showAuth(e); return; }
    showLoadProblem(e);
  } finally {
    view.removeAttribute("aria-busy");
  }
  window.UI.init(view);
}

async function start() {
  takeLaunchParams();
  applyTheme();
  const webApp = tg();
  if (webApp) {
    webApp.ready?.();
    webApp.expand?.();
    webApp.onEvent?.("themeChanged", applyTheme);
  }
  $("#refresh").addEventListener("click", () => route());
  window.addEventListener("hashchange", () => route());
  try {
    const me = await get.me();
    appearance = me.appearance;
    applyTheme();
    $("#who").textContent = `Signed in as ${me.user.first_name || me.user.id} · Argus ${me.bot.version}`;
  } catch (e) {
    if (!(e instanceof ApiError)) throw e;
    if (e.status === 401 || e.status === 403) { showAuth(e); return; }
    $("#who").textContent = "not reachable";
    showLoadProblem(e);                    // e.g. the network view; the refresh button retries
    return;
  }
  await route();
}

start();
