/* GENERATED from ui.js by kits/web-ui/build/build.py. Classic script: exposes window.UI. */
(function () {
/* Arsenal UI — vanilla, dependency-free behaviours for kits/web-ui components.
   This file is an ES module. ui.classic.js (window.UI) is GENERATED from it by build/build.py:
   use the classic file in pages opened from disk (module scripts are blocked on file://).
   Everything auto-initialises on DOMContentLoaded; call UI.init(el) after injecting markup.
   Each initialiser documents its markup contract. Errors in authoring (a tab without a panel,
   a glossary term without a definition) throw — they are bugs to fix, not states to hide. */

const THEMES = /*<gen:themes>*/[{"id": "ledger", "label": "Ledger", "scheme": "light"}, {"id": "dark", "label": "Ledger Dark", "scheme": "dark"}, {"id": "steel", "label": "Steel", "scheme": "light"}, {"id": "graphite", "label": "Graphite", "scheme": "dark"}, {"id": "solar", "label": "Solar", "scheme": "light"}, {"id": "slate", "label": "Slate", "scheme": "dark"}, {"id": "paper", "label": "Paper", "scheme": "light"}, {"id": "contrast", "label": "High contrast", "scheme": "dark"}]/*</gen:themes>*/;
const BUILTIN_ICONS = /*<gen:icons>*/{"panel-left":["l","<rect width=\"18\" height=\"18\" x=\"3\" y=\"3\" rx=\"2\" /><path d=\"M9 3v18\" />"],"sun":["l","<circle cx=\"12\" cy=\"12\" r=\"4\" /><path d=\"M12 2v2\" /><path d=\"M12 20v2\" /><path d=\"m4.93 4.93 1.41 1.41\" /><path d=\"m17.66 17.66 1.41 1.41\" /><path d=\"M2 12h2\" /><path d=\"M20 12h2\" /><path d=\"m6.34 17.66-1.41 1.41\" /><path d=\"m19.07 4.93-1.41 1.41\" />"],"moon":["l","<path d=\"M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401\" />"],"search":["l","<path d=\"m21 21-4.34-4.34\" /><circle cx=\"11\" cy=\"11\" r=\"8\" />"],"skip-back":["l","<path d=\"M17.971 4.285A2 2 0 0 1 21 6v12a2 2 0 0 1-3.029 1.715l-9.997-5.998a2 2 0 0 1-.003-3.432z\" /><path d=\"M3 20V4\" />"],"chevron-left":["l","<path d=\"m15 18-6-6 6-6\" />"],"chevron-right":["l","<path d=\"m9 18 6-6-6-6\" />"],"play":["l","<path d=\"M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z\" />"],"pause":["l","<rect x=\"14\" y=\"3\" width=\"5\" height=\"18\" rx=\"1\" /><rect x=\"5\" y=\"3\" width=\"5\" height=\"18\" rx=\"1\" />"],"hash":["l","<line x1=\"4\" x2=\"20\" y1=\"9\" y2=\"9\" /><line x1=\"4\" x2=\"20\" y1=\"15\" y2=\"15\" /><line x1=\"10\" x2=\"8\" y1=\"3\" y2=\"21\" /><line x1=\"16\" x2=\"14\" y1=\"3\" y2=\"21\" />"],"palette":["l","<path d=\"M12 22a1 1 0 0 1 0-20 10 9 0 0 1 10 9 5 5 0 0 1-5 5h-2.25a1.75 1.75 0 0 0-1.4 2.8l.3.4a1.75 1.75 0 0 1-1.4 2.8z\" /><circle cx=\"13.5\" cy=\"6.5\" r=\".5\" fill=\"currentColor\" /><circle cx=\"17.5\" cy=\"10.5\" r=\".5\" fill=\"currentColor\" /><circle cx=\"6.5\" cy=\"12.5\" r=\".5\" fill=\"currentColor\" /><circle cx=\"8.5\" cy=\"7.5\" r=\".5\" fill=\"currentColor\" />"],"printer":["l","<path d=\"M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2\" /><path d=\"M6 9V3a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v6\" /><rect x=\"6\" y=\"14\" width=\"12\" height=\"8\" rx=\"1\" />"]}/*</gen:icons>*/;
const ICON_ATTRS = {
  l: 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"',
  t: 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"',
  tf: 'viewBox="0 0 24 24" fill="currentColor"',
  p: 'viewBox="0 0 256 256" fill="currentColor"',
};
const THEME_KEY = "arsenal-theme";
const RAIL_KEY = "arsenal-rail";
const doc = document;
const root = doc.documentElement;
const $ = (s, r = doc) => r.querySelector(s);
const $$ = (s, r = doc) => Array.from(r.querySelectorAll(s));
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
let uid = 0;
const newId = (p) => `${p}-${++uid}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const slug = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "") || "s";
const isTyping = (el) => el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
/* Storage throws in sandboxed iframes and some privacy modes; the page must still work, unpersisted. */
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* not persisted */ } },
  json(k, fallback) {
    const raw = this.get(k);
    if (raw === null) return fallback;
    try { return JSON.parse(raw); } catch { console.warn(`arsenal ui: discarding unreadable saved state "${k}"`); return fallback; }
  },
};
const emit = (el, name, detail) => el.dispatchEvent(new CustomEvent(name, { detail, bubbles: true }));
const sectionLabel = (sec) => sec ? (sec.dataset.kicker || $(".kicker", sec)?.textContent || $("h2", sec)?.textContent || "").trim() : "";

/* ---- icons --------------------------------------------------------------------------
   <i data-icon="search"></i> → inline SVG. Registry: window.ARSENAL_ICONS (kits/web-ui/icons/icons.js
   or your own subset from build/icons.py) plus the few icons this file needs itself. */
function icon(name, cls = "icon") {
  const ic = (window.ARSENAL_ICONS || {})[name] || BUILTIN_ICONS[name];
  if (!ic) throw new Error(`UI.icon: unknown icon "${name}" (load kits/web-ui/icons/icons.js or build a subset with build/icons.py)`);
  return `<svg class="${esc(cls)}" ${ICON_ATTRS[ic[0]]} aria-hidden="true" focusable="false">${ic[1]}</svg>`;
}
function initIcon(el) {
  const tpl = doc.createElement("template");
  tpl.innerHTML = icon(el.dataset.icon, el.className || "icon");
  el.replaceWith(tpl.content.firstElementChild);
}

/* ---- theme -------------------------------------------------------------------------- */
const getTheme = () => root.getAttribute("data-theme") || "system";
function setTheme(t) {
  if (t !== "system" && !THEMES.some((x) => x.id === t)) throw new Error(`UI.setTheme: unknown theme "${t}"`);
  if (t === "system") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", t);
  store.set(THEME_KEY, t);
}
const isDark = () => {
  const t = THEMES.find((x) => x.id === getTheme());
  return t ? t.scheme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};
/* A saved choice applies only when the page itself did not pin a theme in its markup. */
(() => {
  const saved = store.get(THEME_KEY);
  if (saved && saved !== "system" && !root.hasAttribute("data-theme") && THEMES.some((x) => x.id === saved)) root.setAttribute("data-theme", saved);
})();
const themeSyncers = new Set();
new MutationObserver(() => themeSyncers.forEach((f) => f())).observe(root, { attributes: true, attributeFilter: ["data-theme"] });

/* <div data-ui="theme-picker"></div> → swatch radiogroup: System + every theme in tokens.json */
function initThemePicker(el) {
  const opts = [{ id: "system", label: "System" }, ...THEMES];
  el.classList.add("themes");
  el.setAttribute("role", "radiogroup");
  if (!el.hasAttribute("aria-label") && !el.hasAttribute("aria-labelledby")) el.setAttribute("aria-label", "Theme");
  el.innerHTML = opts.map((o) => `<button type="button" class="tsw" role="radio" data-theme-id="${o.id}" aria-label="${esc(o.label)}" data-tip="${esc(o.label)}"><i></i></button>`).join("")
    + (el.hasAttribute("data-no-label") ? "" : '<span class="theme-name" aria-hidden="true"></span>');
  const btns = $$(".tsw", el);
  const sync = () => {
    const cur = getTheme();
    btns.forEach((b) => { const on = b.dataset.themeId === cur; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
    const name = $(".theme-name", el);
    if (name) name.textContent = (opts.find((o) => o.id === cur) || { label: cur }).label;
  };
  el.addEventListener("click", (e) => { const b = e.target.closest(".tsw"); if (b) setTheme(b.dataset.themeId); });
  el.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!step) return;
    e.preventDefault();
    const i = (btns.findIndex((b) => b.dataset.themeId === getTheme()) + step + btns.length) % btns.length;
    setTheme(btns[i].dataset.themeId);
    btns[i].focus();
  });
  themeSyncers.add(sync);
  sync();
}
/* <button data-ui="theme-toggle"></button> → flips between the house light and dark themes */
function initThemeToggle(b) {
  const light = (THEMES.find((t) => t.scheme === "light") || {}).id;
  const dark = (THEMES.find((t) => t.scheme === "dark") || {}).id;
  const sync = () => {
    b.innerHTML = icon(isDark() ? "sun" : "moon");
    b.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
    b.dataset.tip = b.getAttribute("aria-label");
  };
  b.addEventListener("click", () => setTheme(isDark() ? light : dark));
  themeSyncers.add(sync);
  sync();
}

/* ---- scroll spy (rail stations, docs TOC) ------------------------------------------- */
function scrollSpy(links, onActive) {
  const targets = links.map((a) => doc.getElementById(decodeURIComponent(a.hash.slice(1)))).filter(Boolean);
  if (!targets.length || !("IntersectionObserver" in window)) return;
  const seen = new Map();
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => seen.set(e.target, e.isIntersecting));
    const first = targets.find((t) => seen.get(t));
    if (!first) return;
    links.forEach((a) => {
      const on = a.hash.slice(1) === first.id;
      a.classList.toggle("on", on);
      if (on) a.setAttribute("aria-current", "location"); else a.removeAttribute("aria-current");
    });
    if (onActive) onActive(first);
  }, { rootMargin: "-12% 0px -70% 0px" });
  targets.forEach((t) => io.observe(t));
}

/* ---- rail ---------------------------------------------------------------------------
   <div class="shell"><aside class="rail" data-ui="rail">
     <div class="rail-head"><span class="mark-word"><b>Name</b><span>Kit</span></span><button class="rail-toggle" data-rail-toggle></button></div>
     <nav class="rail-nav" aria-label="Sections"><ol class="track" data-auto></ol></nav> …
   data-auto builds one station per `main section[id]` (kicker from data-kicker or .kicker, title from h2);
   give it a selector value (data-auto="article h2[id]") to choose other targets. */
function buildTrack(track) {
  const sel = track.dataset.auto || "main section[id]";
  const secs = $$(sel);
  if (!secs.length) throw new Error(`rail: data-auto found no targets for "${sel}"`);
  track.innerHTML = secs.map((s) => {
    const kick = s.matches("section") ? sectionLabel(s) : "";
    const title = (s.dataset.title || (s.matches("h2,h3") ? s.textContent : $("h2", s)?.textContent) || s.id).trim();
    const k = kick && kick !== title ? `<span class="kick">${esc(kick)}</span>` : "";
    return `<li><a href="#${esc(s.id)}"><span>${k}${esc(title)}</span></a></li>`;
  }).join("");
}
function initRail(rail) {
  const shell = rail.closest(".shell");
  const track = $(".track", rail);
  if (track && track.hasAttribute("data-auto")) buildTrack(track);
  const links = track ? $$('a[href^="#"]', track) : [];
  const tog = $("[data-rail-toggle]", rail);
  if (tog && shell) {
    tog.type = "button";
    if (!tog.innerHTML.trim()) tog.innerHTML = icon("panel-left");
    const apply = (collapsed) => {
      shell.dataset.rail = collapsed ? "collapsed" : "open";
      tog.setAttribute("aria-expanded", String(!collapsed));
      tog.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
      tog.dataset.tip = tog.getAttribute("aria-label");
      links.forEach((a) => {
        if (collapsed) { a.setAttribute("aria-label", a.textContent.trim()); a.dataset.tip = a.querySelector("span")?.lastChild?.textContent.trim() || a.textContent.trim(); }
        else { a.removeAttribute("aria-label"); delete a.dataset.tip; }
      });
    };
    apply(store.get(RAIL_KEY) === "collapsed");
    tog.addEventListener("click", () => {
      const c = shell.dataset.rail !== "collapsed";
      apply(c);
      store.set(RAIL_KEY, c ? "collapsed" : "open");
    });
  }
  scrollSpy(links);
}

/* ---- learned sections + progress ------------------------------------------------------
   <button class="learn" data-learn="sec-id"><span class="learn-box"></span>Mark as learned</button>
   <div data-ui="learn-progress"><span><b data-learn-count></b> sections learned</span><div class="bar"><i></i></div></div> */
let learnReady = false;
function initLearn() {
  const btns = $$("[data-learn]");
  if (learnReady || !btns.length) return;
  learnReady = true;
  const key = "arsenal-learn:" + location.pathname;
  const ids = [...new Set(btns.map((b) => b.dataset.learn))];
  const got = new Set(store.json(key, []));
  const paint = () => {
    btns.forEach((b) => b.setAttribute("aria-pressed", String(got.has(b.dataset.learn))));
    $$(".track a").forEach((a) => a.classList.toggle("done", got.has(a.hash.slice(1))));
    const n = ids.filter((i) => got.has(i)).length;
    $$("[data-ui=learn-progress]").forEach((p) => {
      const c = $("[data-learn-count]", p); if (c) c.textContent = `${n} / ${ids.length}`;
      const bar = $(".bar", p); if (bar) bar.style.setProperty("--p", `${(100 * n) / ids.length}%`);
    });
  };
  btns.forEach((b) => {
    b.type = "button";
    b.addEventListener("click", () => {
      const i = b.dataset.learn;
      if (got.has(i)) got.delete(i); else got.add(i);
      store.set(key, JSON.stringify([...got]));
      paint();
    });
  });
  paint();
}

/* ---- tabs ----------------------------------------------------------------------------
   <div class="tabs" data-ui="tabs"><div class="tablist"><button>One</button><button>Two</button></div>
   <div class="tabpanel">…</div><div class="tabpanel">…</div></div>   (roles/ids/aria are added here) */
function initTabs(el) {
  const list = $(":scope > .tablist", el);
  if (!list) throw new Error("tabs: missing .tablist");
  const tabs = $$(":scope > button", list);
  const panels = $$(":scope > .tabpanel", el);
  if (tabs.length !== panels.length) throw new Error(`tabs: ${tabs.length} tabs but ${panels.length} panels`);
  list.setAttribute("role", "tablist");
  tabs.forEach((t, i) => {
    const p = panels[i];
    t.id ||= newId("tab"); p.id ||= newId("tabpanel");
    t.type = "button";
    t.setAttribute("role", "tab"); t.setAttribute("aria-controls", p.id);
    p.setAttribute("role", "tabpanel"); p.setAttribute("aria-labelledby", t.id); p.tabIndex = 0;
  });
  const select = (i, focus) => {
    tabs.forEach((t, j) => { const on = i === j; t.setAttribute("aria-selected", String(on)); t.tabIndex = on ? 0 : -1; panels[j].hidden = !on; });
    if (focus) tabs[i].focus();
    emit(el, "tabchange", { index: i, label: tabs[i].textContent.trim() });
  };
  list.addEventListener("click", (e) => { const t = e.target.closest('[role="tab"]'); if (t) select(tabs.indexOf(t)); });
  list.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(doc.activeElement);
    if (i < 0) return;
    const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    select((to + tabs.length) % tabs.length, true);
  });
  select(Math.max(0, tabs.findIndex((t) => t.getAttribute("aria-selected") === "true")));
}

/* ---- segmented control (button variant; the radio variant needs no JS) ----------------
   <div class="seg" data-ui="seg" aria-label="Range"><button aria-pressed="true" value="7d">7 days</button>…</div>
   Emits "change" with detail.value. */
function initSeg(el) {
  const bs = $$(":scope > button", el);
  el.setAttribute("role", "group");
  bs.forEach((b) => { b.type = "button"; if (!b.hasAttribute("aria-pressed")) b.setAttribute("aria-pressed", "false"); });
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.parentElement !== el) return;
    bs.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    emit(el, "change", { value: b.value || b.textContent.trim() });
  });
}

/* ---- accordion helpers: native <details class="acc" name="group">; this adds "expand all"
   <button data-expand-all="#drills">Open all</button> */
function initExpandAll(b) {
  const scope = $(b.dataset.expandAll);
  if (!scope) throw new Error(`expand-all: no element matches "${b.dataset.expandAll}"`);
  const label = b.textContent.trim();
  b.type = "button";
  b.setAttribute("aria-pressed", "false");
  b.addEventListener("click", () => {
    const open = b.getAttribute("aria-pressed") !== "true";
    $$("details", scope).forEach((d) => { if (open) d.removeAttribute("name"); d.open = open; });
    b.setAttribute("aria-pressed", String(open));
    b.textContent = open ? (b.dataset.closeLabel || "Close all") : label;
  });
}

/* ---- dialog --------------------------------------------------------------------------
   <button data-dialog-open="confirm">Delete…</button>
   <dialog class="dlg" id="confirm"><form method="dialog">… <button value="ok">OK</button></form></dialog>
   Clicking the backdrop closes it ("dismiss"); listen for the native "close" event and read dialog.returnValue. */
function initDialog(d) {
  d.addEventListener("click", (e) => {
    if (e.target !== d) return;
    const r = d.getBoundingClientRect();
    const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    if (!inside) d.close("dismiss");
  });
}

/* ---- toast ----------------------------------------------------------------------------
   UI.toast("Saved", {title, kind: "cost"|"trap"|"why"|"mech"|"info", timeout: ms|0, action: {label, run}})
   or declaratively: <button data-toast="Index rebuilt" data-toast-kind="cost" data-toast-title="Done"> */
function toast(msg, opts = {}) {
  let region = $(".toasts");
  if (!region) {
    region = doc.createElement("section");
    region.className = "toasts";
    region.setAttribute("aria-label", "Notifications");
    region.setAttribute("aria-live", "polite");
    doc.body.append(region);
  }
  const t = doc.createElement("div");
  t.className = "toast";
  t.setAttribute("role", opts.kind === "trap" ? "alert" : "status");
  if (opts.kind) t.dataset.kind = opts.kind;
  t.innerHTML = `<div>${opts.title ? `<b>${esc(opts.title)}</b>` : ""}<p>${esc(msg)}</p></div>`
    + '<button type="button" class="toast-x" aria-label="Dismiss notification">&times;</button>'
    + (opts.action ? `<button type="button" class="btn btn-s toast-a">${esc(opts.action.label)}</button>` : "");
  region.append(t);
  let timer = null;
  const close = () => { clearTimeout(timer); t.classList.add("out"); setTimeout(() => t.remove(), reducedMotion() ? 0 : 320); };
  const arm = (ms) => { if (opts.timeout !== 0) timer = setTimeout(close, ms); };
  arm(opts.timeout || 4600);
  t.addEventListener("mouseenter", () => clearTimeout(timer));
  t.addEventListener("mouseleave", () => arm(2200));
  $(".toast-x", t).addEventListener("click", close);
  if (opts.action) $(".toast-a", t).addEventListener("click", () => { close(); opts.action.run(); });
  return { el: t, close };
}

/* ---- tooltip: any element with data-tip="text" (optional data-tip-kbd="Ctrl K") ---------- */
let tipEl = null, tipFor = null, tipTimer = null;
function showTip(target) {
  if (!tipEl) {
    tipEl = doc.createElement("div");
    tipEl.className = "tip";
    tipEl.id = newId("tip");
    tipEl.setAttribute("role", "tooltip");
    doc.body.append(tipEl);
  }
  tipFor = target;
  tipEl.innerHTML = esc(target.dataset.tip) + (target.dataset.tipKbd ? target.dataset.tipKbd.split(" ").map((k) => `<kbd>${esc(k)}</kbd>`).join("") : "");
  if (!target.hasAttribute("aria-label") && !target.hasAttribute("aria-describedby")) target.setAttribute("aria-describedby", tipEl.id);
  tipEl.classList.remove("on");
  tipEl.style.left = "0px"; tipEl.style.top = "0px";
  const a = target.getBoundingClientRect(), w = tipEl.offsetWidth, h = tipEl.offsetHeight;
  const right = target.closest(".rail") && target.closest('.shell[data-rail="collapsed"]');
  let top = right ? a.top + a.height / 2 - h / 2 : a.top - h - 8;
  let left = right ? a.right + 10 : a.left + a.width / 2 - w / 2;
  if (!right && top < 6) top = a.bottom + 8;
  left = Math.min(Math.max(6, left), innerWidth - w - 6);
  tipEl.style.left = `${left}px`; tipEl.style.top = `${top}px`;
  tipEl.classList.add("on");
}
function hideTip() {
  clearTimeout(tipTimer);
  if (!tipEl || !tipFor) return;
  tipEl.classList.remove("on");
  if (tipFor.getAttribute("aria-describedby") === tipEl.id) tipFor.removeAttribute("aria-describedby");
  tipFor = null;
}

/* ---- popover positioning: native [popover] + popovertarget; .pop is placed under its invoker.
   <button popovertarget="m1">Options</button><div popover id="m1" class="pop menu" data-place="bottom|top" data-align="start|end">… */
function place(el, anchor, side = "bottom", align = "start") {
  const a = anchor.getBoundingClientRect(), w = el.offsetWidth, h = el.offsetHeight, m = 8;
  let top = side === "top" ? a.top - h - 6 : a.bottom + 6;
  if (side !== "top" && top + h > innerHeight - m && a.top - h - 6 > m) top = a.top - h - 6;
  if (side === "top" && top < m) top = a.bottom + 6;
  let left = align === "end" ? a.right - w : a.left;
  left = Math.min(Math.max(m, left), innerWidth - w - m);
  el.style.top = `${top}px`;
  el.style.left = `${left}px`;
}

/* ---- glossary --------------------------------------------------------------------------
   Term: <button class="gl" data-term="kv-cache">KV cache</button>
   Definitions: <dl class="glossary"><div id="g-kv-cache"><dt>KV cache</dt><dd>…</dd></div></dl>
   (or window.GLOSSARY = {"kv-cache": {t: "KV cache", d: "html"}}). */
let glossReady = false;
function initGlossary() {
  const terms = $$("[data-term]");
  if (glossReady || !terms.length) return;
  glossReady = true;
  const defs = {};
  $$(".glossary > div[id]").forEach((d) => { defs[d.id.replace(/^g-/, "")] = { t: $("dt", d).textContent, d: $("dd", d).innerHTML, href: `#${d.id}` }; });
  Object.assign(defs, window.GLOSSARY || {});
  const missing = terms.map((t) => t.dataset.term).filter((k) => !defs[k]);
  if (missing.length) throw new Error(`glossary: no definition for ${[...new Set(missing)].join(", ")}`);
  const pop = doc.createElement("div");
  pop.className = "gl-pop";
  pop.id = newId("gl-pop");
  pop.setAttribute("role", "dialog");
  pop.hidden = true;
  doc.body.append(pop);
  let owner = null;
  const close = (refocus) => {
    if (!owner) return;
    owner.setAttribute("aria-expanded", "false");
    if (refocus) owner.focus();
    owner = null;
    pop.hidden = true;
  };
  const open = (t) => {
    const d = defs[t.dataset.term];
    owner = t;
    t.setAttribute("aria-expanded", "true");
    pop.setAttribute("aria-label", d.t);
    pop.innerHTML = `<span class="gp-t">${esc(d.t)}</span><p>${d.d}</p>${d.href && !t.closest(".glossary") ? `<p class="gp-w"><a href="${d.href}">In the glossary</a></p>` : ""}`;
    pop.hidden = false;
    const r = t.getBoundingClientRect();
    const w = pop.offsetWidth;
    pop.style.left = `${Math.min(Math.max(12, r.left + scrollX), scrollX + innerWidth - w - 12)}px`;
    pop.style.top = `${r.bottom + scrollY + 8}px`;
  };
  terms.forEach((t) => {
    if (t.tagName !== "BUTTON") { t.setAttribute("role", "button"); t.tabIndex = 0; } else t.type = "button";
    t.setAttribute("aria-expanded", "false");
    t.setAttribute("aria-controls", pop.id);
  });
  doc.addEventListener("click", (e) => {
    const t = e.target.closest("[data-term]");
    if (t) { const same = owner === t; close(); if (!same) open(t); return; }
    if (owner && !pop.contains(e.target)) close();
  });
  doc.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && owner) close(true);
    const t = e.target.closest?.("[data-term]");
    if (t && t.tagName !== "BUTTON" && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); t.click(); }
  });
}

/* ---- command palette (Ctrl/Cmd+K, or any [data-ui=cmdk-open]) ----------------------------
   Built-in commands: jump to every section, switch theme, toggle the rail, print.
   Add your own: UI.registerCommands([{title, group, hint, icon, run}]). */
const extraCommands = [];
function registerCommands(list) { extraCommands.push(...list); }
function builtinCommands() {
  const cmds = $$("main section[id]").filter((s) => $("h2", s)).map((s) => ({
    group: "Go to", title: $("h2", s).textContent.trim(), hint: sectionLabel(s), icon: "hash",
    run: () => { s.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth" }); s.classList.add("flash"); setTimeout(() => s.classList.remove("flash"), 1900); },
  }));
  THEMES.forEach((t) => cmds.push({ group: "Theme", title: `Theme: ${t.label}`, hint: t.scheme, icon: "palette", run: () => setTheme(t.id) }));
  cmds.push({ group: "Theme", title: "Theme: follow system", icon: "palette", run: () => setTheme("system") });
  if ($("[data-rail-toggle]")) cmds.push({ group: "View", title: "Toggle sidebar", icon: "panel-left", run: () => $("[data-rail-toggle]").click() });
  cmds.push({ group: "View", title: "Print or save as PDF", icon: "printer", run: () => toast("Press Ctrl+P (Cmd+P on a Mac). The page has print styles; choose Save as PDF as the printer.", { title: "Print", kind: "info" }) });
  return [...extraCommands, ...cmds];
}
let palette = null;
function openPalette() {
  if (!palette) palette = buildPalette();
  palette.open();
}
function buildPalette() {
  const d = doc.createElement("dialog");
  d.className = "cmdk";
  d.setAttribute("aria-label", "Command palette");
  const lid = newId("cmdk-list");
  d.innerHTML = `<div class="cmdk-in">${icon("search")}<input type="text" role="combobox" aria-expanded="true" aria-controls="${lid}" aria-autocomplete="list" placeholder="Jump to a section, switch theme, run a command" spellcheck="false" autocomplete="off"></div>`
    + `<ul class="cmdk-list" id="${lid}" role="listbox" aria-label="Commands"></ul>`
    + '<div class="cmdk-f"><span><kbd>&uarr;</kbd><kbd>&darr;</kbd> move</span><span><kbd>Enter</kbd> run</span><span><kbd>Esc</kbd> close</span></div>';
  doc.body.append(d);
  initDialog(d);
  const input = $("input", d), list = $("ul", d);
  let cmds = [], shown = [], sel = 0;
  const score = (c, q) => {
    const s = `${c.title} ${c.hint || ""} ${c.group || ""}`.toLowerCase();
    if (!q) return 1;
    if (c.title.toLowerCase().startsWith(q)) return 4;
    if (new RegExp(`\\b${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`).test(s)) return 3;
    if (s.includes(q)) return 2;
    let i = 0; for (const ch of s) if (ch === q[i]) i++;
    return i === q.length ? 1 : 0;
  };
  const hl = (text, q) => {
    const i = q ? text.toLowerCase().indexOf(q) : -1;
    return i < 0 ? esc(text) : esc(text.slice(0, i)) + `<mark>${esc(text.slice(i, i + q.length))}</mark>` + esc(text.slice(i + q.length));
  };
  const render = () => {
    const q = input.value.trim().toLowerCase();
    shown = cmds.map((c, i) => ({ c, s: score(c, q), i })).filter((x) => x.s > 0)
      .sort((a, b) => (q ? b.s - a.s : 0) || a.i - b.i).map((x) => x.c);
    if (q) shown = shown.slice(0, 40);
    sel = Math.min(sel, Math.max(0, shown.length - 1));
    let html = "", group = null;
    shown.forEach((c, i) => {
      if (!q && c.group !== group) { group = c.group; html += `<li class="cmdk-g" role="presentation">${esc(group || "")}</li>`; }
      html += `<li class="cmdk-o" role="option" id="${lid}-${i}" data-i="${i}" aria-selected="${i === sel}">${c.icon ? icon(c.icon) : ""}<span class="cmdk-t">${hl(c.title, q)}</span>${c.hint ? `<span class="cmdk-h">${esc(c.hint)}</span>` : ""}</li>`;
    });
    list.innerHTML = html || '<li class="cmdk-empty" role="presentation">No matching command</li>';
    if (shown.length) input.setAttribute("aria-activedescendant", `${lid}-${sel}`); else input.removeAttribute("aria-activedescendant");
    $(`#${lid}-${sel}`, list)?.scrollIntoView({ block: "nearest" });
  };
  const run = (i) => { const c = shown[i]; if (!c) return; d.close(); c.run(); };
  input.addEventListener("input", () => { sel = 0; render(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + shown.length) % Math.max(1, shown.length); render(); }
    else if (e.key === "Enter") { e.preventDefault(); run(sel); }
  });
  list.addEventListener("click", (e) => { const o = e.target.closest(".cmdk-o"); if (o) run(+o.dataset.i); });
  list.addEventListener("mousemove", (e) => { const o = e.target.closest(".cmdk-o"); if (o && +o.dataset.i !== sel) { sel = +o.dataset.i; render(); } });
  return {
    open() { cmds = builtinCommands(); input.value = ""; sel = 0; render(); d.showModal(); input.focus(); },
    toggle() { if (d.open) d.close(); else this.open(); },
  };
}

/* ---- search with highlight -----------------------------------------------------------------
   <div class="search" data-ui="search" data-scope="main"><label class="vh" for="q">Search</label>
   <input id="q" type="search" placeholder="Search this page"><kbd class="q-key">/</kbd></div>
   Press / anywhere to focus. Matches are listed with their section and highlighted in the page.
   Put data-nosearch on any block that should not be indexed (e.g. markup samples). */
function initSearch(box) {
  const input = $("input", box);
  if (!input) throw new Error("search: missing <input>");
  const scope = $(box.dataset.scope || "main");
  if (!scope) throw new Error(`search: scope "${box.dataset.scope || "main"}" not found`);
  let res = $(".q-res", box);
  if (!res) { res = doc.createElement("div"); res.className = "q-res"; res.hidden = true; box.append(res); }
  res.id ||= newId("q-res");
  input.setAttribute("aria-controls", res.id);
  input.setAttribute("autocomplete", "off");
  let index = null, timer = null;
  const BLOCKS = "p,li,h2,h3,h4,td,th,dd,dt,pre,figcaption,blockquote,summary";
  const build = () => $$(BLOCKS, scope)
    .filter((el) => !el.closest(".fc-src,.q-res,[aria-hidden=true],[data-nosearch]") && !el.querySelector("p,li,pre,dd") && el.textContent.trim())
    .map((el) => ({ el, text: el.textContent.replace(/\s+/g, " ").trim(), sec: el.closest("section[id]") }));
  const highlight = (q) => {
    if (!window.CSS || !CSS.highlights) return;
    CSS.highlights.delete("arsenal-search");
    if (!q) return;
    const ranges = [], walk = doc.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
    for (let n = walk.nextNode(); n; n = walk.nextNode()) {
      const t = n.nodeValue.toLowerCase();
      for (let i = t.indexOf(q); i >= 0; i = t.indexOf(q, i + q.length)) {
        const r = new Range(); r.setStart(n, i); r.setEnd(n, i + q.length); ranges.push(r);
      }
    }
    CSS.highlights.set("arsenal-search", new Highlight(...ranges));
  };
  const run = () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { res.hidden = true; res.innerHTML = ""; highlight(""); return; }
    index ||= build();
    const hits = index.filter((x) => x.text.toLowerCase().includes(q));
    const secs = new Set(hits.map((h) => h.sec));
    const snip = (t) => {
      const i = t.toLowerCase().indexOf(q), a = Math.max(0, i - 50), b = Math.min(t.length, i + q.length + 70);
      return (a ? "…" : "") + esc(t.slice(a, i)) + `<mark>${esc(t.slice(i, i + q.length))}</mark>` + esc(t.slice(i + q.length, b)) + (b < t.length ? "…" : "");
    };
    res.innerHTML = `<p class="q-count" role="status">${hits.length ? `${hits.length} match${hits.length > 1 ? "es" : ""} in ${secs.size} section${secs.size > 1 ? "s" : ""}` : "No matches"}</p>`
      + `<ol>${hits.slice(0, 40).map((h, i) => `<li><button type="button" class="qr" data-i="${index.indexOf(h)}"><span class="qr-sec">${esc(sectionLabel(h.sec) || "page")}</span><span class="qr-snip">${snip(h.text)}</span></button></li>`).join("")}</ol>`;
    res.hidden = false;
    highlight(q);
  };
  input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 110); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { input.value = ""; run(); input.blur(); }
    if (e.key === "ArrowDown") { const f = $(".qr", res); if (f) { e.preventDefault(); f.focus(); } }
  });
  res.addEventListener("keydown", (e) => {
    const all = $$(".qr", res), i = all.indexOf(doc.activeElement);
    if (e.key === "ArrowDown" && i < all.length - 1) { e.preventDefault(); all[i + 1].focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); (i > 0 ? all[i - 1] : input).focus(); }
    if (e.key === "Escape") input.focus();
  });
  res.addEventListener("click", (e) => {
    const b = e.target.closest(".qr");
    if (!b) return;
    const el = index[+b.dataset.i].el;
    el.closest("details")?.setAttribute("open", "");
    el.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 1900);
  });
}

/* ---- sortable table ---------------------------------------------------------------------------
   <table data-ui="sort">…</table>; a cell may carry data-v="0.731" as its sort value;
   <th data-sort="none"> opts a column out. Numbers (incl. 1,204 · 12% · −3.1 · 85 ms) sort numerically. */
function initSort(table) {
  const ths = $$("thead th", table);
  const body = table.tBodies[0];
  if (!body) throw new Error("sort: table needs a <tbody>");
  const num = (raw) => {
    const s = raw.replace(/−/g, "-").replace(/[,\s%$€£]|ms$|s$|x$/gi, "");
    return /^[-+]?\d*\.?\d+(e[-+]?\d+)?$/i.test(s) ? parseFloat(s) : null;
  };
  const sortBy = (i, th) => {
    const dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
    ths.forEach((x) => x.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", dir);
    const rows = Array.from(body.rows).map((r, k) => {
      const c = r.cells[i], raw = (c?.dataset.v ?? c?.textContent ?? "").trim();
      return { r, k, raw, n: num(raw) };
    });
    const numeric = rows.every((x) => x.raw === "" || x.n !== null);
    const sgn = dir === "ascending" ? 1 : -1;
    rows.sort((a, b) => (numeric ? (a.n ?? -Infinity) - (b.n ?? -Infinity) : a.raw.localeCompare(b.raw, undefined, { numeric: true })) * sgn || a.k - b.k);
    body.append(...rows.map((x) => x.r));
  };
  ths.forEach((th, i) => {
    if (th.dataset.sort === "none") return;
    const b = doc.createElement("button");
    b.type = "button";
    b.className = "th-sort";
    b.append(...th.childNodes);
    th.append(b);
    b.addEventListener("click", () => sortBy(i, th));
  });
}

/* ---- step-through diagram -------------------------------------------------------------------
   <figure class="stp" data-ui="steps" data-interval="2600">
     <div class="stp-head"><span class="kicker">step-through</span><h3>Title</h3></div>
     <div class="stp-body"><div class="stp-stage"><svg>… parts with data-on="1 3" data-from="2" data-until="4" data-dim="5" …</svg></div>
       <div class="stp-side"><ol class="stp-steps"><li><h4>Step title</h4><p>What happens.</p></li>…</ol></div></div>
     <figcaption>…</figcaption></figure>
   Sets data-step on the figure (CSS hook) and emits "stepchange" {step, total}. Keys: ← → Home End, Space plays. */
function initSteps(fig) {
  const stage = $(".stp-stage", fig), side = $(".stp-side", fig), list = $(".stp-steps", fig);
  if (!stage || !side || !list) throw new Error("steps: needs .stp-stage, .stp-side and .stp-steps");
  const items = $$(":scope > li", list), n = items.length;
  if (!n) throw new Error("steps: .stp-steps has no <li>");
  items.forEach((li, i) => { if (!$(".stp-n", li)) li.insertAdjacentHTML("afterbegin", `<p class="stp-n">Step ${i + 1} of ${n}</p>`); });
  list.setAttribute("aria-live", "polite");
  const ctl = doc.createElement("div");
  ctl.className = "stp-ctl";
  ctl.innerHTML = `<button type="button" class="ib" data-act="first" aria-label="First step">${icon("skip-back")}</button>`
    + `<button type="button" class="ib" data-act="prev" aria-label="Previous step">${icon("chevron-left")}</button>`
    + `<button type="button" class="ib ib-a" data-act="play" aria-label="Play">${icon("play")}</button>`
    + `<button type="button" class="ib" data-act="next" aria-label="Next step">${icon("chevron-right")}</button>`
    + '<span class="stp-count"></span>';
  side.insertBefore(ctl, list);
  const dots = doc.createElement("div");
  dots.className = "stp-dots";
  dots.setAttribute("role", "group");
  dots.setAttribute("aria-label", "Steps");
  dots.innerHTML = items.map((_, i) => `<button type="button" aria-label="Step ${i + 1}"></button>`).join("");
  side.append(dots);
  stage.tabIndex = 0;
  if (!stage.hasAttribute("aria-label")) stage.setAttribute("aria-label", "Diagram. Use the arrow keys to step through.");
  const parts = $$("[data-on],[data-from],[data-until],[data-dim]", stage);
  const nums = (s) => (s || "").split(/[\s,]+/).filter(Boolean).map(Number);
  let cur = 0, timer = null;
  const seen = new Set();
  const go = (i) => {
    cur = Math.max(0, Math.min(n - 1, i));
    const step = cur + 1;
    seen.add(cur);
    items.forEach((li, j) => { li.hidden = j !== cur; });
    fig.dataset.step = String(step);
    $(".stp-count", ctl).textContent = `${step} / ${n}`;
    $$("button", dots).forEach((b, j) => { b.classList.toggle("seen", seen.has(j)); if (j === cur) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current"); });
    $('[data-act="first"]', ctl).disabled = $('[data-act="prev"]', ctl).disabled = cur === 0;
    $('[data-act="next"]', ctl).disabled = cur === n - 1;
    parts.forEach((p) => {
      const on = nums(p.dataset.on), from = +p.dataset.from || 1, until = +p.dataset.until || n;
      p.classList.toggle("is-on", on.includes(step));
      p.classList.toggle("is-past", !on.includes(step) && on.some((s) => s < step));
      p.classList.toggle("is-off", step < from || step > until);
      p.classList.toggle("is-dim", nums(p.dataset.dim).includes(step));
    });
    emit(fig, "stepchange", { step, total: n });
  };
  const setPlay = (on) => {
    const b = $('[data-act="play"]', ctl);
    b.innerHTML = icon(on ? "pause" : "play");
    b.setAttribute("aria-label", on ? "Pause" : "Play");
  };
  const stop = () => { clearInterval(timer); timer = null; setPlay(false); };
  const play = () => {
    if (timer) return stop();
    if (cur === n - 1) go(0);
    setPlay(true);
    timer = setInterval(() => { if (cur >= n - 1) stop(); else go(cur + 1); }, +fig.dataset.interval || 2600);
  };
  ctl.addEventListener("click", (e) => {
    const a = e.target.closest("[data-act]")?.dataset.act;
    if (!a) return;
    if (a === "play") return play();
    stop();
    go({ first: 0, prev: cur - 1, next: cur + 1 }[a]);
  });
  dots.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { stop(); go($$("button", dots).indexOf(b)); } });
  fig.addEventListener("keydown", (e) => {
    if (isTyping(e.target)) return;
    const to = { ArrowRight: cur + 1, ArrowLeft: cur - 1, Home: 0, End: n - 1 }[e.key];
    if (to !== undefined) { e.preventDefault(); stop(); go(to); }
    else if (e.key === " " && e.target === stage) { e.preventDefault(); play(); }
  });
  go(0);
}

/* ---- flashcards (Leitner boxes, saved per page) ---------------------------------------------
   <div data-ui="flashcards" id="drills"><ol class="fc-src">
     <li data-band="mechanism"><p class="q">Question?</p><div class="a"><p>Model answer.</p></div></li>…</ol></div>
   Keys on the card: Space reveals; 1 missed · 2 shaky · 3 knew it. Missed cards come back first. */
function initFlashcards(el) {
  const cards = $$(".fc-src > li", el).map((li) => {
    const q = $(".q", li), a = $(".a", li);
    if (!q || !a) throw new Error("flashcards: each <li> needs .q and .a");
    return { q: q.innerHTML, a: a.innerHTML, band: li.dataset.band || "" };
  });
  if (!cards.length) throw new Error("flashcards: no cards in .fc-src");
  const key = `arsenal-fc:${location.pathname}:${el.id || "cards"}`;
  let box = store.json(key, null);
  if (!Array.isArray(box) || box.length !== cards.length) box = cards.map(() => -1);
  let cur = 0, shown = false;
  const NAMES = ["missed", "shaky", "good", "known"];
  const ui = doc.createElement("div");
  ui.className = "fc";
  el.append(ui);
  const pick = () => {
    const order = [0, -1, 1, 2, 3];
    for (const b of order) {
      for (let k = 1; k <= cards.length; k++) { const i = (cur + k) % cards.length; if (box[i] === b) return i; }
    }
    return (cur + 1) % cards.length;
  };
  const render = () => {
    const count = (b) => box.filter((x) => x === b).length;
    const c = cards[cur];
    ui.innerHTML = `<div class="fc-top"><div class="fc-boxes">`
      + `<span class="bxn"><i></i>new <b>${count(-1)}</b></span>`
      + NAMES.map((nm, b) => `<span class="bx${b}"><i></i>${nm} <b>${count(b)}</b></span>`).join("")
      + `</div><div class="fc-map" role="group" aria-label="Cards">${cards.map((_, i) => `<button type="button" class="fcm ${box[i] >= 0 ? `seen bx${box[i]}` : "bxn"}${i === cur ? " cur" : ""}" aria-label="Card ${i + 1}${box[i] >= 0 ? `, ${NAMES[box[i]]}` : ", new"}" data-i="${i}"></button>`).join("")}</div></div>`
      + `<div class="fc-card" tabindex="0" aria-label="Flashcard ${cur + 1} of ${cards.length}">`
      + (c.band ? `<span class="fc-band">${esc(c.band)}</span>` : "")
      + `<div class="fc-q">${c.q}</div>`
      + (shown
        ? `<div class="fc-a">${c.a}</div><div class="fc-grade"><span>How did it go?</span>`
          + `<button type="button" class="gb" data-g="0"><kbd>1</kbd>Missed it</button><button type="button" class="gb" data-g="1"><kbd>2</kbd>Shaky</button><button type="button" class="gb" data-g="2"><kbd>3</kbd>Knew it</button></div>`
        : '<button type="button" class="btn btn-a fc-reveal">Show the answer <kbd>Space</kbd></button>')
      + `</div><div class="fc-foot"><span>${cards.length - count(-1)} of ${cards.length} graded · saved in this browser only</span><button type="button" class="btn btn-s" data-fc="reset">Reset</button></div>`;
  };
  const grade = (g) => {
    box[cur] = g === 0 ? 0 : g === 1 ? 1 : (box[cur] >= 2 ? 3 : 2);
    store.set(key, JSON.stringify(box));
    emit(el, "fcgrade", { card: cur, box: box[cur] });
    cur = pick(); shown = false; render();
    $(".fc-card", ui).focus();
  };
  ui.addEventListener("click", (e) => {
    const t = e.target;
    if (t.closest(".fc-reveal")) { shown = true; render(); $(".fc-card", ui).focus(); }
    else if (t.closest(".gb")) grade(+t.closest(".gb").dataset.g);
    else if (t.closest(".fcm")) { cur = +t.closest(".fcm").dataset.i; shown = false; render(); }
    else if (t.closest('[data-fc="reset"]')) { box = cards.map(() => -1); store.set(key, JSON.stringify(box)); cur = 0; shown = false; render(); }
  });
  ui.addEventListener("keydown", (e) => {
    if (!e.target.closest(".fc-card")) return;
    if (!shown && (e.key === " " || e.key === "Enter")) { e.preventDefault(); shown = true; render(); $(".fc-card", ui).focus(); }
    else if (shown && ["1", "2", "3"].includes(e.key)) grade(+e.key - 1);
  });
  render();
}

/* ---- annotated code study ---------------------------------------------------------------------
   <figure class="study" data-ui="study"><div class="study-head">…</div>
     <pre class="code-pane" data-start="40">plain code</pre>
     <ol class="notes"><li data-kind="why" data-at="3-5"><h4>Title</h4><p>Body</p></li>…</ol></figure>
   data-at is relative to the excerpt (1 = first line). Hovering or focusing a note lights its lines and back. */
function initStudy(fig) {
  const pre = $("pre", fig), notes = $$(".notes > li", fig);
  if (!pre) throw new Error("study: missing <pre>");
  const start = +(pre.dataset.start || 1);
  const lines = pre.textContent.replace(/\n$/, "").split("\n");
  const owners = {};
  notes.forEach((li, i) => {
    const [a, b = a] = (li.dataset.at || "").split("-").map(Number);
    if (!a || a < 1 || b > lines.length || b < a) throw new Error(`study: note ${i + 1} data-at="${li.dataset.at}" is outside lines 1-${lines.length}`);
    for (let L = a; L <= b; L++) (owners[L] ||= []).push(i);
    li.classList.add("note");
    li.tabIndex = 0;
    li.dataset.note = String(i);
    const h = $("h4", li);
    const at = a === b ? `line ${start + a - 1}` : `lines ${start + a - 1}–${start + b - 1}`;
    const top = doc.createElement("div");
    top.className = "note-top";
    top.innerHTML = `<span class="tag">${esc(li.dataset.kind || "note")}</span>`;
    if (h) top.append(h);
    top.insertAdjacentHTML("beforeend", `<span class="note-at">${at}</span>`);
    li.prepend(top);
  });
  pre.classList.add("code-pane");
  pre.innerHTML = lines.map((ln, i) => {
    const own = owners[i + 1];
    const kind = own ? ` data-kind="${esc(notes[own[0]].dataset.kind || "")}" data-notes="${own.join(" ")}"` : "";
    return `<div class="cl"${kind}><span class="ln">${start + i}</span><span class="lc">${esc(ln) || " "}</span></div>`;
  }).join("");
  if (!pre.parentElement.classList.contains("study-body")) {
    const body = doc.createElement("div");
    body.className = "study-body";
    pre.before(body);
    body.append(pre, $(".notes", fig));
  }
  const hot = (i, on) => $$(`.note[data-note="${i}"], .cl[data-notes~="${i}"]`, fig).forEach((x) => x.classList.toggle("hot", on));
  $$(".note, .cl[data-notes]", fig).forEach((el) => {
    const i = el.dataset.note ?? el.dataset.notes.split(" ")[0];
    ["mouseenter", "focus"].forEach((ev) => el.addEventListener(ev, () => hot(i, true)));
    ["mouseleave", "blur"].forEach((ev) => el.addEventListener(ev, () => hot(i, false)));
  });
}

/* ---- clipboard: UI.copyText(text, selectEl?) -------------------------------------------------------
   Clipboard writes reject in sandboxed frames (published artifacts) and insecure contexts. On failure the
   text is selected instead (selectEl, or a toast showing the text) so Ctrl+C still works. */
function copyText(text, selectEl) {
  const write = navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject(new Error("clipboard unavailable"));
  return write.then(() => true, () => {
    let target = selectEl;
    if (!target) target = $("p", toast(text, { title: "Clipboard blocked: text selected, press Ctrl+C", kind: "info", timeout: 0 }).el);
    getSelection().selectAllChildren(target);
    if (selectEl) toast("Clipboard is blocked here. The text is selected: press Ctrl+C.", { kind: "info" });
    return false;
  });
}

/* ---- copy buttons: <pre data-copy> or any <div class="code"><pre>…</pre></div> ------------------- */
function initCopy(pre) {
  let wrap = pre.closest(".code");
  if (!wrap) { wrap = doc.createElement("div"); wrap.className = "code"; pre.before(wrap); wrap.append(pre); }
  const b = doc.createElement("button");
  b.type = "button";
  b.className = "copy";
  b.textContent = "Copy";
  b.setAttribute("aria-label", "Copy code");
  (($(".code-h", wrap)) || wrap).append(b);
  b.addEventListener("click", () => copyText(pre.textContent, $("code", pre) || pre).then((ok) => {
    if (!ok) return;
    b.textContent = "Copied";
    setTimeout(() => { b.textContent = "Copy"; }, 1400);
  }));
}

/* ---- docs TOC: <nav class="toc" data-ui="toc" data-scope="article"></nav> (h2/h3 with or without ids) */
function initToc(nav) {
  const scope = $(nav.dataset.scope || "article") || $("main");
  const hs = $$("h2, h3", scope).filter((h) => !h.closest(".stp, .study, .acc, .dlg, .card"));
  hs.forEach((h) => { if (!h.id) { let s = slug(h.textContent), k = 2; while (doc.getElementById(s)) s = `${slug(h.textContent)}-${k++}`; h.id = s; } });
  nav.innerHTML = `${nav.dataset.title === "" ? "" : `<span class="rail-k">${esc(nav.dataset.title || "On this page")}</span>`}<ol>${hs.map((h) => `<li class="l${h.tagName[1]}"><a href="#${h.id}">${esc(h.textContent.trim())}</a></li>`).join("")}</ol>`;
  scrollSpy($$("a", nav));
}

/* ---- annotation-kind filter (Atlas legend) --------------------------------------------------------
   <div data-ui="kind-filter" data-scope="main"><button class="kchip" data-kind="why"><span class="tag">why</span>the decision</button>…</div> */
function initKindFilter(el) {
  const scope = $(el.dataset.scope || "main");
  if (!scope) throw new Error("kind-filter: scope not found");
  el.setAttribute("role", "group");
  const chips = $$(".kchip[data-kind]", el);
  chips.forEach((c) => { c.type = "button"; c.setAttribute("aria-pressed", "false"); });
  el.addEventListener("click", (e) => {
    const c = e.target.closest(".kchip");
    if (!c) return;
    const on = c.getAttribute("aria-pressed") !== "true";
    chips.forEach((x) => x.setAttribute("aria-pressed", String(on && x === c)));
    if (on) scope.setAttribute("data-kind-filter", c.dataset.kind); else scope.removeAttribute("data-kind-filter");
    $$("[data-kind]", scope).forEach((k) => k.classList.toggle("kind-out", on && k.dataset.kind !== c.dataset.kind && !k.closest("[data-ui=kind-filter]")));
  });
}

/* ---- view transition helper: UI.vt(() => reorderTheDom()) — animates when supported and allowed */
function vt(update) {
  if (doc.startViewTransition && !reducedMotion()) return doc.startViewTransition(update);
  update();
  return null;
}

/* ---- <kbd data-kbd-mod>Ctrl</kbd> shows the platform's modifier (Cmd on Apple devices) */
const isApple = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
function initKbdMod(k) { k.textContent = isApple ? "⌘" : "Ctrl"; }

/* ---- live counts: <span data-count="section.sec"></span> → how many elements match (never hardcode counts) */
function initCount(el) {
  el.textContent = String($$(el.dataset.count).length);
}

/* ---- global wiring (once) ------------------------------------------------------------------------- */
doc.addEventListener("click", (e) => {
  const o = e.target.closest("[data-dialog-open]");
  if (o) {
    const d = doc.getElementById(o.dataset.dialogOpen);
    if (!d) throw new Error(`dialog: no element with id "${o.dataset.dialogOpen}"`);
    d.showModal();
  }
  const c = e.target.closest("[data-dialog-close]");
  if (c) c.closest("dialog").close(c.value || "");
  const t = e.target.closest("[data-toast]");
  if (t) toast(t.dataset.toast, { kind: t.dataset.toastKind, title: t.dataset.toastTitle });
  if (e.target.closest("[data-ui=cmdk-open]")) openPalette();
});
doc.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    if (!palette) palette = buildPalette();
    palette.toggle();
  } else if (e.key === "/" && !isTyping(doc.activeElement)) {
    const q = $("[data-ui=search] input");
    if (q) { e.preventDefault(); q.focus(); q.select(); }
  } else if (e.key === "Escape") hideTip();
});
doc.addEventListener("pointerover", (e) => {
  const t = e.target.closest?.("[data-tip]");
  if (!t || t === tipFor) return;
  clearTimeout(tipTimer);
  tipTimer = setTimeout(() => showTip(t), 280);
});
doc.addEventListener("pointerout", (e) => {
  const t = e.target.closest?.("[data-tip]");
  if (t && !t.contains(e.relatedTarget)) hideTip();
});
doc.addEventListener("focusin", (e) => { const t = e.target.closest?.("[data-tip]"); if (t && t.matches(":focus-visible")) showTip(t); });
doc.addEventListener("focusout", hideTip);
addEventListener("scroll", hideTip, { passive: true });
doc.addEventListener("toggle", (e) => {
  const p = e.target;
  if (e.newState !== "open" || !p.matches?.("[popover]") || !p.id) return;
  const inv = $(`[popovertarget="${CSS.escape(p.id)}"]`);
  if (inv) place(p, inv, p.dataset.place, p.dataset.align);
}, true);

const INIT = [
  ["[data-icon]", initIcon], ["[data-ui=theme-picker]", initThemePicker], ["[data-ui=theme-toggle]", initThemeToggle],
  ["[data-ui=rail]", initRail], ["[data-ui=tabs]", initTabs], ["[data-ui=seg]", initSeg], ["[data-expand-all]", initExpandAll],
  ["dialog.dlg", initDialog], ["[data-ui=sort]", initSort], ["[data-ui=steps]", initSteps], ["[data-ui=flashcards]", initFlashcards],
  ["[data-ui=study]", initStudy], ["pre[data-copy], .code > pre", initCopy], ["[data-ui=search]", initSearch],
  ["[data-ui=toc]", initToc], ["[data-ui=kind-filter]", initKindFilter], ["[data-count]", initCount], ["[data-kbd-mod]", initKbdMod],
];
const done = new WeakMap();
function init(scope = doc) {
  for (const [sel, fn] of INIT) {
    const els = $$(sel, scope);
    if (scope !== doc && scope.matches?.(sel)) els.unshift(scope);
    for (const el of els) {
      const set = done.get(el) || new Set();
      if (set.has(fn)) continue;
      set.add(fn);
      done.set(el, set);
      fn(el);
    }
  }
  initLearn();
  initGlossary();
}

const UI = { init, icon, setTheme, getTheme, themes: THEMES, toast, copyText, openPalette, registerCommands, place, vt };
if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", () => init()); else init();
window.UI = UI;
})();
