// Swipe between tabs on a touch screen. A deliberate sideways swipe on the page moves to the next or the
// previous tab of the innermost visible tab strip (the Media tabs), or, on a page without one, to the
// next or previous section of the console's nav. It clicks that tab, so the hash router, the back
// button and the tab highlight behave as for a tap. Tabs are read from the page, never listed here.
// Touches that belong to something else are left alone: fields, players, sideways scrollers, dialogs,
// text being selected, and the screen edges where the system's and Telegram's back gestures start.
import { $$ } from "./views/common.js";

const EDGE = 24;        // px from either screen edge
const LOCK = 10;        // px of travel before the touch is read as a sideways swipe or a scroll
const MIN = 64;         // px of sideways travel that commits
const SLOPE = 0.58;     // |dy| / |dx| above this (about 30°) is not sideways
const NO_SWIPE = "input,textarea,select,[contenteditable],video,audio,dialog,[popover],.tw,.chips";

/** The tab a swipe of (dx, dy) lands on from tab i of n: i + 1 (finger to the left), i - 1, or -1
 *  when the move is too short, too steep, or runs off the end of the strip. */
export function swipeTarget(n, i, dx, dy) {
  if (i < 0 || Math.abs(dx) < MIN || Math.abs(dy) > Math.abs(dx) * SLOPE) return -1;
  const j = dx < 0 ? i + 1 : i - 1;
  return j < n ? j : -1;
}

const shown = (el) => el.getClientRects().length > 0;

/** The tabs a swipe moves between, and the index of the current one (-1: none is current). */
function strip(view) {
  const list = $$('[role="tablist"]', view).filter(shown).at(-1);   // a nested strip follows its parent's
  if (list) {
    const tabs = $$('[role="tab"]', list).filter(shown);
    return { tabs, i: tabs.findIndex((t) => t.getAttribute("aria-selected") === "true") };
  }
  const tabs = $$("#nav a").filter(shown);
  return { tabs, i: tabs.findIndex((t) => t.classList.contains("on")) };
}

/** True when the touch starts inside something that scrolls sideways itself (a wide table, a code block). */
function scrollsSideways(el, view) {
  for (; el && el !== view; el = el.parentElement) {
    const { overflowX } = getComputedStyle(el);
    if ((overflowX === "auto" || overflowX === "scroll") && el.scrollWidth > el.clientWidth) return true;
  }
  return false;
}

export function initSwipe(view) {
  if (!("ontouchstart" in window)) return;          // touch screens only: a mouse never swipes
  const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  const selecting = () => !getSelection().isCollapsed;
  const settle = () => { view.classList.remove("swiping"); view.style.translate = ""; };   // springs back (CSS)
  let g = null;                                      // the touch being followed
  view.addEventListener("touchstart", (e) => {
    g = null;
    const t = e.touches[0];
    if (e.touches.length !== 1 || t.clientX < EDGE || t.clientX > innerWidth - EDGE) return;
    if (e.target.closest(NO_SWIPE) || scrollsSideways(e.target, view) || selecting()) return;
    g = { x: t.clientX, y: t.clientY, dx: 0, dy: 0, sideways: null, ...strip(view) };
  }, { passive: true });
  view.addEventListener("touchmove", (e) => {
    if (!g) return;
    const t = e.touches[0];
    g.dx = t.clientX - g.x;
    g.dy = t.clientY - g.y;
    if (g.sideways === null) {
      if (Math.hypot(g.dx, g.dy) < LOCK) return;
      g.sideways = Math.abs(g.dy) <= Math.abs(g.dx) * SLOPE;
    }
    // a scroll, a second finger, a scroll the browser already started, a selection: not a swipe
    if (!g.sideways || e.touches.length !== 1 || !e.cancelable || selecting()) { g = null; settle(); return; }
    e.preventDefault();                              // the page does not also scroll under a sideways swipe
    if (still()) return;
    const room = g.i >= 0 && g.tabs[g.dx < 0 ? g.i + 1 : g.i - 1];   // a tab that way: follow; none: resist
    const pull = room ? Math.min(Math.abs(g.dx) * 0.4, 96) : Math.min(Math.abs(g.dx) * 0.12, 28);
    view.classList.add("swiping");
    view.style.translate = `${Math.sign(g.dx) * pull}px 0`;
  }, { passive: false });
  view.addEventListener("touchend", () => {
    if (!g?.sideways) { g = null; return; }
    const { tabs, i, dx, dy } = g;
    g = null;
    const j = swipeTarget(tabs.length, i, dx, dy);
    if (j < 0) { settle(); return; }
    if (!still()) {
      // drop the pull at once (no spring back) and slide the new tab in from the side the finger left
      view.classList.add("swiping");
      view.style.translate = "";
      view.dataset.swipe = j > i ? "next" : "prev";
      view.classList.remove("swipe-in");
      void view.offsetWidth;                         // restart the slide-in animation
      view.classList.add("swipe-in");
      view.classList.remove("swiping");
    }
    tabs[j].click();
  });
  view.addEventListener("touchcancel", () => { g = null; settle(); });
}
