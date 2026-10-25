// Telegram launch parameters: the signed initData (sent as `Authorization: tma <initData>` on every
// API call) and the route the console opens on. Telegram puts both in the URL fragment
// (`#tgWebAppData=…&tgWebAppVersion=…`); a route may lead it (`#/run/<id>&tgWebAppData=…`). The SDK,
// when it loaded, is the authority. initData is kept in memory only — never stored, logged or put
// in a URL the server sees (the fragment never leaves the browser).

let initData = null;

/** Read the launch parameters once and drop them from the address bar (the route stays). */
export function takeLaunchParams() {
  const raw = location.hash.replace(/^#/, "");
  const amp = raw.indexOf("&");
  const head = amp < 0 ? raw : raw.slice(0, amp);
  const route = head.startsWith("/") ? head : null;
  const data = new URLSearchParams(route === null ? raw : raw.slice(head.length + 1)).get("tgWebAppData") || "";
  initData = window.Telegram?.WebApp?.initData || data;
  if (data) history.replaceState(null, "", `${location.pathname}${location.search}#${route || "/runs"}`);
}

/** The initData to authorise with ("" when the console was not opened from Telegram: the API says so). */
export function authData() {
  if (initData === null) throw new Error("auth.takeLaunchParams() must run first");
  return initData;
}
