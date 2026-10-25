// The console API (spec §2.7). Every call carries `Authorization: tma <initData>`; a non-2xx answer
// throws ApiError with the server's {error: {code, message}}.
import { authData } from "./auth.js";

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

const UNREACHABLE = "the tunnel may have restarted — reopen the console from the menu button";

/** No answer, or one that cannot be read (a dropped tunnel, a proxy's error page): ApiError(0, "network"). */
const network = (what) => new ApiError(0, "network", `${what}; ${UNREACHABLE}`);

export async function api(path, { method = "GET", body } = {}) {
  const headers = { Authorization: `tma ${authData()}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let r;
  try {
    r = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch (e) {
    throw network(`the console could not reach the bot (${e.message})`);
  }
  const json = (r.headers.get("Content-Type") || "").includes("application/json");
  let data;
  try {
    data = json ? await r.json() : await r.text();
  } catch (e) {
    throw network(`the answer to ${method} ${path.split("?")[0]} could not be read (${e.message})`);
  }
  if (!r.ok) {
    const err = json ? data?.error : null;           // the console's own refusals are always JSON
    if (!err) throw network(`the bot did not answer (HTTP ${r.status} ${r.statusText}, not the console's own error)`);
    throw new ApiError(r.status, err.code, err.message);
  }
  return data;
}

const enc = encodeURIComponent;
export const get = {
  me: () => api("/api/me"),
  runs: (limit = 50) => api(`/api/runs?limit=${limit}`),
  run: (id) => api(`/api/runs/${enc(id)}`),
  events: (id, after) => api(`/api/runs/${enc(id)}/events?after=${after}`),
  report: (id) => api(`/api/runs/${enc(id)}/report?format=html`),
  evidence: (id) => api(`/api/runs/${enc(id)}/evidence`),
  engines: () => api("/api/engines"),
  models: (provider) => api(`/api/engines/providers/${enc(provider)}/models`),
  presets: () => api("/api/config/presets"),
};
export const post = {
  start: (body) => api("/api/runs", { method: "POST", body }),
  plan: (id, body) => api(`/api/runs/${enc(id)}/plan`, { method: "POST", body }),
  report: (id, body) => api(`/api/runs/${enc(id)}/report`, { method: "POST", body }),
  cancel: (id) => api(`/api/runs/${enc(id)}/cancel`, { method: "POST" }),
  send: (id, what) => api(`/api/runs/${enc(id)}/send`, { method: "POST", body: { what } }),
};
export const putEngines = (body) => api("/api/engines", { method: "PUT", body });
