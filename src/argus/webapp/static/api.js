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

export async function api(path, { method = "GET", body } = {}) {
  const headers = { Authorization: `tma ${authData()}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const r = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const json = (r.headers.get("Content-Type") || "").includes("application/json");
  if (!r.ok) {
    const err = json ? (await r.json()).error : null;
    throw new ApiError(r.status, err?.code || `http_${r.status}`, err?.message || `${r.status} ${r.statusText}`);
  }
  return json ? r.json() : r.text();
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
