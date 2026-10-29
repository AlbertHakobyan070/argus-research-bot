// #/runs — the library's runs plus the in-flight registry (GET /api/runs).
import { get } from "../api.js";
import { esc, LENGTHS, statusHtml, when } from "./common.js";

export async function render(el, _params, ctx) {
  ctx.crumbs([["", "Runs"]]);
  const { runs } = await get.runs();
  const waiting = runs.filter((r) => r.awaiting).length;
  const live = runs.filter((r) => r.in_flight).length;
  const lede = runs.length === 0 ? "No runs yet. Start one here or with /research in the chat."
    : [live ? `${live} running` : "", waiting ? `${waiting} waiting for you` : "", "newest first"].filter(Boolean).join(" · ");
  const rows = runs.map((r) => {
    const facts = [r.run_id, `${LENGTHS[r.length] || r.length || "—"}${r.depth ? ` · ${r.depth}` : ""}`, when(r.updated_at || r.created_at)];
    return `<li><a class="run-row" href="#/run/${esc(r.run_id)}"><span class="run-t">${esc(r.topic || r.run_id)}</span>`
      + `<span class="meta run-m">${esc(facts.join(" · "))}</span><span class="run-st">${statusHtml(r)}</span></a></li>`;
  }).join("");
  el.innerHTML = `<div class="page-head">
      <div><span class="kicker">Runs</span><h1>Your research runs</h1><p>${esc(lede)}</p></div>
      <div class="btn-row"><a class="btn btn-a" href="#/new"><i data-icon="plus"></i>New research</a></div>
    </div>
    <section class="sec" id="runs">
      ${runs.length ? `<ul class="runlist" aria-label="Runs, newest first">${rows}</ul>` : ""}
    </section>`;
}
