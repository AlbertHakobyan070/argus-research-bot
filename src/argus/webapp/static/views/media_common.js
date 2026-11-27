// Shared by the media views (#/media, #/media/item/<id>): formatting, library cards, the tag review panel
// (suggested tags as toggles, confirmed only on Save), the Investigate panel (repos, tools and sites the item
// mentions, checked; attached only on Attach), the device download, and waiting on a job.
import { media } from "../api.js";
import { $, $$, act, esc, openLink, tg, toast } from "./common.js";

export const TYPES = { "": "All", video: "Video", audio: "Audio", transcript: "Transcripts" };
const TYPE_LABEL = { video: "Video", audio: "Audio", transcript: "Transcript" };

export function bytes(n) {
  if (!n) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(i >= 2 ? 1 : 0)} ${units[i]}`;
}

export function duration(s) {
  if (s === null || s === undefined) return "";
  const t = Math.round(s);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const x = String(t % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${x}` : `${m}:${x}`;
}

/** A signed link the server returned (relative) as an absolute URL on this console. */
export const abs = (rel) => new URL(rel, location.origin).href;

export const typeLabel = (it) => (it.format === "mp3" ? "MP3" : TYPE_LABEL[it.type] || it.type);
export const facts = (it) => [it.platform, duration(it.duration_s), bytes(it.bytes)].filter(Boolean).join(" · ");
export const tagChips = (tags) => tags.map((t) => `<span class="chip">#${esc(t)}</span>`).join("");

/** One library card: thumbnail when there is one, kicker, title, facts, a few tags, the review state.
 *  No thumbnail means no placeholder block: the card is text only, so unthumbed items stay dense. */
const TAGS_ON_CARD = 3;
export function card(it) {
  const thumb = it.links.thumb
    ? `<span class="media-thumb" data-type="${esc(it.type)}"><img src="${esc(it.links.thumb)}" alt="" loading="lazy">${it.duration_s ? `<span class="media-dur">${esc(duration(it.duration_s))}</span>` : ""}</span>`
    : "";
  const review = it.review === "pending" ? '<span class="status" data-s="serious">Tags to review</span>' : "";
  const rest = it.tags.length - TAGS_ON_CARD;
  return `<a class="card media-card${thumb ? "" : " no-thumb"}" href="#/media/item/${it.id}">
      ${thumb}<span class="media-body">
      <span class="media-kick">${esc(typeLabel(it))}${it.group ? `<span class="media-group"> · ${esc(it.group)}</span>` : ""}</span>
      <h4>${esc(it.title)}</h4>
      <span class="meta media-facts">${esc(facts(it))}</span>
      ${it.tags.length ? `<span class="media-tags">${it.tags.slice(0, TAGS_ON_CARD).map((t) => `#${esc(t)}`).join(" ")}${rest > 0 ? ` <span class="faint">+${rest}</span>` : ""}</span>` : ""}${review}</span>
    </a>`;
}

/** Download a file to the device: Telegram's own downloader where the client has it, else the browser. */
export function downloadToDevice(it) {
  const url = abs(it.links.download);
  const w = tg();
  if (w?.downloadFile && w.isVersionAtLeast?.("8.0")) {
    w.downloadFile({ url, file_name: it.file_name });
    return;
  }
  openLink(url);
}

/** Wait for a job to end (polling the job list); onEnd(job) once. Returns a cancel function. */
export function waitJob(jobId, onEnd, { every = 1500, maxMs = 15 * 60 * 1000 } = {}) {
  let stopped = false;
  let timer = null;
  const t0 = Date.now();
  const tickJob = async () => {
    if (stopped) return;
    try {
      const { jobs } = await media.jobs();
      const job = jobs.find((j) => j.id === jobId);
      if (!job || job.state !== "running") { onEnd(job || null); return; }
    } catch (e) {
      onEnd(null, e);
      return;
    }
    if (Date.now() - t0 < maxMs) timer = setTimeout(tickJob, every);
  };
  timer = setTimeout(tickJob, every);
  return () => { stopped = true; clearTimeout(timer); };
}

export const jobToast = (job, what) => toast(`${what} started — the Jobs tab shows its progress.`, "cost", job.title);

/** The tag review panel for one item. The suggestion's tags start pressed when it is pending; nothing is
 *  confirmed until Save. onSaved(item) gets the updated item. Returns a cleanup function. */
export function reviewPanel(root, it, facets, { onSaved, compact = false } = {}) {
  const sug = it.suggest || {};
  const pending = it.review === "pending";
  const chosen = new Set(pending ? [...(sug.tags || []), ...it.tags] : it.tags);
  const all = [...new Set([...it.tags, ...(sug.tags || [])])];
  const vocab = `tags-${it.id}`;
  const groups = `groups-${it.id}`;
  let cancelWait = null;
  const links = (sug.links || []).slice(0, compact ? 4 : 12);
  root.innerHTML = `<div class="review">
      ${sug.summary ? `<p class="gist">${esc(sug.summary)}</p>` : ""}
      ${!sug.tags && !it.tags.length ? '<p class="meta">No suggestion yet — ask for one, or add tags yourself.</p>' : ""}
      <div class="chips" role="group" aria-label="Tags (tap to keep or drop)"></div>
      <div class="review-add"><input class="input" type="text" list="${vocab}" placeholder="Add a tag" aria-label="Add a tag" maxlength="32">
        <button class="btn btn-s" type="button" data-act="add"><i data-icon="plus"></i>Add</button></div>
      <datalist id="${vocab}">${facets.tags.map(([t]) => `<option value="${esc(t)}">`).join("")}</datalist>
      <div class="field review-group"><label for="g-${it.id}">Group</label>
        <input class="input" id="g-${it.id}" type="text" list="${groups}" maxlength="48" value="${esc(pending ? sug.group || it.group : it.group || sug.group || "")}" placeholder="A shelf, e.g. AI agents">
        <datalist id="${groups}">${facets.groups.map(([g]) => `<option value="${esc(g)}">`).join("")}</datalist></div>
      ${links.length ? `<ul class="review-links">${links.map((l) => `<li><span class="tag">${l.kind === "repo" ? "Repo" : "Link"}</span>
        <a href="${esc(l.url)}" data-ext>${esc(l.label || l.url)}</a></li>`).join("")}</ul>` : ""}
      <div class="btn-row"><button class="btn btn-a" type="button" data-act="save"><i data-icon="check"></i>${pending ? "Accept tags" : "Save tags"}</button>
        <button class="btn" type="button" data-act="suggest"><i data-icon="sparkles"></i>${sug.tags ? "Suggest again" : "Suggest tags"}</button></div>
      <div class="sec-notice" hidden></div>
    </div>`;
  const chips = $(".chips", root);
  const notice = $(".sec-notice", root);
  const paintChips = () => {
    chips.innerHTML = all.map((t) => `<button class="chip" type="button" aria-pressed="${chosen.has(t)}" data-tag="${esc(t)}">#${esc(t)}</button>`).join("")
      || '<span class="meta">No tags yet.</span>';
  };
  paintChips();
  chips.addEventListener("click", (e) => {
    const b = e.target.closest("[data-tag]");
    if (!b) return;
    const t = b.dataset.tag;
    if (chosen.has(t)) chosen.delete(t); else chosen.add(t);
    b.setAttribute("aria-pressed", String(chosen.has(t)));
  });
  const input = $(".review-add input", root);
  const add = () => {
    const t = input.value.trim().toLowerCase().replace(/^#/, "").replace(/[^a-z0-9+#.-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 32);
    if (!t) return;
    if (!all.includes(t)) all.push(t);
    chosen.add(t);
    input.value = "";
    paintChips();
  };
  $('[data-act="add"]', root).addEventListener("click", add);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); add(); } });
  $$("a[data-ext]", root).forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openLink(a.href); }));
  const save = $('[data-act="save"]', root);
  save.addEventListener("click", async () => {
    const r = await act(save, notice, () => media.saveTags(it.id, all.filter((t) => chosen.has(t)), $(`#g-${it.id}`, root).value.trim()));
    if (r) { toast(`${r.item.tags.length} tag(s) saved${r.item.group ? ` · ${r.item.group}` : ""}.`, "cost", "Tags"); onSaved?.(r.item); }
  });
  const again = $('[data-act="suggest"]', root);
  again.addEventListener("click", async () => {
    const r = await act(again, notice, () => media.suggest(it.id));
    if (!r) return;
    again.disabled = true;
    again.textContent = "Asking the model…";
    cancelWait = waitJob(r.job.id, async (job, err) => {
      again.disabled = false;
      if (err || !job || job.state === "failed") {
        notice.hidden = false;
        notice.innerHTML = `<aside class="callout problem" data-kind="trap" role="alert"><h4>Tags<i>No suggestion</i></h4><p>${esc(job?.error || err?.message || "the job was lost (did the bot restart?)")}</p></aside>`;
        again.textContent = "Suggest again";
        return;
      }
      const fresh = await media.item(it.id);
      reviewPanel(root, fresh.item, fresh.facets, { onSaved, compact });
      window.UI.init(root);
    });
  });
  return () => cancelWait?.();
}

// ---- Investigate ----------------------------------------------------------------------------------
const WHERE = { title: "in the title", description: "in the description", comments: "in the uploader's comment",
  tags: "in the tags", chapters: "in the chapters", transcript: "said" };

/** 45485 → "45.5k"; null → "". */
export const stars = (n) => (n == null ? "" : n >= 1000 ? `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k` : String(n));

/** One repo / tool / site row: a tick (when `tick` is not null), the name (opens outside), its facts, the
 *  project's own words, where the item says it, and a bookmark button. */
export function repoRow(r, { tick = null, marked = false, attached = false } = {}) {
  const facts = [r.kind === "tool" ? "tool or site" : "", r.stars != null ? `★${stars(r.stars)}` : "", r.archived ? "archived" : "",
    r.fork ? "fork" : "", r.renamed_from ? `was ${r.renamed_from}` : "", r.sure === false ? "looked up" : ""].filter(Boolean);
  const where = r.quote ? `<p class="meta repo-quote">${esc(WHERE[r.from] || "in the material")}: “${esc(r.quote)}”${r.at_s != null ? ` <span class="mono">${esc(duration(r.at_s))}</span>` : ""}</p>` : "";
  return `<li class="repo-row">
      ${tick === null ? "" : `<label class="check repo-tick"><input type="checkbox" data-key="${esc(r.key)}"${tick ? " checked" : ""} aria-label="Attach ${esc(r.name)}"></label>`}
      <div class="repo-main">
        <p class="repo-head"><a class="repo-name" href="${esc(r.url)}" data-ext>${esc(r.name)}</a>${facts.length ? ` <span class="meta">${esc(facts.join(" · "))}</span>` : ""}${attached ? ' <span class="status" data-s="good">attached</span>' : ""}</p>
        ${r.about ? `<p class="repo-about">${esc(r.about)}</p>` : ""}${where}
      </div>
      <button class="btn btn-s btn-ghost repo-bm" type="button" data-bm="${esc(r.key)}" aria-pressed="${marked}" data-tip="${marked ? "Bookmarked" : "Bookmark it"}" aria-label="${marked ? "Bookmarked" : "Bookmark"} ${esc(r.name)}"><i data-icon="bookmark"></i></button>
    </li>`;
}

function skippedHtml(inv) {
  const s = inv.skipped || {};
  const read = inv.read || {};
  const lines = [
    s.not_found?.length ? `Not public on GitHub (gone or private): ${s.not_found.slice(0, 8).join(", ")}` : "",
    s.unresolved?.length ? `No match found for: ${s.unresolved.slice(0, 8).join(", ")}` : "",
    s.unchecked?.length ? `Not checked (${s.unchecked.length}): ${s.stopped || "the per-run limit"}` : "",
    s.errors?.length ? `Lookups that failed: ${s.errors.slice(0, 3).join("; ")}` : "",
    s.more ? `${s.more} more were left out.` : "",
    inv.note || "",
    read.cut ? "The transcript is long: its first part was read." : "",
    inv.at && !read.transcript && !inv.note ? "No transcript was read." : "",
  ].filter(Boolean);
  return lines.length ? `<ul class="meta repo-skipped">${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : "";
}

/** The Investigate panel for one item: what it carries, and the latest investigation as a checklist (found
 *  entries ticked when Argus is sure, plus what is already attached). Nothing changes until Attach;
 *  Investigate (again) starts a job the panel waits on. onSaved(item) after an attach. Returns a cleanup. */
export function reposPanel(root, it, bookmarked, { onSaved } = {}) {
  const inv = it.investigate || {};
  const found = inv.found || [];
  const pending = inv.state === "pending" && found.length > 0;
  const attached = it.repos || [];
  const attachedKeys = new Set(attached.map((r) => r.key));
  const marked = new Set(bookmarked || []);
  const rows = pending ? [...found, ...attached.filter((r) => !found.some((f) => f.key === r.key))] : attached;
  const chosen = new Set(pending ? [...found.filter((f) => f.on).map((f) => f.key), ...attachedKeys] : attachedKeys);
  let cancelWait = null;
  root.innerHTML = `<div class="repos-panel">
      ${rows.length ? `<ul class="repo-list">${rows.map((r) => repoRow(r, { tick: chosen.has(r.key), marked: marked.has(r.key), attached: pending && attachedKeys.has(r.key) })).join("")}</ul>`
        : `<p class="meta">${inv.at ? "Nothing that checks out was found." : "Investigate reads the title, description, the uploader's comments and the transcript (it transcribes first when there is none), then checks every repo on GitHub and every site before it shows it."}</p>`}
      ${skippedHtml(inv)}
      <div class="btn-row">
        ${rows.length ? `<button class="btn btn-a" type="button" data-act="save"><i data-icon="check"></i>${pending ? "Attach ticked" : "Save"}</button>` : ""}
        ${pending ? '<button class="btn" type="button" data-act="save-bm"><i data-icon="bookmark"></i>Attach + bookmark</button>' : ""}
        <button class="btn" type="button" data-act="investigate"><i data-icon="scan-search"></i>${inv.at ? "Investigate again" : "Investigate"}</button>
      </div>
      <div class="sec-notice" hidden></div>
    </div>`;
  const notice = $(".sec-notice", root);
  $$("a[data-ext]", root).forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openLink(a.href); }));
  $$('input[type="checkbox"][data-key]', root).forEach((c) => c.addEventListener("change", () => {
    if (c.checked) chosen.add(c.dataset.key); else chosen.delete(c.dataset.key);
  }));
  const save = async (button, withBookmarks) => {
    const keys = rows.map((r) => r.key).filter((k) => chosen.has(k));
    const r = await act(button, notice, () => media.saveRepos(it.id, keys, withBookmarks ? keys : []));
    if (r) { toast(`${r.item.repos.length} attached${withBookmarks ? " and bookmarked" : ""}.`, "cost", "Repos"); onSaved?.(r.item); }
  };
  const saveBtn = $('[data-act="save"]', root);
  if (saveBtn) saveBtn.addEventListener("click", () => save(saveBtn, false));
  const saveBm = $('[data-act="save-bm"]', root);
  if (saveBm) saveBm.addEventListener("click", () => save(saveBm, true));
  $$("[data-bm]", root).forEach((b) => b.addEventListener("click", async () => {
    const r = await act(b, notice, () => media.bookmark(it.id, b.dataset.bm));
    if (r) { b.setAttribute("aria-pressed", "true"); toast(`${r.bookmark.name} is in your bookmarks (Media → Repos).`, "cost", "Bookmarked"); }
  }));
  const again = $('[data-act="investigate"]', root);
  again.addEventListener("click", async () => {
    const r = await act(again, notice, () => media.investigate(it.id));
    if (!r) return;
    again.disabled = true;
    again.textContent = "Investigating… (the Jobs tab shows each step)";
    cancelWait = waitJob(r.job.id, async (job, err) => {
      again.disabled = false;
      if (err || !job || job.state === "failed") {
        notice.hidden = false;
        notice.innerHTML = `<aside class="callout problem" data-kind="trap" role="alert"><h4>Investigate<i>Did not finish</i></h4><p>${esc(job?.error || err?.message || "the job was lost (did the bot restart?)")}</p></aside>`;
        again.textContent = "Investigate again";
        return;
      }
      const fresh = await media.item(it.id);
      reposPanel(root, fresh.item, fresh.bookmarked, { onSaved });
      window.UI.init(root);
    });
  });
  return () => cancelWait?.();
}
