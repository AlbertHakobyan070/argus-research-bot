// #/media/item/<id> — one library item: the player (signed, expiring link; seeking works), Download to the
// device, Send to chat (a link when Telegram's cap is passed), MP3, Transcribe, the tag review, the item's
// parts (its mp3s and transcripts), Add a transcript to a research run (then Continue it), Delete.
import { continueRun, get, media } from "../api.js";
import { $, $$, act, confirmDialog, esc, openLink, toast, when } from "./common.js";
import { bytes, downloadToDevice, duration, facts, jobToast, reviewPanel, tagChips, typeLabel, waitJob } from "./media_common.js";

const MP3_LABEL = { 320: "320 kbps", v0: "V0 VBR", 192: "192 kbps", 128: "128 kbps" };

export async function render(el, [id], ctx) {
  const st = { el, id, cleanups: [], stopped: false };
  const paint = async () => {
    st.cleanups.splice(0).forEach((f) => f());
    const [data, settings] = await Promise.all([media.item(id), media.settings()]);
    if (st.stopped) return;
    ctx.crumbs([["#/media", "Media"], ["", data.item.title]]);
    draw(st, data, settings, paint);
    window.UI.init(el);
  };
  await paint();
  return () => { st.stopped = true; st.cleanups.forEach((f) => f()); $$("video,audio", el).forEach((m) => m.pause()); };
}

function draw(st, { item: it, parts, text, present, facets }, settings, repaint) {
  const isMedia = it.kind === "media";
  const transcripts = it.kind === "transcript" ? [it] : parts.filter((p) => p.kind === "transcript");
  const player = !present
    ? '<aside class="callout problem" data-kind="trap"><h4>Missing<i>The file is not on disk</i></h4><p>It was moved or deleted outside Argus. Delete the item to tidy the library.</p></aside>'
    : it.type === "video" ? `<video class="media-player" controls preload="metadata" playsinline src="${esc(it.links.file)}"${it.links.thumb ? ` poster="${esc(it.links.thumb)}"` : ""}></video>`
    : it.type === "audio" ? `<audio class="media-audio" controls preload="metadata" src="${esc(it.links.file)}"></audio>`
    : `<pre class="transcript">${esc(text || "")}</pre>${(text || "").length >= 12000 ? '<p class="meta">The first part is shown; Download has the whole file.</p>' : ""}`;
  st.el.innerHTML = `<div class="page-head">
      <div><span class="kicker">${esc(typeLabel(it))}${it.platform ? ` · ${esc(it.platform)}` : ""}${it.imported ? " · imported" : ""}</span><h1 id="title">${esc(it.title)}</h1>
        <form class="rename" id="rename-form" hidden novalidate>
          <input class="input" id="rename-input" type="text" maxlength="200" value="${esc(it.title)}" aria-label="New title" required>
          <button class="btn btn-a btn-s" type="submit"><i data-icon="check"></i>Save</button>
          <button class="btn btn-s" type="button" id="rename-cancel">Cancel</button>
          <p class="hint">Downloads and files sent to the chat take this name${it.derived_from || it.media_asset_id ? "" : "; so do the item's parts"}. The vault file keeps <code>${esc(it.vault_name)}</code>.</p></form>
        <p class="meta">${esc([facts(it), it.quality && `quality ${it.quality}`, it.backend, it.language, `saved ${when(it.created_at)}`].filter(Boolean).join(" · "))}</p>
        ${it.tags.length || it.group ? `<p class="chips media-item-tags">${it.group ? `<span class="tag">${esc(it.group)}</span>` : ""}${tagChips(it.tags)}</p>` : ""}</div>
      <div class="btn-row"><button class="btn btn-s" type="button" id="rename"><i data-icon="file-text"></i>Rename</button>
        ${it.source_url ? '<button class="btn btn-s" type="button" id="source"><i data-icon="external-link"></i>Source</button>' : ""}</div>
    </div>
    <div id="notice" hidden></div>
    <section class="sec media-view">${player}</section>
    <section class="sec">
      <div class="sec-head"><span class="kicker">Actions</span><h2>Get it, share it, change it</h2></div>
      <div class="btn-row">
        <button class="btn btn-a" type="button" id="dl"${present ? "" : " disabled"}><i data-icon="download"></i>Download to this device</button>
        <button class="btn" type="button" id="send"${present ? "" : " disabled"}><i data-icon="message-square"></i>Send to chat</button>
        <button class="btn btn-danger" type="button" id="del"><i data-icon="trash-2"></i>Delete…</button>
      </div>
      <p class="hint media-hint">Over ${esc(bytes(49 * 1048576))}, Send to chat sends a private download link instead (valid ${settings.settings.link_hours} h, or until the bot restarts).</p>
      ${isMedia && present ? `<div class="media-actions">
        <form class="action-row" id="mp3-form"><b>MP3</b>
          <select class="input" name="quality" aria-label="MP3 quality">${settings.mp3_qualities.map((q) => `<option value="${q}"${q === settings.settings.mp3_quality ? " selected" : ""}>${esc(MP3_LABEL[q] || q)}</option>`).join("")}</select>
          <button class="btn" type="submit"${settings.ffmpeg ? "" : " disabled"}>Make mp3</button></form>
        <form class="action-row" id="tr-form"><b>Transcript</b>
          <input class="input" name="language" type="text" maxlength="3" placeholder="auto" value="${esc(settings.settings.language)}" aria-label="Language code (empty: detect)">
          <label class="check"><input type="checkbox" name="timestamps"${settings.settings.timestamps ? " checked" : ""}> Timestamps</label>
          <button class="btn" type="submit"><i data-icon="file-text"></i>Transcribe</button></form>
      </div>` : ""}
    </section>
    <section class="sec"><div class="sec-head"><span class="kicker">Tags</span><h2>${it.review === "pending" ? "Suggested tags — confirm or edit" : "Tags and group"}</h2>
      <p>What you will search for later. Suggestions come from the title, description and transcript.</p></div><div id="review"></div></section>
    ${parts.length ? `<section class="sec"><div class="sec-head"><span class="kicker">Parts</span><h2>Made from this item</h2></div>
      <div class="tw"><table><thead><tr><th>Part</th><th>Size</th><th></th></tr></thead><tbody>${parts.map((p) => `<tr>
        <td><a href="#/media/item/${p.id}">${esc(typeLabel(p))}${p.quality ? ` · ${esc(p.quality)}` : ""}${p.backend ? ` · ${esc(p.backend)}` : ""}</a><span class="meta row-id">${esc(p.vault_name)}</span></td>
        <td>${esc(bytes(p.bytes))}${p.duration_s ? ` · ${esc(duration(p.duration_s))}` : ""}</td>
        <td class="btn-row"><button class="btn btn-s" type="button" data-dl="${p.id}"><i data-icon="download"></i>Download</button>
          <button class="btn btn-s" type="button" data-send="${p.id}">Send</button></td></tr>`).join("")}</tbody></table></div></section>` : ""}
    ${transcripts.length ? `<section class="sec set-sec"><div class="set-intro"><h2>Use it in research</h2>
        <p>Queue ${transcripts.length > 1 ? "a" : "the"} transcript on a research run; Continue ingests it (an append-only pass on a finished run, no new searches).</p></div>
      <form class="set-panel" id="attach-form" novalidate>
        ${transcripts.length > 1 ? `<div class="set-row"><label class="set-l" for="att-t"><b>Transcript</b></label><select class="input" id="att-t" name="transcript">${transcripts.map((p) => `<option value="${p.id}">${esc(p.vault_name)}</option>`).join("")}</select></div>` : `<input type="hidden" name="transcript" value="${transcripts[0].id}">`}
        <div class="set-row"><label class="set-l" for="att-run"><b>Run</b><small>Newest first.</small></label><select class="input" id="att-run" name="run"><option value="">Loading runs…</option></select></div>
        <div class="set-panel-f"><span class="hint" id="att-hint"></span><button class="btn btn-a" type="submit"><i data-icon="plus"></i>Add to run</button></div>
      </form></section>` : ""}`;
  const notice = $("#notice", st.el);
  const byId = Object.fromEntries([it, ...parts].map((p) => [p.id, p]));
  const src = $("#source", st.el);
  if (src) src.addEventListener("click", () => openLink(it.source_url));
  const renameForm = $("#rename-form", st.el);
  const renameInput = $("#rename-input", st.el);
  const renameBtn = $("#rename", st.el);
  const showRename = (on) => {
    renameForm.hidden = !on;
    $("#title", st.el).hidden = on;
    renameBtn.hidden = on;
    if (on) { renameInput.focus(); renameInput.select(); }
  };
  renameBtn.addEventListener("click", () => showRename(true));
  $("#rename-cancel", st.el).addEventListener("click", () => { renameInput.value = it.title; showRename(false); });
  renameForm.addEventListener("keydown", (e) => { if (e.key === "Escape") { renameInput.value = it.title; showRename(false); } });
  renameForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = renameInput.value.trim();
    if (!title) { renameInput.focus(); return; }
    if (title === it.title) { showRename(false); return; }
    const r = await act($('button[type="submit"]', renameForm), notice, () => media.rename(it.id, title));
    if (r) { toast(`Renamed to “${r.item.title}”.`, "cost", "Rename"); repaint(); }
  });
  $("#dl", st.el).addEventListener("click", () => downloadToDevice(it));
  const sendTo = async (button, p) => {
    const r = await act(button, notice, () => media.send(p.id));
    if (r) toast(r.message, "cost", "Send to chat");
  };
  $("#send", st.el).addEventListener("click", (e) => sendTo(e.currentTarget, it));
  $$("[data-dl]", st.el).forEach((b) => b.addEventListener("click", () => downloadToDevice(byId[b.dataset.dl])));
  $$("[data-send]", st.el).forEach((b) => b.addEventListener("click", () => sendTo(b, byId[b.dataset.send])));
  const del = $("#del", st.el);
  del.addEventListener("click", async () => {
    const what = it.derived_from || it.media_asset_id ? "this part (the item it belongs to stays)"
      : parts.length ? `the item and its ${parts.length} part(s)` : "the item";
    const ok = await confirmDialog({ title: `Delete ${it.title}?`, body: `This deletes ${what} from the library and the files from your vault. It cannot be undone.`, ok: "Delete" });
    if (!ok) return;
    const r = await act(del, notice, () => media.remove(it.id));
    if (r) { toast(r.message, "why", "Deleted"); location.hash = it.derived_from ? `#/media/item/${it.derived_from}` : it.media_asset_id ? `#/media/item/${it.media_asset_id}` : "#/media"; }
  });
  const follow = (job, what) => {
    jobToast(job, what);
    st.cleanups.push(waitJob(job.id, (j, err) => {
      if (st.stopped) return;
      if (err || !j || j.state === "failed") toast(j?.error || err?.message || "the job was lost", "trap", `${what} failed`);
      else { toast(j.message || "Done.", "cost", `${what} done`); repaint(); }
    }));
  };
  const mp3 = $("#mp3-form", st.el);
  if (mp3) mp3.addEventListener("submit", async (e) => {
    e.preventDefault();
    const r = await act($("button", mp3), notice, () => media.convert(it.id, mp3.quality.value));
    if (r) follow(r.job, "MP3");
  });
  const tr = $("#tr-form", st.el);
  if (tr) tr.addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = { language: tr.language.value.trim().toLowerCase(), timestamps: tr.timestamps.checked };
    const r = await act($('button[type="submit"]', tr), notice, () => media.transcribe(it.id, body));
    if (r) follow(r.job, "Transcript");
  });
  st.cleanups.push(reviewPanel($("#review", st.el), it, facets, { onSaved: () => repaint() }));
  const att = $("#attach-form", st.el);
  if (att) attachForm(st, att, notice);
}

async function attachForm(st, form, notice) {
  const sel = form.run;
  try {
    const { runs } = await get.runs(50);
    sel.innerHTML = runs.length ? runs.map((r) => `<option value="${esc(r.run_id)}">${esc((r.topic || r.run_id).slice(0, 70))} · ${esc(r.status)}</option>`).join("")
      : '<option value="">No runs yet</option>';
  } catch (e) {
    sel.innerHTML = '<option value="">Runs could not load</option>';
  }
  const submit = $('button[type="submit"]', form);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!sel.value) { sel.focus(); return; }
    const runId = sel.value;
    const r = await act(submit, notice, () => media.attach(form.transcript.value, runId));
    if (!r) return;
    $("#att-hint", form).innerHTML = `${esc(r.message)} <button class="btn btn-s" type="button" id="att-go">Continue run ${esc(runId)}</button>
      <a class="btn btn-s" href="#/run/${esc(runId)}">Open run</a>`;
    const go = $("#att-go", form);
    go.addEventListener("click", async () => {
      const c = await act(go, notice, () => continueRun(runId));
      if (c) { toast(c.message || "Continuing.", "cost", `Run ${runId}`); location.hash = `#/run/${runId}`; }
    });
  });
}
