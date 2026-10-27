// #/media[/<tab>] — the media library: Library (browse, search, filter by type, tag and group), Get (read a
// link, choose video quality or mp3; upload from the device), Review (tag suggestions to confirm), Jobs
// (downloads, conversions, transcripts, tagging — polled while any runs) and Settings.
import { ApiError, media } from "../api.js";
import { $, $$, act, esc, showProblem, toast, when } from "./common.js";
import { bytes, card, duration, jobToast, reviewPanel, TYPES } from "./media_common.js";

const TABS = [["library", "Library"], ["get", "Get"], ["review", "Review"], ["jobs", "Jobs"], ["settings", "Settings"]];
const PAGE = 30;
const TAG_PEEK = 8;                                // tags shown before "Show all"
const JOB_POLL_MS = 2000;
const MP3_LABEL = { 320: "320 kbps (mp3's maximum)", v0: "V0 VBR (~245 kbps, transparent)", 192: "192 kbps", 128: "128 kbps" };

export async function render(el, [tab = "library"], ctx) {
  ctx.crumbs([["#/media", "Media"], ...(tab === "library" ? [] : [["", TABS.find(([k]) => k === tab)[1]]])]);
  const { facets } = await media.list({ limit: 1 });
  el.innerHTML = `<div class="page-head"><div><span class="kicker">Media</span><h1>Your saved videos and audio</h1>
      <p>Files live on your PC; links to them are private and expire. Argus suggests tags — you confirm them.</p></div></div>
    <nav class="tablist media-tabs" role="tablist" aria-label="Media">${TABS.map(([k, label]) => `<a role="tab" href="#/media/${k}" aria-selected="${k === tab}">${esc(label)}${k === "review" && facets.pending ? ` <span class="count">${facets.pending}</span>` : ""}</a>`).join("")}</nav>
    <div id="notice" hidden></div><div id="tab"></div>`;
  const ctxT = { el, notice: $("#notice", el), root: $("#tab", el), facets, cleanups: [] };
  await ({ library, get: getTab, review, jobs, settings }[tab])(ctxT);
  return () => ctxT.cleanups.forEach((f) => f());
}

// ---- Library -------------------------------------------------------------------------------------
async function library(t) {
  const f = { q: "", type: "", tag: "", group: "", offset: 0 };
  t.root.innerHTML = `<section class="sec media-lib">
      <form class="media-filters" role="search">
        <input class="input" type="search" name="q" placeholder="Search titles, tags, gists" aria-label="Search the library">
        <div class="seg seg-s" role="radiogroup" aria-label="Type">${Object.entries(TYPES).map(([v, label]) => `<label><input type="radio" name="type" value="${v}"${v === "" ? " checked" : ""}><span>${esc(label)}</span></label>`).join("")}</div>
        <select class="input" name="group" aria-label="Group"><option value="">Every group</option>${t.facets.groups.map(([g, n]) => `<option value="${esc(g)}">${esc(g)} (${n})</option>`).join("")}</select>
      </form>
      ${t.facets.tags.length ? `<div class="chips media-tagbar" role="group" aria-label="Filter by tag">${t.facets.tags.map(([tag, n]) => `<button class="chip" type="button" aria-pressed="false" data-tag="${esc(tag)}">#${esc(tag)} <span class="faint">${n}</span></button>`).join("")}${t.facets.tags.length > TAG_PEEK ? `<button class="chip chip-more" type="button" id="tag-more" aria-expanded="false"></button>` : ""}</div>` : ""}
      <p class="meta" id="lib-count"></p>
      <div class="media-grid" id="lib-grid"></div>
      <div class="btn-row"><button class="btn" type="button" id="lib-more" hidden>Show more</button></div>
    </section>`;
  const form = $(".media-filters", t.root);
  const grid = $("#lib-grid", t.root);
  const more = $("#lib-more", t.root);
  let seq = 0;
  const load = async (append) => {
    const mine = ++seq;
    if (!append) f.offset = 0;
    let r;
    try {
      r = await media.list({ q: f.q, type: f.type, tag: f.tag, group: f.group, limit: PAGE, offset: f.offset });
    } catch (e) { showProblem(t.notice, e); return; }
    if (mine !== seq) return;                      // a newer filter answered first
    const html = r.items.map(card).join("");
    if (append) grid.insertAdjacentHTML("beforeend", html); else grid.innerHTML = html;
    f.offset += r.items.length;
    more.hidden = f.offset >= r.total;
    $("#lib-count", t.root).textContent = r.total === 0
      ? (f.q || f.type || f.tag || f.group ? "Nothing matches these filters." : "The library is empty. Get a video from a link, upload one, or send it to the bot in the chat.")
      : `${r.total} item${r.total === 1 ? "" : "s"}, newest first`;
  };
  let typing = null;
  form.addEventListener("input", (e) => {
    f.q = form.q.value.trim();
    f.type = form.elements.namedItem("type").value;
    f.group = form.group.value;
    clearTimeout(typing);
    typing = setTimeout(() => load(false), e.target.name === "q" ? 300 : 0);
  });
  form.addEventListener("submit", (e) => e.preventDefault());
  // The most used tags show; the rest wait behind "Show all" (a chosen tag stays visible either way).
  const tagBtns = $$(".media-tagbar [data-tag]", t.root);
  const tagMore = $("#tag-more", t.root);
  const peekTags = () => {
    const all = tagMore?.getAttribute("aria-expanded") === "true";
    tagBtns.forEach((b, i) => { b.hidden = !all && i >= TAG_PEEK && b.dataset.tag !== f.tag; });
    if (tagMore) tagMore.textContent = all ? "Show fewer" : `Show all ${tagBtns.length} tags…`;
  };
  tagMore?.addEventListener("click", () => {
    tagMore.setAttribute("aria-expanded", String(tagMore.getAttribute("aria-expanded") !== "true"));
    peekTags();
  });
  tagBtns.forEach((b) => b.addEventListener("click", () => {
    const on = f.tag !== b.dataset.tag;
    tagBtns.forEach((x) => x.setAttribute("aria-pressed", "false"));
    b.setAttribute("aria-pressed", String(on));
    f.tag = on ? b.dataset.tag : "";
    peekTags();
    load(false);
  }));
  peekTags();
  more.addEventListener("click", () => load(true));
  t.cleanups.push(() => clearTimeout(typing));
  await load(false);
}

// ---- Get: a link, or a file from the device ------------------------------------------------------
async function getTab(t) {
  const s = await media.settings();
  const cfg = s.settings;
  t.root.innerHTML = `${s.ffmpeg ? "" : `<aside class="callout" data-kind="trap"><h4>trap<i>ffmpeg missing</i></h4>
      <p>Without ffmpeg there are no mp3s, thumbnails or merged high-resolution video. Set ARGUS_FFMPEG or install imageio-ffmpeg.</p></aside>`}
    <section class="sec set-sec">
      <div class="set-intro"><h2>From a link</h2><p>YouTube, Shorts, X, Reddit, Instagram — and most other video sites yt-dlp reads.
        Read the link first to see what it offers, or download straight away with your defaults.</p></div>
      <form class="set-panel" id="link-form" novalidate>
        <div class="fields"><div class="field"><label for="url">Link</label>
          <input class="input" id="url" name="url" type="url" inputmode="url" placeholder="https://…" required>
          <p class="err" id="url-err" hidden>Paste an http(s) link first.</p></div></div>
        <div id="probe" hidden></div>
        <div class="set-row"><span class="set-l"><b>Save as</b><small>The video, or only its audio as mp3.</small></span>
          <div class="seg seg-s" role="radiogroup" aria-label="Save as"><label><input type="radio" name="mode" value="video" checked><span>Video</span></label><label><input type="radio" name="mode" value="mp3"><span>MP3</span></label></div></div>
        <div class="set-row" id="row-quality"><label class="set-l" for="quality"><b>Video quality</b><small>Caps the short side, so portrait clips keep their resolution.</small></label>
          <select class="input" id="quality" name="quality">${qualityOptions(null, cfg.video_quality)}</select></div>
        <div class="set-row" id="row-mp3" hidden><label class="set-l" for="mp3"><b>MP3 quality</b><small id="mp3-note">An mp3 cannot be better than its source audio.</small></label>
          <select class="input" id="mp3" name="mp3">${s.mp3_qualities.map((q) => `<option value="${q}"${q === cfg.mp3_quality ? " selected" : ""}>${esc(MP3_LABEL[q] || q)}</option>`).join("")}</select></div>
        <div class="set-panel-f"><button class="btn" type="button" id="read-link"><i data-icon="search"></i>Read link</button>
          <button class="btn btn-a" type="submit"><i data-icon="download"></i>Download to the library</button></div>
      </form>
    </section>
    <section class="sec set-sec">
      <div class="set-intro"><h2>From this device</h2><p>A video, an audio file, or a transcript (.txt .md .srt .vtt) to add to a research run.
        Up to ${s.upload_max_mb} MB, sent in pieces — keep the console open until it finishes.</p></div>
      <form class="set-panel" id="upload-form" novalidate>
        <div class="fields"><div class="field"><label for="file">File</label>
          <input class="input" id="file" name="file" type="file" accept="video/*,audio/*,.txt,.md,.srt,.vtt"></div></div>
        <div class="set-row"><span class="set-l"><b>Then</b><small>Optional, after the upload.</small></span>
          <div class="checks"><label class="check"><input type="checkbox" name="transcribe"> Transcribe it</label>
            <label class="check"><input type="checkbox" name="mp3"> Make an mp3 (videos)</label></div></div>
        <div class="meter" id="up-meter" hidden><div class="meter-top"><span id="up-label">Uploading…</span><b id="up-pct">0%</b></div><progress class="bar" id="up-bar" max="100" value="0"></progress></div>
        <div class="set-panel-f"><button class="btn" type="button" id="up-cancel" hidden>Stop</button>
          <button class="btn btn-a" type="submit"><i data-icon="upload"></i>Upload</button></div>
      </form>
    </section>`;
  const form = $("#link-form", t.root);
  const syncMode = () => {
    const mp3 = form.elements.namedItem("mode").value === "mp3";
    $("#row-quality", form).hidden = mp3;
    $("#row-mp3", form).hidden = !mp3;
  };
  form.addEventListener("change", syncMode);
  const url = () => {
    const v = form.url.value.trim();
    const ok = /^https?:\/\//i.test(v);
    $("#url-err", form).hidden = ok;
    form.url.setAttribute("aria-invalid", String(!ok));
    if (!ok) form.url.focus();
    return ok ? v : null;
  };
  let probed = null;
  const readBtn = $("#read-link", form);
  readBtn.addEventListener("click", async () => {
    const u = url();
    if (!u) return;
    readBtn.textContent = "Reading…";
    const r = await act(readBtn, t.notice, () => media.probe(u));
    readBtn.innerHTML = '<i data-icon="search"></i>Read link';
    window.UI.init(readBtn);
    if (!r) return;
    probed = r.probe;
    const p = probed;
    const box = $("#probe", form);
    box.hidden = false;
    box.innerHTML = `<div class="probe"><span class="tag">${esc(p.platform)}</span><h3>${esc(p.title || u)}</h3>
      <p class="meta">${esc([p.uploader, duration(p.duration_s), p.heights.length ? `up to ${Math.max(...p.heights)}p` : "audio only", p.audio_kbps ? `audio ~${p.audio_kbps} kbps` : ""].filter(Boolean).join(" · "))}</p>
      ${p.description ? `<p class="probe-desc">${esc(p.description)}</p>` : ""}</div>`;
    $("#quality", form).innerHTML = qualityOptions(p.heights, cfg.video_quality);
    $("#mp3-note", form).textContent = p.audio_kbps
      ? `The source audio is about ${p.audio_kbps} kbps; a higher mp3 bitrate adds size, not quality.`
      : "An mp3 cannot be better than its source audio.";
  });
  const submit = $('button[type="submit"]', form);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const u = url();
    if (!u) return;
    const mode = form.elements.namedItem("mode").value;
    const body = { url: u, mode, title: probed?.url === u ? probed.title : undefined };
    if (mode === "mp3") body.mp3_quality = form.mp3.value; else body.quality = form.quality.value;
    const r = await act(submit, t.notice, () => media.fetch(body));
    if (r) { jobToast(r.job, mode === "mp3" ? "The mp3 download" : "The download"); location.hash = "#/media/jobs"; }
  });
  uploadForm(t);
}

function qualityOptions(heights, current) {
  const opts = [["auto", "Auto (up to 1080p)"], ["max", "Maximum available"]];
  const rungs = heights?.length ? [...heights].reverse().filter((h) => h >= 144) : [2160, 1440, 1080, 720, 480, 360];
  for (const h of rungs) opts.push([String(h), `${h}p`]);
  opts.push(["min", "Smallest"]);
  return opts.map(([v, label]) => `<option value="${v}"${v === current ? " selected" : ""}>${esc(label)}</option>`).join("");
}

function uploadForm(t) {
  const form = $("#upload-form", t.root);
  const submit = $('button[type="submit"]', form);
  const stop = $("#up-cancel", form);
  const meter = $("#up-meter", form);
  let cancelled = false;
  let uid = null;
  stop.addEventListener("click", () => { cancelled = true; });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = form.file.files[0];
    if (!file) { form.file.focus(); return; }
    cancelled = false;
    submit.disabled = true;
    stop.hidden = false;
    meter.hidden = false;
    t.notice.hidden = true;
    try {
      const open = await media.uploadOpen(file.name, file.size);
      uid = open.upload_id;
      for (let off = 0; off < file.size; off += open.chunk_size) {
        if (cancelled) { await media.uploadCancel(uid); toast("Upload stopped; nothing was saved.", "why", "Upload"); return; }
        await media.uploadChunk(uid, off, file.slice(off, off + open.chunk_size));
        const pct = Math.min(100, Math.round(((off + open.chunk_size) / file.size) * 100));
        $("#up-bar", form).value = pct;
        $("#up-pct", form).textContent = `${pct}%`;
        $("#up-label", form).textContent = `Uploading ${bytes(Math.min(file.size, off + open.chunk_size))} of ${bytes(file.size)}`;
      }
      const done = await media.uploadFinish(uid, { transcribe: form.transcribe.checked, mp3: form.mp3.checked });
      uid = null;
      toast(done.jobs.length ? `Saved; ${done.jobs.length} job(s) started.` : "Saved to the library.", "cost", done.item.title);
      location.hash = `#/media/item/${done.item.id}`;
    } catch (err) {
      if (!(err instanceof ApiError)) throw err;
      showProblem(t.notice, err);
    } finally {
      submit.disabled = false;
      stop.hidden = true;
    }
  });
}

// ---- Review --------------------------------------------------------------------------------------
async function review(t) {
  const [{ items, total, facets }, untagged] = await Promise.all([
    media.list({ review: "pending", limit: 50 }), media.list({ review: "untagged", limit: 1 })]);
  t.root.innerHTML = `<section class="sec">
      <div class="sec-head"><span class="kicker">Review</span><h2>${total ? `${total} suggestion${total === 1 ? "" : "s"} to confirm` : "Nothing to review"}</h2>
        <p>Tap a tag to keep or drop it, add your own, pick a group, then accept. Nothing counts as a tag until you do.</p></div>
      ${untagged.total ? `<div class="btn-row"><button class="btn" type="button" id="tag-batch"><i data-icon="sparkles"></i>Suggest tags for ${Math.min(untagged.total, 20)} untagged item${untagged.total === 1 ? "" : "s"}</button>
        <span class="meta">${untagged.total} untagged in all</span></div>` : ""}
      <div class="review-list">${items.map((it) => `<article class="review-item" data-id="${it.id}">
          <div class="review-head"><a href="#/media/item/${it.id}"><b>${esc(it.title)}</b></a><span class="meta">${esc([it.platform, duration(it.duration_s)].filter(Boolean).join(" · "))}</span></div>
          <div class="review-body"></div></article>`).join("")}</div>
    </section>`;
  for (const it of items) {
    const art = $(`.review-item[data-id="${it.id}"]`, t.root);
    t.cleanups.push(reviewPanel($(".review-body", art), it, facets, {
      compact: true,
      onSaved: () => { art.remove(); },
    }));
  }
  const batch = $("#tag-batch", t.root);
  if (batch) batch.addEventListener("click", async () => {
    const r = await act(batch, t.notice, () => media.tagUntagged());
    if (r) { toast(r.message, "cost", "Tags"); location.hash = "#/media/jobs"; }
  });
}

// ---- Jobs ----------------------------------------------------------------------------------------
const JOB_STATE = { running: ["warning", "Running"], done: ["good", "Done"], failed: ["critical", "Failed"] };

async function jobs(t) {
  let timer = null;
  let stopped = false;
  t.cleanups.push(() => { stopped = true; clearTimeout(timer); });
  t.root.innerHTML = `<section class="sec"><div class="sec-head"><span class="kicker">Jobs</span><h2>Downloads, conversions, transcripts, tags</h2>
      <p>Jobs run on the PC whether this console is open or not; this list is lost when the bot restarts (the files are not).</p></div>
      <div id="job-list"></div></section>`;
  const list = $("#job-list", t.root);
  const paint = async () => {
    if (stopped) return;
    let r;
    try { r = await media.jobs(); } catch (e) { showProblem(t.notice, e); return; }
    if (stopped) return;
    list.innerHTML = r.jobs.length ? `<ol class="jobs">${r.jobs.map((j) => {
      const [s, label] = JOB_STATE[j.state] || ["idle", j.state];
      return `<li class="job"><div class="job-top"><b>${esc(j.title)}</b><span class="status" data-s="${s}">${esc(label)}</span></div>
        ${j.state === "running" ? (j.pct === null ? '<progress class="bar" aria-label="In progress"></progress>' : `<progress class="bar" max="100" value="${j.pct}" aria-label="${Math.round(j.pct)}%"></progress>`) : ""}
        <p class="meta">${esc(j.error || j.message || "")}${j.finished ? ` · ${esc(when(new Date(j.finished * 1000).toISOString()))}` : ""}</p>
        ${j.state === "done" && j.asset_id ? `<a class="btn btn-s" href="#/media/item/${j.asset_id}">Open</a>` : ""}</li>`;
    }).join("")}</ol>` : '<p class="meta">No jobs since the bot started.</p>';
    if (r.jobs.some((j) => j.state === "running")) timer = setTimeout(paint, JOB_POLL_MS);
  };
  await paint();
}

// ---- Settings ------------------------------------------------------------------------------------
async function settings(t) {
  const s = await media.settings();
  const c = s.settings;
  t.root.innerHTML = `<section class="sec set-sec">
      <div class="set-intro"><h2>Media defaults</h2><p>Used by the Get tab, uploads and the chat. The speech-to-text engine itself is set on the <a href="#/engines">Engines</a> page.</p></div>
      <form class="set-panel" id="media-settings" novalidate>
        <div class="set-row"><label class="set-l" for="vq"><b>Video quality</b><small>Also the chat's /quality.</small></label>
          <select class="input" id="vq" name="video_quality">${qualityOptions(null, c.video_quality)}</select></div>
        <div class="set-row"><label class="set-l" for="mq"><b>MP3 quality</b><small>320 kbps is mp3's ceiling.</small></label>
          <select class="input" id="mq" name="mp3_quality">${s.mp3_qualities.map((q) => `<option value="${q}"${q === c.mp3_quality ? " selected" : ""}>${esc(MP3_LABEL[q] || q)}</option>`).join("")}</select></div>
        <div class="set-row"><label class="set-l" for="lang"><b>Transcript language</b><small>Empty: detect it. A code like en, ru, hy.</small></label>
          <input class="input" id="lang" name="language" type="text" maxlength="3" value="${esc(c.language)}" placeholder="auto"></div>
        <div class="set-row"><span class="set-l"><b>Timestamps</b><small>[m:ss] on each transcript line.</small></span>
          <input class="switch" type="checkbox" name="timestamps" aria-label="Timestamps"${c.timestamps ? " checked" : ""}></div>
        <div class="set-row"><span class="set-l"><b>Suggest tags automatically</b><small>After each download, upload and transcript (uses the cheap model tier).</small></span>
          <input class="switch" type="checkbox" name="auto_tag" aria-label="Suggest tags automatically"${c.auto_tag ? " checked" : ""}></div>
        <div class="set-row"><label class="set-l" for="hours"><b>Link lifetime</b><small>Hours a download or player link works (1–168).</small></label>
          <input class="input" id="hours" name="link_hours" type="number" min="1" max="168" value="${c.link_hours}"></div>
        <div class="set-panel-f"><span class="hint">ffmpeg: ${s.ffmpeg ? "available" : "missing"} · uploads up to ${s.upload_max_mb} MB</span>
          <button class="btn btn-a" type="submit"><i data-icon="check"></i>Save</button></div>
      </form></section>`;
  const form = $("#media-settings", t.root);
  const submit = $('button[type="submit"]', form);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = { video_quality: form.video_quality.value, mp3_quality: form.mp3_quality.value,
                   language: form.language.value.trim().toLowerCase(), timestamps: form.timestamps.checked,
                   auto_tag: form.auto_tag.checked, link_hours: Number(form.link_hours.value) };
    const r = await act(submit, t.notice, () => media.saveSettings(body));
    if (r) toast("Media settings saved.", "cost", "Settings");
  });
}
