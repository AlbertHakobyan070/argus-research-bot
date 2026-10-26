# Media library — design (2026-10-26)

## Intent

Albert saves two kinds of video for later: long talks and tutorials (SWE, AI agents, LLMs), where the value is the
advice inside, and short clips (Shorts, Reels, X), where the value is often one repo, tool or workflow. Downloading
already works. Two things break the habit:

1. **Getting the file back.** Telegram caps bot uploads at about 50 MB (downloads from chat at 20 MB), so most
   saved videos never come back to him.
2. **Finding it again.** Weeks later, what was inside a video is forgotten. Search needs labels Albert didn't
   have to write himself.

So the PC hosts the files and hands out short-lived private links, and Argus suggests tags and a group for every
item, which Albert confirms or edits (human in the loop).

## Goals

- A **Media** section in the Mini App: Library (browse, search, filter by tag/group/kind), Get (paste link → probe →
  video at a chosen quality or mp3; upload from device), Review (tag/group suggestions to confirm), Jobs (progress).
- An **item page**: in-app player, download to device, send to chat, convert to mp3, transcribe (language,
  timestamps), suggested + confirmed tags, group, extracted links/repos, attach the transcript to a research run.
- **Private links** served by the console itself: HMAC-signed, expiring, Range-capable (video seeking).
- **Chat import**: video/audio/voice/video-note/document messages are saved to the library with action buttons;
  over Telegram's 20 MB bot-download cap the bot says so and points to the Mini App upload.
- **No dead ends**: a file too big for Telegram is offered as a download link + "Open in Argus".
- **Auto-tagging**: an LLM role `media.tag` (cheap tier, JSON) proposes 3–8 tags, one group, a 2–3 sentence gist
  and the links it saw; links/GitHub repos are also pulled deterministically from the description and transcript.
  Suggestions never become labels until Albert accepts them (Mini App Review, or ✅ in chat).
- **Transcript → research run**: a transcript asset (made here or imported as .txt/.md/.srt/.vtt) can be queued on a
  run; the run page shows pending sources and a Continue button that ingests them (the existing /append path).
- **Media settings** in the Mini App: default video quality (the existing `media_quality` key, shared with
  /quality), mp3 quality (320k CBR / V0 VBR / 192k / 128k), transcription language and timestamps, auto-tag on/off,
  link lifetime.

## Non-goals (v1)

Google Drive upload, a VPS, splitting files for Telegram, playlists, a persistent job history (jobs are in memory;
results are library rows), re-tagging the whole back catalogue automatically (one button per item, and a
"suggest for untagged" batch capped at 20).

## Anti-goals

- A link that works without the signature, outlives its expiry, or serves a path outside the Argus roots.
- A suggestion silently treated as a confirmed tag.
- A job failure that is swallowed: every failed job keeps its error and shows it.
- Breaking the existing chat media flow (/fetch, /find, pasted links, /quality, /transcripts, /append).

## Constraints

- The library schema is not rebuilt: mp3s are `kind='media'` with `meta.format='mp3'`, `meta.derived_from=<id>`;
  uploads are `platform='upload'`, chat imports `platform='telegram'`. Tags live in `meta_json` (`tags`, `group`,
  `suggest`, `review`), filtered with SQLite JSON1 (`json_each`). Ponytail ceiling: fine to thousands of items; a
  tags table if it ever grows past that.
- Signed links live outside `/api` (no initData on a `<video>` request): `GET /files/{token}`. Key = HMAC of the bot
  token with a fixed label; token = `asset_id.expiry.kind.sig`; the served path is the asset's registered path, checked
  inside the Argus roots (`_safe_to_delete`'s rule).
- Uploads are chunked (8 MiB `PUT`s, under Cloudflare's ~100 MB request cap), streamed to a `.part` file, size
  checked against `ARGUS_MEDIA_UPLOAD_MAX_MB` (default 4096).
- The tunnel URL changes per restart, so links are minted fresh from the current public URL and never stored.
- Telegram Mini App download: `WebApp.downloadFile` when the client has it (8.0+), else open the link externally.
- mp3 from a link: yt-dlp `-x --audio-format mp3` on the best audio stream; mp3 from a file: ffmpeg libmp3lame.
  "Max" is 320k CBR (mp3's ceiling); V0 is ~245k VBR. Upconverting a low-bitrate source adds no quality — the UI says so.
- The Mini App's Get tab accepts any http(s) URL (yt-dlp decides; folder `other`); the chat keeps its four platforms.

## Build order

1. Library: `update_asset_meta`, `search_assets` (+ total), `asset_facets`.
2. Engine: `download_media(audio=…, any_url=…, thumbnail)`, `probe_url`, `convert_to_mp3`, `make_thumbnail`,
   `media_duration`; `transcribe_file` (language, timestamps); role `media.tag` + `media_tags.py`.
3. `media_jobs.py`: in-memory jobs (state, pct, message, asset_id); `media_service.py`: fetch / convert /
   transcribe / tag / import, shared by the Mini App and the chat.
4. Web: `webapp/media_api.py` routes + `/files/{token}`; run payload pending sources + continue route.
5. Frontend: Media nav, views `media.js` (tabs), `media_item.js`; api helpers; CSS.
6. Chat: file handlers, mx:* buttons, big-file link, /media.
7. Docs (help.md), tests, live check (real bot: link download, upload, mp3, tag, attach).
