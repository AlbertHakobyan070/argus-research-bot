# Investigate: repos, tools and sites a saved video mentions (2026-11-27)

Albert's request: "Some videos mention one or multiple Github repo-s; I'd like to have an investigate button which
figures out from transcript/video-content/advanced-lookup for the mentioned or inferred repo-s, checks that the
links are real and valid and lets me see them and attach to that media just like the tags are possible to be
attached to the video … and it must be possible to save cool repositories in a separate repository bookmark
section where short repo descriptions are linked to the respective saved github repo-s (descriptions from the
repo-s themselves may be possible to attach to them with optional button) … website/software_name/tool related
mentions could be also scraped … (optional)".

## Settled choices
- **Video content** means what is said, plus the metadata. Text that is only shown on screen (frame OCR or a
  vision model) is left out of this version.
- **No transcript yet:** Investigate transcribes first. YouTube goes to captions first, so this usually takes
  seconds; the other sites run speech-to-text over the file. If that fails, the finder reads the rest and says why.
- **Tools and sites** are included as their own kind: `kind = "tool"`, checked by loading the page.
- **GitHub token:** `GITHUB_TOKEN` in `.env` is optional. Without it GitHub allows 60 checks an hour and 10
  searches a minute; the code works either way and reports when a limit stops the checks.

## Data
- **The item's `meta_json`** (no schema change). It sits next to `suggest` and `tags`, but is separate from
  `meta.review`, which belongs to tags:
  - `meta.investigate` holds the latest result:
    `{rev, at, state: pending|done, found: [...], skipped, model, read, note}`. Each entry of `found` is
    `{kind, key, name, url, about, said, stars, archived, fork, renamed_from, from, quote, at_s, how, sure, on}`.
    `rev` goes up on every run, and a chat button carries it, so an older list cannot act.
  - `meta.repos` holds the attached entries, `[{kind, key, name, url, about, stars}]`. Library search matches it
    (LIKE over `meta_json`).
- **Table `bookmarks`**, created with `CREATE TABLE IF NOT EXISTS`, so it is additive:
  - Columns: `bookmark_id, kind (repo|tool), key, name, url, note, about, stars, asset_id`
    (`ON DELETE SET NULL`), `created_at, updated_at`, with `UNIQUE(kind, key)`.
  - `key` is the lowercase `owner/name`, or a tool's canonical URL.
  - `note` is yours. It is prefilled with what the video says about the project and never overwritten by a
    re-bookmark.
  - `about` holds the project's own words. Only the From button fills it.
  - The items carrying a bookmark come from `json_each(meta.repos)`.

## Pipeline (`argus/investigate.py`, job kind `investigate`)
1. **Material:**
   - the library title and the info.json title;
   - the description, tags and chapters;
   - the uploader's own comments (Instagram info.json carries comments);
   - the whole transcript, with the header and VTT cues stripped and `[m:ss]` stamps kept.
2. **Short links** (t.co, bit.ly …) in the title, description and comments are followed through the fetchers'
   SSRF guard.
3. **Deterministic:** github.com links (not gist.), github.io, raw.githubusercontent, and bare `owner/repo`
   (the title, a line of its own, or a line that says GitHub, repo or stars). These are "sure".
4. **The model:** role `media.investigate` (cheap tier, JSON, MUST_RAISE) lists the mentions as
   `{name, kind, owner, repo, url, quote, about}`, reading transcripts in chunks of 12k characters (at most 4).
   - A mention is kept only when the material contains its quote, or its name with spaces, dots and hyphens
     ignored. The evidence shown is the material's own words.
   - An `owner/repo` from the model is sure only when its words say both (a spoken `owner slash repo`); an
     owner the model supplied itself is checked but starts unticked.
5. **Resolve and check:**
   - A bare repo name: GitHub search `<name> in:name` (most stars first); the repos whose name contains it,
     with their descriptions, go to the same role, which picks the one that IS the project or none — a repo
     that only shares the name is not it (live: "Stryker" had matched a CPU-Z exploit tool). None there: the
     repos a ddgs search for `<name> github` links (a URL carrying the name), picked the same way; else the
     name is listed as no match.
   - Every repo is checked with `GET /repos/{o}/{r}`, redirects followed (renames). A 404 or a private repo is
     dropped and counted.
   - A tool's site is the link from the material or a web search whose domain carries its name; it must load
     (2xx).
   - Limits per run: 15 checks, 6 GitHub searches, 6 web searches, 6 pages and 10 short links.
   - A GitHub rate limit stops the checks; what is left is listed as unchecked, with the reset time.
6. **Defaults:** sure finds start ticked; looked-up ones start unticked.

## Chat (`bot/repos.py`)
- **Item buttons:** `🏷 Tags · 🔎 Investigate · 📱 Open` (`mx:inv:<id>`).
- **Progress:** one message follows the job, edited at most every 3 s, then shows the result:
  - numbered entries with ☑/☐;
  - a toggle per entry, `rp:t:<id>.<rev>.<i>`;
  - `📎 Attach` (`rp:ok:<id>.<rev>`), which adds the ticked ones to what the item already carries;
  - `⭐ Attach + bookmark` (`rp:bm:…`);
  - `✏️ Review` (the console).
- **`/repos`:** the bookmarks, 10 to a page (`bk:pg:<p>`), with number buttons that open a card
  (`bk:o:<id>.<p>`). The card has `📥 From GitHub/the site` (`bk:gh`) and `🗑 Remove` (`bk:rm` → `bk:rmy`).
  Notes are edited in the console.

## Console
- **Item page:** a Repos section with the attached entries, or the pending checklist. Each row has a ⭐
  bookmark button, and the section has Attach ticked, Attach + bookmark, and Investigate (again), which waits
  on the job.
- **Media → Repos tab:** the bookmarks with search, an inline note, the From button, the items that carry each
  one, and Remove.
- **API:** `POST /api/media/{id}/investigate`, `PUT /api/media/{id}/repos {repos, bookmark}`,
  `POST /api/media/{id}/bookmark {key}`, `GET /api/bookmarks`, `PUT/DELETE /api/bookmarks/{id}`,
  `POST /api/bookmarks/{id}/about`. None is open to the service token.

## Not in this version
- Frame OCR of on-screen text.
- Editing a note in the chat.
- Tracking renames of an already-bookmarked repo.
