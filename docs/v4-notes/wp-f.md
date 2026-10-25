# WP-F — transcriber engines (Task 13)

Spec: `docs/argus-v4-design.md` §3 WP-F, with §2.2 (`TranscriberConfig`, the library key `engines.transcriber`). Transcription is now driven by the engines config's `transcriber` block: the YouTube step order, one ASR engine, and that engine's data. Model ids live in `argus_engines.json`, not in code.

### What changed

- `src/argus/engines/transcribers.py` (new):
  - `asr_engine(transcriber, cfg)` builds the engine `transcriber.asr_engine` names, from its block. The block is read here, at use time. Startup never reads it: `TranscriberConfig`'s blocks default to `{}` and load without complaint.
  - `FasterWhisper`: the block `{"model", "device", "compute_type"}` maps onto `faster_whisper.WhisperModel(model, device=…, compute_type=…)`. One model is loaded per process; a changed block loads its own. `transcribe` keeps v3's `vad_filter=True` and the rolling-caption dedup. `backend = "faster_whisper:<model>"`.
  - `OpenAIAudio`: the block `{"provider", "model", "chunk_seconds", "language"}`. The provider comes from `cfg.providers`, its URL from `registry.base_url` and its key from `registry.api_key`.
    - The audio is cut by `media.extract_audio_chunks` into 16 kHz mono MP3 chunks of `chunk_seconds`, in a temporary directory.
    - Each chunk is one multipart `POST {base_url}/audio/transcriptions` with `file`, `model`, `response_format=json` and `language` when it is not null. The headers are the provider's `headers` plus `Authorization: Bearer <key>`. The timeout is the provider's `timeout_s`.
    - The chunks run in order, and their texts are joined with newlines.
    - `backend = "openai_audio:<provider>/<model>"`.
  - Errors:
    - `EngineConfigError` (the registry's) for a configuration that cannot run. That covers a block missing a key (named, with "the transcriber block in argus_engines.json"), an unknown provider, a `chunk_seconds` that is not a positive integer, `faster-whisper` not installed, and a missing key. The registry's text names the env var: "provider groq: GROQ_API_KEY is not set".
    - `TranscribeError` for a transcription that ran and failed:
      - an HTTP status of 400 or more, with the status and a body excerpt (413 adds the `chunk_seconds` hint);
      - a transport error;
      - an answer that is not JSON or has no string `text`;
      - a chunk over `MAX_UPLOAD_BYTES`, checked for every chunk before the first upload;
      - an ffmpeg failure;
      - a faster-whisper failure, with the cause chained;
      - **an all-empty result** ("produced no speech"). A silent empty transcript is never returned.
    - The provider key is replaced by `***` in every error text.
    - A transport error is raised outside its handler, so neither `__cause__` nor `__context__` holds a key the transport quoted.
    - Nothing logs headers, which a `caplog` test checks with every logger at DEBUG.
  - No retry and no cross-engine fallback.
- `src/argus/media.py`: `async extract_audio_chunks(src, out_dir, *, chunk_seconds, ffmpeg=None, timeout_s=3600) -> list[Path]`. Fix round 1 changed it from blocking and segment-muxed to the description below.
  - It uses the resolved ffmpeg (`resolve_ffmpeg`).
  - It decodes the audio once to 16 kHz mono raw PCM, cuts it at exact sample offsets, and encodes each cut to MP3 on its own. It returns `chunk_000.mp3`, … in order, and the intermediate PCM is removed.
  - The chunks are contiguous and nothing is lost: each chunk decodes alone to exactly its share. They don't overlap, so no word is sent twice.
  - `timeout_s` bounds each ffmpeg run.
  - A cancel or a timeout kills the running ffmpeg and awaits it, so no child outlives the call holding files open.
  - `RuntimeError` names the cause: no ffmpeg, ffmpeg's exit code and stderr tail, the timeout, or no audio.
  - On POSIX it runs with `start_new_session=True`.
- `src/argus/transcribe.py`: `transcribe_url(url, *, transcripts_root, media_root, quality="auto", library=None, engines=None)`. The existing callers' arguments are unchanged.
  - `engines` defaults to `registry.current_config()`. When startup did not load the config, it loads it now, and a failing load is reported.
  - YouTube runs the steps of `transcriber.youtube` in order until one succeeds. Other platforms run `asr`. Only the listed steps run: `["captions"]` never downloads or transcribes.
  - The ASR engine is built **before** any download, so a configuration error (a missing key, a bad block) costs no media download.
  - Failed steps are recorded:
    - in `backend`, as `captions→faster_whisper:small`;
    - in the file header, as a `# captions failed: <why>` line under `# backend: …`;
    - in the error of a fully failed transcript: every step's reason, joined.
  - Still fail-soft: a `TranscriptResult(ok=False, error=…)` for every config, download and ASR failure, as before. The bot's `/transcript` shows the text unchanged.
  - The `TranscriptResult.backend` strings are `captions`, `faster_whisper:<model>`, `openai_audio:<provider>/<model>`, with `<failed step>→` before them.
- `argus_engines.json`: the `transcriber` block, with the spec's data verbatim: `youtube ["captions","asr"]`, `asr_engine "faster_whisper"`, `faster_whisper {"model":"small","device":"cpu","compute_type":"int8"}`, `openai_audio {"provider":"groq","model":"whisper-large-v3-turbo","chunk_seconds":600,"language":null}`.
- `src/argus/bot/engines_cmd.py`: `/engines transcriber [faster_whisper|openai_audio [provider/model]]`.
  - With no argument, it shows the engine's backend and the YouTube order.
  - `openai_audio <provider>/<model>` replaces only the provider and model of the `openai_audio` block. `chunk_seconds` and `language` stay as they are.
  - It writes through `registry.set_transcriber`, so the change persists under `engines.transcriber`.
  - An unknown engine, an unknown provider, a malformed `provider/model`, or a provider/model given to `faster_whisper` is refused, and nothing persists.
  - A choice that can't run yet (for example, no `GROQ_API_KEY`) is saved, with a `⚠` line naming the problem. The transcription is what fails on it.

### Decisions where the spec is silent

- **Config errors and run errors use different classes.** `EngineConfigError` is used for configuration, as everywhere else in v4. `TranscribeError` is used for a transcription that ran and failed. `transcribe_url` turns both into the result's `error`, prefixed "transcriber configuration:" or "ASR failed (<backend>):".
- **The backend string for faster-whisper changed** from `whisper` to `faster_whisper:<model>`, to match the `openai_audio:<provider>/<model>` pattern. Nothing in the code reads the old value. The library's `meta.backend` of earlier transcripts keeps `whisper`.
- **The faster-whisper model can't be changed from chat.** The spec's `[provider/model]` doesn't fit a local engine, so `/engines transcriber faster_whisper <x>` is refused with a pointer to `transcriber.faster_whisper.model`.
- **The upload cap is a code constant** (`MAX_UPLOAD_BYTES`, the OpenAI and Groq limit), because the spec's block has no key for it. At the shipped `chunk_seconds`, a chunk is a few MB.
- **`engines.transcriber` holds only what differs from the files** (fix round 1, `registry.set_transcriber`), so a file edit keeps reaching every key a choice left alone. See the fix-round section.
- **`ARGUS_WHISPER_MODEL` is no longer read.** The model is `transcriber.faster_whisper.model`. Since fix round 1, startup warns once if the variable is set (`registry.REMOVED_ENV_KNOBS`, approved by the orchestrator), and spec §7 lists it as removed.

### Behaviour changes for handoff.md

- The whisper model, device and compute type come from `argus_engines.json` `transcriber.faster_whisper` (or the local file), not from `ARGUS_WHISPER_MODEL`, which startup now warns about. `handoff.md` §"Whisper on Apple Silicon" says so (fix round 1).
- `/engines transcriber openai_audio groq/<model>` switches transcription to Groq's Whisper API. It needs `GROQ_API_KEY`; without it, every transcription answers "transcriber configuration: provider groq: GROQ_API_KEY is not set", and nothing is downloaded.
- The transcript caption and the file header read `faster_whisper:small` (or `openai_audio:groq/…`) instead of `whisper`. A YouTube video whose captions failed reads `captions→…`, and its header has the caption error.
- A YouTube order of `["captions"]` no longer falls back to ASR; `["asr"]` skips captions.

### New env vars

None. The provider key names already in `argus_engines.json` apply: `GROQ_API_KEY` for the shipped `openai_audio` provider. `ARGUS_WHISPER_MODEL` is no longer read (see above).

### Test files touched

- New: `tests/test_transcribers.py`:
  - selection, and the block checked at use time only;
  - the key error naming the env var, and the block errors;
  - the `openai_audio` request shape and the chunk concatenation (httpx `MockTransport`);
  - the null language omitted;
  - HTTP 429 without the key, the oversized chunk refused before any upload, 413, the unusable answers, the transport error, all-empty → "no speech", and the extraction failure;
  - the `faster_whisper` mapping and the model reload;
  - `extract_audio_chunks` on a generated tone (real local ffmpeg), and its errors;
  - the `engines.transcriber` persistence round-trip through the library;
  - the shipped block;
  - `/engines transcriber` switch, persist, show, warn and refusals.
- Rewritten: `tests/test_transcribe.py`:
  - captions; ASR with the configured engine; library media reuse;
  - the captions→ASR order recorded in `backend` and the header; `["asr"]` skipping captions; `["captions"]` never falling back;
  - a config error surfacing before any download (the real engine, no key);
  - a `TranscribeError` reported, not raised; no speech;
  - the bot path reading the registry and the persisted choice; an unloadable config reported.

### Verification evidence (2026-10-21)

Logs and scripts are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t13\`. No call reached Groq, OpenAI or any live service.

- **RED** (`red.log`): both test files fail at collection with `ModuleNotFoundError: No module named 'argus.engines.transcribers'`.
- **GREEN, focused** (`green.log`): both files pass. After the last `engines_cmd` edit (`green2.log`), they pass again together with the engines, webapp-API and media test files, with no warnings summary.
- **Full suite, proxy URL on a closed port** (`suite.log`): only the 4 live-proxy tests fail (2 failures in `test_llm.py`, 2 errors in the `test_graph.py` fixture: "cannot list models at http://127.0.0.1:9/v1/models"). There is no warnings summary.
- **Live checks** (`live_check.py`, `fw_block.py`, `live_check.log`), with the shipped config and a clip made by the local Windows TTS (`speech.wav`):
  - `nokey`: `EngineConfigError: provider groq: GROQ_API_KEY is not set`. `transcribe_url` returns `ok=False` with "transcriber configuration: provider groq: GROQ_API_KEY is not set", and nothing is downloaded.
  - `chunks`: `extract_audio_chunks` cut the clip into ordered MP3 chunks.
  - `fw` with the shipped `small` block: `TranscribeError: RuntimeError: mkl_malloc: failed to allocate memory`, from `ctranslate2` loading the model. The PC had about 0.2 GB of free RAM, both during and after the suite. The error path surfaced as designed.
  - `fw_block.py` with `{"model":"tiny","device":"cpu","compute_type":"int8"}` (cached weights, `HF_HUB_OFFLINE=1`): `backend=faster_whisper:tiny language=en`, text "The quick brown fox jumps over the lazy dog. / Argus transcribes this short clip."
- **Not verified:** the shipped `openai_audio` model id `whisper-large-v3-turbo` against Groq. The spec asks for it to be confirmed before the block is committed, but this PC has no `GROQ_API_KEY` and the task forbade live calls. It remains unverified data until the orchestrator's Groq check runs. If the id differs, only `argus_engines.json` changes.

### Fix round 1 (review of abd1463)

The review is `.superpowers/sdd/2026-10-19-argus-v4/task-13-review.md`. The orchestrator approved edits outside WP-F's files for items 1 and 2 only: `src/argus/engines/registry.py`, the spec §7 "Removed" list and `handoff.md` §4. Item 7 needed `src/argus/bot/media.py`.

1. **A stored choice no longer shadows file edits.**
   - `registry.set_transcriber(t)` keeps its signature, so `/engines transcriber` and the console's `PUT /api/engines` are unchanged callers. It stores only `t`'s difference from the files' block, with the schema defaults filled in: dicts key by key, a list or scalar whole.
   - A value set back to the files' value leaves the library.
   - `_deep_merge` no longer applies the engine-spec "a dict naming `provider` replaces whole" rule inside `transcriber`. A partial `openai_audio` row (just `model`) therefore merges key by key, as a local file's `transcriber` edit does.
   - **Pre-fix rows:** a stored row holding every `TranscriberConfig` field was written before this rule. It still loads, and its in-memory layer keeps only what a choice could set there: `asr_engine`, and `openai_audio.provider`/`model` where they differ from the files. The copied rest (`youtube`, the `faster_whisper` block, `chunk_seconds`, `language`) follows the files again. Startup writes nothing; the pruned layer is persisted at the next change.
     - The trade-off: a pre-fix row's `provider`/`model` that differs from today's file is kept, as a user choice, even if it was only an older file value.
     - This PC's library has no `engines.*` rows (read-only check).
2. **`ARGUS_WHISPER_MODEL` warns at startup.** It is in `REMOVED_ENV_KNOBS`, pointing at `argus_engines.json transcriber.faster_whisper.model` and `/engines transcriber`. It is also in spec §7 "Removed", and `handoff.md` §4 no longer advises setting it.
3. **Contiguous chunks**, via the PCM-cut approach described above.
   - Experiment (`scratchpad\t13fix\exp_gapless.py`): a 7.3 s tone cut into 2 s PCM slices, each encoded on its own. MP3 decodes to `[2.0, 2.0, 2.0, 1.3]`, a sum of 7.3. FLAC gives the same.
   - So MP3 was kept: it is a fraction of FLAC's size against the upload cap.
4. **Cancel hygiene.**
   - Each ffmpeg runs as an asyncio subprocess (`media._run_ffmpeg`). A cancel or the timeout kills the child and awaits it before propagating.
   - `OpenAIAudio` awaits the chunker directly, no longer in a worker thread. The `TemporaryDirectory` cleanup therefore runs after the child is gone, and the `CancelledError` is not replaced by a Windows `PermissionError`.
5. **`asr_available()`** uses `importlib.util.find_spec`, so it no longer imports faster-whisper on the event loop.
6. **New tests** (see the list below).
7. **The bot's "Transcribing…" note** names what will run: for example `🎙 Transcribing via captions → faster_whisper:small (…)` or `via openai_audio:groq/…`.
   - It mentions the model download only for faster-whisper.
   - With an unloadable config it says "🎙 Transcribing…", and `transcribe_url`'s configuration error follows.
   - `transcribe.engines_config` is now public, for this caller. `transcribers.asr_label` gives the display string, and `/engines transcriber` reuses it.

Review Minor 5 (the configured vs detected language) is parked by the orchestrator and is unchanged.

**Tests added or changed:**
- `test_transcribers.py`:
  - the reviewer's shadowing probe, plus a file-corrected model reaching a stored provider choice;
  - a value set back to the file leaving the library;
  - a pre-fix whole-block row loading;
  - the startup warning, through `bot.app._post_init` via `test_engines_cmd`'s helper;
  - chunk durations summing to the source (on a path with a space and non-ASCII characters);
  - a cancel killing and awaiting ffmpeg, and the folder removable;
  - a timeout killing ffmpeg;
  - a cancelled `openai_audio` leaving no temp folder;
  - `asr_available` not importing the package (a subprocess check);
  - the provider's `timeout_s` reaching `_client` and httpx;
  - no key or Authorization header in any log at DEBUG;
  - no `__context__` on a transport error;
  - the `/engines transcriber` persistence assertions updated to the stored difference.
- `test_transcribe.py`: the bot note per engine and per YouTube order, and the broken-config note.

**Evidence** (`scratchpad\t13fix\`):
- `red.log`: 29 failed (the new behaviour, and the async chunker the old fakes did not match).
- `green.log`: the WP-F, engines, webapp-API and media files, 260 passed.
- `suite.log`: the full suite.
- `live_check.log`: the no-key error again, and the async chunker on the TTS clip.
