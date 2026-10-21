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
    - The provider key is replaced by `***` in every error text, and HTTP errors are not chained, so a key quoted by the transport can't leak through `__cause__`. Nothing logs headers.
  - No retry and no cross-engine fallback.
- `src/argus/media.py`: `extract_audio_chunks(src, out_dir, *, chunk_seconds, ffmpeg=None, timeout_s=3600) -> list[Path]`.
  - It is blocking: the engine runs it in a worker thread.
  - It uses the resolved ffmpeg (`resolve_ffmpeg`) with the segment muxer, and returns `chunk_000.mp3`, … in order.
  - `RuntimeError` names the cause: no ffmpeg, ffmpeg's exit code and stderr tail, the timeout, or no output.
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
- **`set_transcriber` stores the whole block in the library** (WP-A's registry). A later edit to the file's `transcriber` values is therefore shadowed by a stored `/engines transcriber` or console choice, until that setting is changed again.
- **`ARGUS_WHISPER_MODEL` is no longer read.** The model is `transcriber.faster_whisper.model`. The startup-WARNING table (`registry.REMOVED_ENV_KNOBS`, whose comment says the WP that stops reading a knob adds it there) belongs to WP-A's file, and §7 does not list the knob. It is left to the orchestrator: one line. The variable is not set in this PC's `.env`.

### Behaviour changes for handoff.md

- The whisper model, device and compute type come from `argus_engines.json` `transcriber.faster_whisper` (or the local file), not from `ARGUS_WHISPER_MODEL`. `handoff.md` §"Whisper on Apple Silicon" still names the old variable; that is WP-H's file.
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
