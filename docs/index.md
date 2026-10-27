# Argus

A single-user Telegram bot for research you can check, and for saving media.

Ask it a question and it plans the research with you, searches live, reads
every source into notes with verbatim quotes, writes a cited report, has a
separate judge check each claim against the quotes it cites, and cuts what
fails. You get Markdown, a PDF and an `evidence.json`, and every claim in the
report can be traced back to the page it came from.

- **Two checkpoints with you.** A plan gate before anything is fetched — only
  the links you put in the question are read first — (approve, edit, set
  length, depth, engines and recency) and a report gate before delivery (send,
  extend, revise, cancel).
- **Trust first.** Links come only from live search or from you, quotes are checked word
  for word, unsupported claims leave the body, and anything unchecked is
  announced at the top of the report.
- **Swappable engines.** Every model role can run on the free proxy, a paid
  API or a local model; the report records which model actually served each
  call.
- **A console in Telegram.** The Argus menu button opens a Mini App with your
  runs, live progress, plan controls, a report reader with tap-to-verify
  citations, and the engines settings.
- **Media.** YouTube, X, Reddit and Instagram downloads into your vault, with
  transcription.

## Where to go next

- [User guide](help.md): every command, the two gates, engines and the console.
- [How it works](how-it-works.md): the pipeline, engines, run configuration
  and project layout.
- Setup and source: the
  [repository README](https://github.com/AlbertHakobyan070/argus-research-bot#readme).
