# Next session — Slipstream

Paste into a new Claude Code session started in this folder.

---

Continue Slipstream (a lean fork of Splash for Qwen3.8-Flash-Next, becoming a
framework for new models). First read `.agents/status.md`, the top 3 entries of
`.agents/journal.md`, `.agents/memory.md`, `.agents/decisions.md`, then
`docs/architecture.md`.

Next task: the better draft head (status.md "Next" 1): literature survey, then a
plan, keeping output quality exactly the model's. After every change: `make
check-native-cpu check-native-metal test-python`, then the identical-output check on
the real model (10-prompt suite, greedy, compare with a saved Splash run), all
engine runs through `dev/benchmarks/guarded.py`. Commit per step; write
plain-language notes in `.agents/`.
