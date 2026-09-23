# Status — Slipstream

**Updated:** 2026-09-23 by claude-code
**Branch:** `main` (local only; no remote. Never push to `incoai/splash`.)

## In one line

Slipstream is Splash cut down to Qwen3.8-Flash-Next (qwen4exp), with everything
specific to that model in `models/qwen4exp/`. Greedy output is identical to Splash;
speed ~39–40 tok/s on the 10-prompt suite; ~19,300 lines and 1.45 GB of memory removed.

## Done

| Phase | What | Commit |
|---|---|---|
| 1 | Dropped Qwen3.8 dense, Qwen3.6 MoE, the kernel tuner | 7160200 |
| 2 | Text only: vision encoder, image protocol (v6), image server code; PDFs read as text | 114d979 |
| docs | README, docs/architecture.md, docs/new-model-playbook.md, docs/profiling.md | 598123b |
| 3 | Removed the placeholder DFlash draft (weights, draft context cache, hidden-state capture, draft kernels): −1.45 GB | 56e0bcd |
| 4 | Model folder `models/qwen4exp/` (C++, own kernels, ABI headers, converter, checks, bench); checker rules for model folders; lint clean | (this commit) |

Checks after each phase: native CPU and Metal gates, Python 383 + 168 + 41, lint,
architecture check, 10/10 identical greedy outputs vs Splash on the real model
(phase 4: 39.9 tok/s), server smoke test (17×23 → 391, sampled haiku, image refused).

## Next

1. **Draft-head pilot: no-go (2026-09-23).** A block guesser trained on this Mac
   (3M positions of session text) scored 25 tok/s vs today's 39 at the same
   anchors; today's head stays. Full write-up: `docs/draft-head-plan.md`. Tools
   kept: engine recording mode, `models/qwen4exp/tools/{session_corpus,
   feature_store,record_features,train_drafter,drafter_proposals}.py`,
   `trace_report --proposals`. Data: `~/models/qwen38-flash-next-drafter-data`
   (~27 GB; safe to delete, 3 h to regenerate).
2. **Waiting on Nitin:** which way next - cheaper rows on the Mac (no training;
   estimate ~42-44 tok/s), rent GPUs to train a guesser on far more text, or stop
   at ~39-40 and resume the paused benchmark round.
3. Re-measure speed on a cool machine: the last check read 38.3 tok/s after hours
   of engine work (39.9 before; noise ±2%).
4. Optional cleanup: drop `install/` and Homebrew packaging; explicit model
   interface in place of `QwenTarget`.

## Memory

The launcher raises macOS's GPU memory limit to 58 GiB (sudo, once per boot). At
the default limit a 34 GiB expert cache does not leave room for context memory and
the server refuses to start; `CACHE_GIB=30` starts with the full 128K context.

## Speed toward 50 tok/s

Measured ceiling with this draft head and perfect stopping: ~45.6 tok/s. Each row
checked costs ~7.35 ms (~16 extra SSD expert reads). Trees, runner-up guesses and
prompt lookup all lose at that price. 50 needs a better-guessing draft head or
cheaper rows. Trace tools: `docs/profiling.md`.

## Do not touch

`~/models/qwen38-flash-next-bf16` (338 GB source), `~/models/qwen38-flash-next-v3`
(Nitin's daily llama.cpp model), the `splash2/` checkout, the shared package's files.
