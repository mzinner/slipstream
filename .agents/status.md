# Status — Slipstream

**Updated:** 2026-09-22 21:40 PDT by claude-code
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

1. **Better draft head, to pass ~45 tok/s** (Nitin chose this, 2026-09-22). Start
   with a literature survey (including https://huggingface.co/papers/2609.26796),
   then a plan. Quality must hold: checking guesses keeps the model's exact output,
   so re-run the identical-output check, the bf16 top-pick/KL check
   (`models/qwen4exp/bench/compare_logits.py`) and the benchmarks.
2. Optional cleanup: drop Homebrew/release packaging and `install/` (wired into 13
   tests, so a step of its own); turn `QwenTarget` into an explicit model interface.
3. Launcher: `~/models/bin/splash-flashnext-server.sh` serves Slipstream with
   `REPO=<this folder>`.

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
