# Status — Slipstream

**Updated:** 2026-09-23 06:10 PDT by claude-code
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

## Running now (2026-09-23 07:03)

Benchmark round 2 (Nitin chose it after the speed work): Slipstream, then
llama.cpp V3, then Qwen3.8-27B HQ, 9 suites each, seed 1234, unattended:
`benchmarking/model-quality-bench/run_round_2.sh`, log
`runs/round2-orchestrator.log`. Every server runs under guarded.py; llama.cpp
keeps the 58 GiB GPU limit (no sudo) with the guard as its safety. No server is
left running at the end. Then: compare (bench.compare), judge sessions
(bench.judge), check Slipstream's answers equal Splash's where both ran,
update BENCHMARKS.md and the hub.

## Next

1. **Cheaper check steps (Nitin chose this, 2026-09-23).** Done so far:
   read-ahead 6 + uneven cache slots, +3-4% (41.4 tok/s on the 10-prompt
   suite, 40.9 on session prompts). Still open, by expected value:
   - tried, no gain: reading experts several rows agree on
     (SPLASH_LOOKAHEAD_CONSENSUS), plain least-recent eviction;
   - not done on purpose: a larger cache. Free memory already dips to ~7.5 GiB
     during runs; +1-2 GiB of cache risks the freezes of 2026-09-22 for an
     estimated 2%;
   - left: fewer GPU-CPU hand-offs (~2 ms a step). The GPU still idles ~15 ms
     a step waiting for ~50 on-demand reads; only better prediction or more
     memory moves that.
2. Draft-head pilot: no-go (docs/draft-head-plan.md); tools kept. Data in
   `~/models/qwen38-flash-next-drafter-data` (~27 GB), safe to delete.
3. Optional cleanup: drop `install/` and Homebrew packaging; explicit model
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
