# Status — qwen4exp port

**Updated:** 2026-09-21 23:30 PDT by claude-code
**Branch:** `qwen4exp-review` (built on Gemini's `qwen4exp-gemini`; Gemini's uncommitted work is the first commit)

## Where we are in one line

Flash-Next runs end to end in Splash, including the server and tool calls, at
~2.5 tok/s. llama.cpp V3 does 18 (27 with its draft head). Nitin asked for a
redesign from a per-component profile, at llama.cpp's memory budget. The
profile and plan are done; implementation has not started.

## The plan (full write-up: https://claude.ai/artifact/PB7F2nj2KRtST91NfMGoEQ)

Floor for one generated token on this Mac: ~5 GB read at 291 GB/s ≈ 17 ms.
Budget: 36 GiB expert cache, wired limit 58 GiB, same as llama.cpp.

1. **Rewrite hyper-connection kernels** — ~224 of ~300 ms per decode step today
   (one threadgroup per row). Floor 4.4 ms. No quality risk. START HERE.
2. **Expert fetch** — direct parallel pread (SSD measured 15 GB/s), 36 GiB
   cache, frequency-aware eviction. Today ~105 ms/token via 16 KB page faults.
3. **Single-row decode kernels** for dense, experts, GDN (8-row kernels now).
4. **Remove per-layer stop-and-wait**: GPU-side slot table, lookahead prefetch.
5. **Prefill streaming**: never bind whole layer files; double-buffered reads.
6. **MTP draft head** in place of the zero placeholder.
7. Optional: 8-bit hyper-connection weights, only if quality holds.

Quality alongside: Splash 80% same-pick vs llama.cpp 89% on the code prompt;
drift grows with position (cause open). Sparse indexer unwired (>2K context).

## Tools (all in repo)

- `generate-sample`: `SPLASH_PROFILE_STEPS=N`, `SPLASH_PROFILE_PREFILL=1`,
  `SPLASH_ROUTE_LOG=path`, `SPLASH_DUMP_PREFILL_LOGITS=path`; several prompts
  separated by `;` run on one warm engine. **Build it explicitly** — `make`
  does not relink it.
- `dev/benchmarks/qwen4exp/`: `bench_splash.py --one-process`, `bench_llama.py`,
  `reference_logits.py` (bf16 ground truth), `compare_logits.py`,
  `write_llama_kld_base.py`, `simulate_cache.py`, `probes/`.

## Do not touch

- `~/models/qwen38-flash-next-bf16` — 338 GB source.
- `~/models/qwen38-flash-next-v3` — the model Nitin runs daily (llama.cpp, port 8080).
- `splash2/` checkout — serves the 27B.

## Before merging this branch

Check 27B speed on it: Gemini's command-splitting in MetalBackend.mm may
trigger for the 27B.
