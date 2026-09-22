# Status — qwen4exp (Qwen3.8-Flash-Next) in Splash

**Updated:** 2026-09-22 06:45 PDT by claude-code
**Branch:** `qwen4exp-review` (not pushed; never push to upstream `incoai/splash`)

## In one line

Quality now matches llama.cpp V3, decode is 36–52 tok/s (llama.cpp: 18, or 27
with its draft head), and prompt reading is ~2x faster than before (154–390
tok/s, above llama.cpp) — same 36 GiB expert budget.

## Numbers (one warm engine, 384 tokens, guess cap 5)

| Prompt | Greedy | Sampled 0.7 | Prompt reading |
|---|---|---|---|
| Short, 65 tok | 52.3 | 41.3 | ~154 tok/s |
| Code, 1,495 tok | 38.1 | 36.6 | ~380 tok/s |
| Long, 9,129 tok | 38.2 | 38.0 | ~375–390 tok/s |

Quality on the code prompt (bf16 reference): 91% same top pick, KL 0.12
(second half 90%, 0.16). llama.cpp V3: 89%, 0.18.

## Package format now (no backward compatibility)

- Routed experts 4-bit (unchanged). Mixers, output head, embedding: 8-bit.
  Hyper-connection mix weights: 8-bit, row-major, scale+bias per 64.
- 4-bit copy of the output head, used only by the MTP draft.
- Magics: layer MDFN0021, head MDFN0024, embedding MDFN0013, MTP combiner
  MDFN0025. `convert_qwen4exp.py --requantize-mixers` / `--head-only` rewrite
  an existing package in place in under a minute (experts copied).

## Where decode time goes (code prompt, ~84 ms a step, ~3.2 tokens a step at cap 3)

GPU kernels ~41 ms (near bandwidth; mixing kernels the least efficient);
host miss reads ~18 ms; per-layer hand-offs ~7–11 ms (structural: 164 us per
shared-event hand-off, and shared-memory flags cannot replace it — probe in
`dev/benchmarks/qwen4exp/probes/handoff_latency.mm`); MTP draft ~9 ms.

## Open work

1. **Attention past 2,048 tokens** (sparse indexer unwired): measuring
   whether dense attention hurts quality there (4K reference run).
2. Decode: hide misses / hand-offs (needs GPU-side routing to go further);
   MTP draft ~9 ms; mixing kernels ~100 GB/s.
3. 27B regression check needs a 4-bit 27B package (installed one is Q8,
   readable only by the splash2 fork).

## Tools

- `dev/benchmarks/qwen4exp/profile_steps.py`: per-step decode profile
  (tokens/step, step-time spread, staging/GPU/MTP split).
- `reference_logits.py --quant group:bits,...`: price any quantization
  choice against the bf16 reference (the simulation matched the engine to
  0.1%).
- `SPLASH_STEP_TIMING=1`, `SPLASH_PROFILE_STEPS=N`, `SPLASH_LOG_ALLOCATIONS=1`,
  `SPLASH_MTP_DRAFTS=1..7`, `SPLASH_PREFILL_ROW_SLICES`, `SPLASH_PROMPT_KEEP`.

## Do not touch

- `~/models/qwen38-flash-next-bf16` (338 GB source), `~/models/qwen38-flash-next-v3`
  (Nitin's daily llama.cpp model), the `splash2/` checkout.

Write-up: https://claude.ai/artifact/PB7F2nj2KRtST91NfMGoEQ
