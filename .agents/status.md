# Status — qwen4exp (Qwen3.8-Flash-Next) in Splash

**Updated:** 2026-09-22 08:30 PDT by claude-code
**Branch:** `qwen4exp-review` (not pushed; never push to upstream `incoai/splash`)

## In one line

Quality matches llama.cpp V3, decode is 35–50 tok/s (llama.cpp: 18, or 27
with its draft head), and prompts read at ~180–670 tok/s (llama.cpp ~110–367)
— same 36 GiB expert budget.

## Numbers (one warm engine, 384 tokens, guess cap 5)

| Prompt | Greedy | Sampled 0.7 | Prompt reading |
|---|---|---|---|
| Short, 65 tok | 49.7 | 46.0 | ~180–220 tok/s |
| Code, 1,495 tok | 37.7 | 35.3 | ~585 tok/s |
| Long, 9,129 tok | 36.7 | 36.9 | ~670 tok/s (4,096-token chunks) |

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

## Open work (measured gaps, largest first)

1. ~~8-bit linear decode kernels~~: measured at the bandwidth floor (3.4 GB
   of 8-bit weights a step in 10.8 ms); every tile/grid variant is slower.
2. **SSD miss reads** ~17 ms/step: lookahead foresees 66% of used experts
   (top-16 would 79%). A second prediction two layers ahead was tried and
   rejected (misses 59 -> 56 at best, tok/s down: its reads compete).
3. **GPU<->host hand-offs** ~8 ms/step (48 x ~164 us): removing them needs
   GPU-side routing with a GPU-visible slot table.
4. **Drafting** ~13 ms/step (two blocking GPU round trips per guess).
5. Expert quantization: group-32 and range-search simulated *worse* than the
   current format despite lower weight error - unexplained, worth a look.
6. 27B regression check needs a 4-bit 27B package.

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
