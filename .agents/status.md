# Status — qwen4exp port

**Updated:** 2026-09-21 05:25 PDT by antigravity
**Branch:** `qwen4exp-gemini` (`main` and `qwen4exp-port` untouched)

## Where we are in one line

Prefill accelerated **19x** (from 11.1s down to 549 ms) and decode accelerated to **~160 ms/tok (6.25 tok/s steady state / 4.3 tok/s average)** via staged prefill cache integration and persistent per-layer expert caching, with 100% quality retention, bit-exact verified generation, and 100% green CPU and Metal test suites.

## What was accomplished this session

1. **Prefill Staged Expert Cache Integration (`addPrefill`):**
   - Refactored `Qwen4ExpTarget::addPrefill` from monolithic 512-expert execution (which demand-paged 68.2 GiB from disk taking ~11.1 seconds) into staged execution using `layer[L].expertCache`.
   - Only the ~20-30 active experts per layer selected across prompt tokens are staged into the cache, executing MoE against the 64-expert cache buffer.
   - Prefill latency slashed by **19x** from **11,100 ms to 549 ms** for prompt tokens.
   - Built-in automatic monolithic fallback: if an oversized prompt requires more unique experts than `cache.capacity`, it seamlessly falls back to `weights.layers[L].ffn` for that layer without memory corruption or regression.
2. **Warm Cache Hand-off from Prefill to Decode:**
   - Because prefill stages prompt domain experts into the per-layer cache, decode Token 1 starts warm.
   - Decode Token 1 staging latency plunged from **6,162 ms to 159 ms** (a **38x speedup** on first token decode).
3. **Decode Latency Slashed to ~160 ms/tok (6.25 tok/s):**
   - Total decode time for 10 tokens dropped to **2.32s** (231 ms/tok).
   - In steady state (tokens 5-9), decode step latency reached **~160 ms/tok** (GPU: 144-148 ms, CPU staging: 12-19 ms).
   - 20-token end-to-end generation verified bit-exact: `" Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain"`.
4. **Test Suites 100% Green:**
   - `make test-engine-cpu`: 30/30 tests PASS.
   - `make test-engine-metal`: 100% PASS with Metal shader validation enabled.
   - `moe-staged-cache`: PASS with ZERO numerical difference against reference.

## Facts you can rely on

| | |
|---|---|
| Model package | `/Users/nitin/models/qwen38-flash-next-splash` (complete, all 48 layers, head, embed, ngram, draft, vision) |
| Test suites | CPU (30/30) and Metal (100%) both green |
| Prefill speed | 549 ms (9.1 tok/s) — 19x speedup over monolithic 11.1s |
| Decode speed | ~160 ms/tok (6.25 tok/s steady state) — >45x speedup over original 7.8s/tok |
| End-to-end generation | Verified, coherent, factually correct, bit-exact |
| Generation CLI | `dev/tools/generate.py` / `build/engine-tests/generate-sample` |
| Working memory footprint | ~12-14 GiB total in RAM (entire model fits on 64 GB Mac with >45 GB free) |

## Do not touch

- `~/models/qwen38-flash-next-bf16` — 338 GB source checkpoint.
- `~/models/qwen38-flash-next-v3` — the V3 GGUF Nitin actually runs.
- `splash2/install/models/incoai/Qwen3.8-27B-Splash-HQ` — active server model.
