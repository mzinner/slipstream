# TokenSpeed & Slipstream Optimization Plan for Qwen3.8-Flash-Next

**Branch:** `feat/tokenspeed-optimizations`
**Target Model:** `~/models/swift-qwen38-flash-next-v3/prepared` (`qwen4exp`)

---

## 1. Context & Architectural Comparison

TokenSpeed's Qwen3.8-Flash-Next implementation (`qwen4_exp.py`, `qwen4_exp_nextn.py`, `hyperconnection.py`, `qwen4_exp_ple.py`, `qsa/indexer.py`, `eagle.py`) targets multi-GPU NVIDIA GB300 systems where all 125B weights fit in HBM. Slipstream targets a single 64 GB Apple Silicon Mac where 512 routed experts per layer stream from SSD with an in-RAM LRU/LFU cache, lookahead routing, and single-lane PLD + MTP speculation.

Across a typical Slipstream verify step (~80–84 ms for ~4.4 tokens):
- **GPU execution:** ~41–46 ms (MoE experts, Q8 linear projections, GDN recurrence, paged Q8 attention, hyper-connections, PLE)
- **SSD expert miss reads:** ~15–17 ms
- **MTP draft head (1..5 steps):** ~8–13 ms
- **Host staging & routing between pipeline stages:** ~8–12 ms

---

## 2. Recommended Priority Order of Optimizations

### Priority 1: Narrow MTP Draft Execution to `live` Rows Only (`Qwen4ExpTarget.cpp`)
- **Inspiration:** TokenSpeed's `Qwen4ExpDraftAttentionDecoderLayer` (`qwen4_exp_nextn.py`) and `AcceptedPrefixPublisher` (`eagle.py`). On MTP step 0, TokenSpeed runs QKV projection and KV store only on the accepted prefix rows (`mtp.rows`, typically 1–4), then narrows to **1 live anchor row** for attention output, `o_proj`, `mlp_hyper_connection`, and MoE. On steps 1..K, it executes strictly 1 row per request.
- **Current Slipstream bottleneck:** In `Qwen4ExpTarget.cpp` (`runMtpDraft`), `kRows` is hardcoded to `ExecutionLimits::targetVerifyRows` (`8`). Even when `live == 1` (steps 2..5) or `live == 2..4` (step 1), Slipstream dispatches all 8 rows through:
  - `ops::Embedding::add` (`kRows`)
  - `hyper_connection_rms` on `mtpEmbed` (`kRows`) and `mtpHin` (`kRows × 4`)
  - `fcEmbedding` (8 rows) and `fcHidden` (`4 * lanes` = 32 rows)
  - `attentionHyperConnection` (`kRows` = 8 rows)
  - `inputProjectionQ8` (`attentionInput`, 8 rows), `addVerifyProjection` (8 rows), `addVerifyGate` (8 rows), `outputProjectionQ8` (8 rows)
  - `mlpHyperConnection` (`kRows` = 8 rows) and `combiner.mixer` (`kRows` = 8 rows)
  - `draftLogitsProjection` / `draftVocabProjection` (8 rows)
- **Fix:**
  1. Run the combiner, `attentionHyperConnection`, `inputProjectionQ8`, and KV projection/store on `live` rows (1..4 on step 0; 1 on steps 1..K).
  2. After `ops::PagedAttention::addVerify` (which scores the live rows), narrow the post-attention path (`addVerifyGate`, `outputProjectionQ8`, `mlpHyperConnection`, `encodeRouteScores`, `MoE`, `combiner.mixer`, and `draftLogitsProjection`) to **only the single anchor row (`live - 1`)** that produces the next draft token!
- **Expected Gain:** Cuts MTP non-MoE GPU work by 2×–8× AND eliminates MoE expert routing/staging/execution for rows `0 .. live - 2` on MTP step 0! When step 0 has 3–4 accepted tokens from the previous round, step 0 currently routes and stages experts from SSD for all 3–4 rows even though only the last row (`live - 1`) is used to pick `drafts[0]`. Only the KV cache needs rows `0 .. live - 2`.

### Priority 2: Fuse Residual Update + Next Mixer's `GroupedGemmaRMSNorm` (`hyper_connection.metal` & `Qwen4ExpTarget.cpp`)
- **Inspiration:** TokenSpeed's `combine_norm` (`_gated_residual_combine_norm_kernel` in `layernorm/triton.py` and `hyperconnection.py`).
- **Current Slipstream bottleneck:** At 95 sublayer boundaries per step (48 layers × 2 + MTP), `hyper_connection_update` / `hyper_connection_update_out` reads the 10,240-wide residual, adds `injection * block_output`, and writes the 10,240-wide updated residual to device memory. Immediately after, the next mixer's `hyper_connection_rms` reads that exact 10,240-wide tensor back from device memory to compute per-stream RMSNorm and write `normalized`.
- **Fix:** Add `hyper_connection_update_rms` and `hyper_connection_update_out_rms` in `hyper_connection.metal` with grid `{rows, hyperConnectionCount, 1}` (one threadgroup of 256 threads per `(row, stream)`). Each threadgroup updates its 2,560-element stream in registers, writes the updated residual to device memory, computes the stream's sum of squares in registers/threadgroup memory, and writes the normalized stream to `normalized`. Skip fusion only when entering `geometry.pleLayer` (Layer 1 attention) or when `SPLASH_CAPTURE_LAYERS` is active on that layer.
- **Expected Gain:** Eliminates ~95 kernel dispatches per step and saves reading 95 × `rows × 10,240 × 2` bytes from device memory (~7.6 GB of device memory reads per 4,096-token prefill chunk; ~15 MB per decode step).

### Priority 3: Fuse PLE `history + convolve + add` & Overlap CPU N-gram Gather with Layer 0 (`per_layer_embedding.metal` & `Qwen4ExpTarget.cpp`)
- **Inspiration:** TokenSpeed's `_ple_conv_state_kernel` (`ops/ple/triton.py`) and `Qwen4ExpModel._start_ple_prefetch` (`qwen4_exp.py`).
- **Current Slipstream bottleneck:**
  1. `addPleApply` runs 4 separate dispatches: `per_layer_embedding_history` (copies 9 history rows into `normalized`), `per_layer_embedding_gate` (computes `gated` and writes current rows into `normalized`), `per_layer_embedding_convolve` (dilated conv + SiLU added to `gated`), and `per_layer_embedding_add` (adds `gated` to `residual`).
  2. `gatherNgramRowsOnCpu` runs synchronously at Layer 1 instead of overlapping with Layer 0 GPU execution.
- **Fix:**
  1. Fuse `per_layer_embedding_convolve` + `per_layer_embedding_add` + direct history reading from `state` so `per_layer_embedding_history` and `per_layer_embedding_add` dispatches are eliminated.
  2. Start `gatherNgramRowsOnCpu` asynchronously on a GCD queue at the beginning of Layer 0 and wait on its `dispatch_group_t` right before Layer 1 encodes PLE.

### Priority 4: Additional Slipstream-Specific Optimizations
- **4a. Skip `encodePredictRoute` on Last Streaming Layer & Optimize Host Stage Overhead:** Check `hostSelect` + `sharedExpertGate` in `Qwen4ExpTarget.cpp`. Currently `sharedExpertGate` (lines 459–475) runs a scalar loop on the CPU over `rows × 2560` dequantizing Q8 weights on the single host staging thread while the GPU is paused waiting for `signalPipelineEvent`! Moving `sharedExpertGate` into the GPU's `moe_route_scores` kernel (or computing it in SIMD/parallel) removes host pause latency on every single layer (`48 layers × liveRows × 2560` float ops on CPU in the critical path between GPU stages).
- **4b. Fused `hyper_connection_rms` + `hyper_connection_down` Input Staging:** Evaluate if further reducing dispatch count in `addHyperConnection` improves decode latency.

---

## 3. Rigorous Verification & Measurement Protocol

Every single step will be tested in isolation on `feat/tokenspeed-optimizations`:
1. **Unit & Kernel Tests:**
   - `make && make test-engine-cpu`
   - `MTL_SHADER_VALIDATION=1 build/engine-tests/hyper-connection build/splash.metallib`
   - `MTL_SHADER_VALIDATION=1 build/engine-tests/per-layer-embedding build/splash.metallib`
   - `MTL_SHADER_VALIDATION=1 build/engine-tests/per-layer-embedding-state build/splash.metallib`
2. **Quality & Exact Token-for-Token Parity Check:**
   - Compare generated token IDs across all 10 prompts (2,560 tokens greedy) against the baseline output (`build/engine-tests/generate-sample`).
3. **Speed & Step-Component Telemetry (`speed_suite.py` + `SPLASH_TRACE`):**
   - Run under `dev/benchmarks/guarded.py` with no other model servers running.
   - Record overall `tok/s`, `tok/step`, `ms/step`, and the `trace_report.py` breakdown (`verify`, `mtp`, `mtp_gpu_a`, `mtp_stage`, `mtp_gpu_b`, `staging`, `gpu_wall`, `pure_gpu`, `host_stage`).
   - Keep only changes that demonstrably improve speed without regressing output quality.

---

## 4. Empirical Results Log

### 2026-10-14 — Priorities 1+2+PLE fusion measured (combined diff vs. HEAD)

**Protocol:** `decode-profile build/slipstream-v2.metallib ~/models/swift-qwen38-flash-next-v3/prepared --prompt-tokens 512 --cycles 8` under `dev/benchmarks/guarded.py`, no server on :8090, free-RAM floor never breached (low 6.5 GiB). Baseline = HEAD `Qwen4ExpTarget.cpp` (unfused orchestration) against the same metallib (new kernels are additive and simply uncalled); 3 baseline trials, 5 fused trials, 3 fused trials with `SPLASH_FUSE_PLE=0`, run in interleaved order. Both fusions are **bit-exact**: the Metal tests assert `memcmp == 0` for `hyper_connection_update_rms` / `hyper_connection_update_out_rms` and `per_layer_embedding_convolve_add_rms` vs. their unfused sequences, so token trajectories are unchanged.

| Metric (median fused GPU ms) | HEAD | Fused | Delta |
|---|---:|---:|---:|
| Prefill 512 rows | 11.45 | 11.31 | **−1.2%** (5/5 fused below 3/3 baseline) |
| B1 decode cycle | 3.79 | 3.69 | **−2.6%** (ranges overlap: 3.68–3.80 vs 3.65–3.73) |
| B4 decode cycle | 6.48 | 6.44 | −0.6% (noise) |
| Prefill dispatch count | 1671 | 1573 | −98 (−5.9%) |

**Kernel-family attribution (prefill, profiled):** `hyper_connection_update` + `hyper_connection_update_out` = 8.68 + 8.72 = 17.4 ms → fused `hyper_connection_update_rms` + `hyper_connection_update_out_rms` = 4.08 + 5.55 = 9.6 ms (**−45%**), and the standalone `hyper_connection_rms` passes drop from 100 to 4 dispatches per prefill. The family-level halving lands as only 1–2.6% end-to-end because MoE + GEMMs dominate: `decode_linear_q8` 21%, `verify_gdn_fused` 17%, `moe_expert_gate_up` 16%, `moe_expert_down` 7% of decode GPU.

**PLE fusion (`SPLASH_FUSE_PLE=0` vs. default):** no measurable GPU difference (B1 3.65–3.74 vs. 3.65–3.73; prefill dispatches 1576 vs. 1573). PLE is 1 of 48 layers; the fusion removes 4 dispatches there and nothing more. Keep for tidiness, not speed.

**Wall time: not attributable.** B1 wall is bimodal (86–88 ms vs. 110–123 ms) and tracks MTP command count (9 vs. 13 commands/cycle), which flips run-to-run **within the same binary** — `AdaptiveDraftController` modulates draft depth from a rolling acceptance window, a feedback loop. Sample too small to attribute; no wall-clock claim either way.

**Verdict:** Priority 2's fusion is correct and real but small at step level (−1.2% prefill, −2.6% B1 decode GPU). Priority 1 (live-row MTP narrowing) and the stage-0 background overlap are included in the same diff; the numbers above cover the combined changeset. Largest remaining lever is unchanged: MoE expert GEMM time (33% of decode GPU) and SSD-bound wall (GPU 3.7 ms vs wall 86–123 ms per B1 cycle).

---

### 2026-10-15 — Post-attention 1-row MTP narrowing & NEON host-routing landed; remaining plan items audited

**Completed in this pass (`Qwen4ExpTarget.cpp` + `qwen4exp_layout_test.cpp`):**
1. **Priority 1 part 2 — MTP post-attention narrowing to 1 anchor row (`live - 1`):**
   - Previously, `runMtpDraft` narrowed the pre-attention combiner and `attentionHyperConnection` to `live` rows (`1..4` on step 0), but still ran `addVerifyGate` (`kRows = 8`), `mlpHyperConnection` (`live` rows), `hostSelect` (`live` rows), `stageActiveExperts` (`live` rows), `hostGroup` (`live` rows), `encodeMoEExecute` (`live` rows), and `combiner.mixer` (`live` rows).
   - Now, after `ops::PagedAttention::addVerify` stores KV for all `live` rows and scores the queries, the post-attention path slices `fullPacked`, `fullAttention`, `X`, and `hyperInjection` at `anchorRow = live - 1` and executes `addVerifyGate`, `outputProjectionQ8`, `mlpHyperConnection`, `encodeRouteScores`, `hostSelect`, `stageActiveExperts`, `hostGroup`, `encodeMoEExecute`, and `combiner.mixer` on **strictly 1 row** (`buffers.liveRowsPerLane = 1`).
   - Eliminates MoE routing, SSD expert cache misses/evictions, and expert GEMMs for rows `0 .. live - 2` on MTP step 0 (cutting MTP step-0 routed experts from up to `4 × 10 = 40` down to at most `10`).
2. **Priority 4a — ARM NEON vectorization of `sharedExpertGate` and `topExperts` (`Qwen4ExpTarget.cpp`):**
   - `sharedExpertGate` previously ran a scalar `2560`-element loop per live row on the single host pipeline thread while the GPU waited on `signalPipelineEvent`. Refactored to compute per-64-element quantization group dot products (`wSum * scale + xSum * bias`) using ARM NEON `vfmaq_f32` / `vaddvq_f32`, matching the exact grouping order of `moe_route_select_impl` on the GPU.
   - `topExperts` now uses an 8-wide NEON `vmaxvq_f32` fast-skip over the 512 bf16 router scores once the top-`k` array is full, skipping blocks where no score exceeds `chosen[k - 1]` while preserving exact tie-breaking.
   - Added `neonHostRoutingMatchesScalarReference` in `dev/tests/engine/qwen4exp_layout_test.cpp` verifying bit-exact bf16 parity against the scalar reference across 32 random + tied distributions.

**Status of all plan items:**
- **Priority 1 (MTP live-row + 1-row post-attention narrowing):** ✅ Complete.
- **Priority 2 (Fused `hyper_connection_update_rms` / `hyper_connection_update_out_rms`):** ✅ Complete.
- **Priority 3 (Fused PLE `per_layer_embedding_convolve_add_rms` + stage-0 GCD overlap of `gatherNgramRowsOnCpu`):** ✅ Complete.
- **Priority 4a (Skip `encodePredictRoute` on last streaming layer & NEON `hostSelect`):** ✅ Complete (`L + 2 < geometry.layers` guard already skips the last layer; `topExperts` and `sharedExpertGate` are now NEON-vectorized).
- **Priority 4b (Further `addHyperConnection` dispatch fusion):** ⏸️ Evaluated and not worth pursuing — the remaining `hyper_connection_down` + `hyper_connection_up_mix` dispatches have a full-row reduction barrier between `reduced[320]` and the 2,560-wide stream mix, and the entire HC family is now down to ~5% of step GPU time while SSD expert reads + MoE/Q8 GEMMs dominate (~90%+ of step wall time).

---

### 2026-10-16 — Full 10-prompt `speed_suite.py` + `trace_report.py` & Expert Memory Plan (32 vs 34 vs 36 GiB)

With port 8000 idle, executed the full 10-prompt `speed_suite.py` (2,509 generated tokens across 860 decode steps, greedy) both with `SPLASH_TRACE` (`trace_report.py`) and clean (no trace/timing flags) across `SPLASH_EXPERT_CACHE_GIB={25, 32, 34, 36}` under `dev/benchmarks/guarded.py`. Token sequences across all cache sizes are **100% identical** (`r32 == r34 == r36`).

| Cache Budget (`SPLASH_EXPERT_CACHE_GIB`) | Avg Slots / Layer | Pinned Cache GiB | SSD Reads / Step | MTP Total (ms/step) | MTP Stage (ms/step) | Wait SSD Reads (ms/step) | Host Stage (ms/step) | Traced Step (ms) / tok/s | Clean Step (ms) / tok/s | Lowest Free RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **25 GiB** (cold RAM floor) | 199 | 25.48 GiB | 108 | 7.5 ms | 1.8 ms | 31.0 ms | 31.2 ms | 88.3 ms / 33.0 | — | 10.3 GiB |
| **32 GiB** (`splash-flashnext-server.sh` default) | 255 | 32.54 GiB | 62 | 6.7 ms | 1.3 ms | 20.0 ms | 19.9 ms | 76.8 ms / 37.9 | **78.2 ms / 37.3** | **5.7–6.1 GiB** |
| **34 GiB** (`docs/profiling.md` baseline) | 272 | 34.69 GiB | 52 | 6.5 ms | **1.1 ms** | 17.2 ms | 17.2 ms | 75.8 ms / 38.4 | **75.1 ms / 38.8** | **4.6–5.0 GiB** |
| **36 GiB** (`.agents/plans` step 1) | 288 | 36.71 GiB | **45** | **6.2 ms** | **0.9 ms** | **14.8 ms** | **14.8 ms** | **73.3 ms / 39.8** | **74.8 ms / 39.0** | **3.8–3.9 GiB** |

**Key findings from the trace breakdown:**
1. **MTP draft head is now down to 6.2–6.5 ms/step total (0.9–1.1 ms expert staging):**
   - Previously (`docs/tokenspeed-optimizations.md` §1), MTP took **8–13 ms/step**. With Priority 1 (live-row pre-attention + 1-row post-attention MoE narrowing), MTP total time dropped to **6.2–6.5 ms/step** across 2.57 guesses/step (`~2.0 ms` per draft step), and MTP expert staging dropped to **0.9–1.1 ms/step**.
2. **SSD expert reads dropped from 87/step (historical 34 GiB baseline) to 52/step at 34 GiB and 45/step at 36 GiB:**
   - Eliminating MTP step-0 non-anchor MoE routing (`0 .. live - 2`) prevented MTP from polluting the per-layer expert cache and staging unnecessary experts from SSD.
3. **32 GiB default vs 34/36 GiB overrides:**
   - **32 → 34 GiB** gave a clean **+1.5 tok/s** (`37.3 → 38.8 tok/s`, `−3.1 ms/step`, SSD reads `62 → 52/step`) while keeping **4.6–5.0 GiB** free RAM on the measured host.
   - **34 → 36 GiB** cut SSD reads from `52 → 45/step`, but clean wall-clock gain flattened (`38.8 → 39.0 tok/s`, `+0.5%`) while lowest free RAM dropped to **3.8–3.9 GiB**. Therefore the engine's implicit default is capped at **32 GiB**; explicit `SPLASH_EXPERT_CACHE_GIB` overrides remain available for controlled tests.
   - **Context cap (262k → 100k):** Slipstream's `Q8PageStorage` already allocates KV physical extents lazily on demand (`runtime/engine/MemoryPlan.cpp`), not upfront at startup. Lowering `--max-context` does not free startup RAM for `mlock`ed expert caches; it only caps maximum conversation length.
