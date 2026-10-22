# Review: DeepInfer, and what Slipstream can borrow

**Date:** 2026-10-20  
**Subject:** `../DeepInfer` (DeepSeek-AI, Apache-2.0) — can any of it make Slipstream faster or
smaller without changing outputs?

**Worked result (same day, branch `feature/deepinfer-borrows`).**

| Idea | Verdict | Evidence |
|---|---|---|
| R1 per-lane speculation for batched decode | **Kept and enabled**: `slipstream serve` sets `SPLASH_MTP_BATCH=1`; `=0` reverts | After fixing the per-lane draft page-table binding, the guarded controlled B2 A/B (6 alternating server sessions, 9 fresh concurrent pairs/mode, 192 output tokens/request) measured median session-weighted HTTP aggregate throughput 40.64 tok/s off vs 72.46 on (+78.3%); sample median 42.89 vs 81.69 (+90.4%). B2 batches observed in every sample; draft acceptance on 2,824/3,125 (90.4%) when enabled vs 868/12,852 (6.8%) in shadow. That run gave each mode different fresh prompts, so its texts were never comparable. The follow-up parity run holds prompts fixed: 3 greedy prompts in duplicate concurrent B2 pairs give **identical response sha256 with the flag off and on** (580/940 accepted, 94 vs 381 decode steps), so MTP keeps greedy output bit-exact while removing ~4x the steps. Widened to 8 prompts x 256 tokens, all on responses equal off byte for byte, steps 1756 -> 501, HTTP aggregate +43%. At width 4 the gain is smaller (+20.3% over 12 prompts) and the live-contended parity check only holds when batch composition is fixed; composition alone changes greedy text with drafting off, so that is pre-existing batch-shape numerics. Seeded sampled decoding is **not** generally reproducible: the draft RNG is keyed by the request seed and predicted position, which decouples co-tenants' proposals, but identical (prompt, seed) requests still diverge on a warm server because KV page reuse and batch shape perturb logits enough to flip a sampled draw (11/12 identical concurrent pairs and 2/3 identical sequential pairs differed; a fresh-server pair matched 3/3). A sampled-distribution A/B at width 2 gives g=185.4 against a within-mode null of 27.4 (p=0.005, head TV 0.148), but the two paths also differ in step structure, so that shift is not separable from the batch-numerics effect already visible with drafting off; isolating the sampler needs a kernel-level statistical test with fixed logits. A 480 s width-2 soak shows the flag is safe on this workload: +31.7% throughput (13.05 -> 17.19 tok/s), lower per-token tail (p99 96.4 -> 73.9 ms), TTFT unchanged, acceptance 0.76 -> 0.45 as the KV cache fills, steady-state memory growth 2.16 vs 2.48 GiB (cache working set, not a leak), zero failures, no restarts. Data: `build/verification/controlled-mtp-ab-results.json`, `mtp-parity-b2-multiprompt.json`, `mtp-parity-wide-greedy.json`, `mtp-b4-benchmark.json`, `mtp-seeded-determinism-3runs.json`, `mtp-sampling-distribution.json`, `mtp-soak.json`. |
| R2 measured-cost speculation depth | **Reverted** | Offline replay + A/B on the 10-prompt suite were neutral-to-negative; recorded with its measurements in the draft-depth runs below. |
| R3 tail-aware expert-cache objective | **Rejected** | `cache_plan.py` replay over a freshly recorded route log (440 steps, 48 layers, 13056-slot budget): fitting the p95 per layer → mean 138.3 / p95 375 reads a step (uniform: 84.4 / 280.0; mean fit: 79.8 / 265.0). Fitting the p95 of the summed step series (the correct whole-step objective) → 105.2 / 308.0 — still worse in both metrics. The mean objective's smoother slot transfer wins; the tail is set by starved layers, not the hot ones. |
| R4 one-step-ahead scheduling | Untouched | Next step if concurrency work continues. |

The rest of this document is the original review, kept for its reasoning and file
references.

---

## 1. Conclusion first

**One idea is worth real work, and it is not a kernel.** DeepInfer's scheduler carries a
verify-row budget *per sequence* and prices each row against a *measured* cost curve; Slipstream
carries one row budget for the whole batch and disables speculation entirely once two requests
decode together. Porting that policy (R1, R2) raises aggregate throughput for concurrent
sessions and removes the bimodal decode wall, with no change to the verification rule and
therefore no quality risk. Two smaller borrows follow: a tail-aware expert-cache objective (R3)
and one-step-ahead scheduling (R4). Everything else in DeepInfer is either already in
Slipstream, architecture-bound to its checkpoint, or bound to CUDA/Ascend.

## 2. What it is

DeepInfer is DeepSeek's production engine for **DeepSeek-V4.1-Flash** on NVIDIA SM100 and
Ascend 950. Python + PyTorch with a C++ scheduler (`csrc/engine`), FP8/FP4 GEMMs, FlashMLA,
DeepEP multi-rank MoE, EPLB. None of the kernels port: different accelerator, different
numerics, different model. What ports is its **speculation and scheduling policy**, which is
the part it spends the most code on and the part Slipstream's remaining bottleneck is in
(rows cost ~8.9 ms because each one wakes experts from SSD — `docs/draft-head-plan.md`).

## 3. Already in Slipstream (do not rebuild)

| DeepInfer feature | Slipstream equivalent |
|---|---|
| DFlash MTP head + confidence head (`models/mtp/dflash.py:54`) | `runMtpDraft`, `lastConfidence` — head already reports per-guess probability |
| Confidence-scheduled verification (`mtp_varlen.hpp`) | `AdaptiveDraftController`, `SPLASH_MTP_CHAIN_MIN` / `_P_MIN` (`Qwen4ExpTarget.cpp:2558`, `:2966`) |
| Prompt-lookup-ish drafting | `runtime/ops/PromptLookup.*` (hash-indexed n-gram, PLD) |
| Paged KV + prefix reuse + block manager (`block_manager.hpp`) | content-addressed `Cache`/`KvPool` + composite recurrent states (stronger) |
| Chunked prefill, prefill/decode interleave | `Scheduler::nextPrefill/nextDecode`, `SPLASH_PREFILL_TOKEN_BUDGET` |
| Lookahead expert prefetch + staged MoE + wave pipelining | `prefetchPredicted`, `submitPipelineAsync`, `SPLASH_DECODE_WAVES` |
| Per-layer expert placement from routing history (`eplb/policy.py`) | `kSlotProfile` fitted by `bench/cache_plan.py` (`Qwen4Exp.cpp:296`) |
| Structured output, deterministic sampling | server grammar masks, greedy sampler |

## 4. Borrowable, in priority order

### R1. Per-sequence verify rows, and a row budget shared across the batch

**DeepInfer.** `MTPVarlen::schedule` (`csrc/engine/mtp_varlen.hpp:151`) emits a
`seqlens_q[]` **per sequence**. Every sequence gets 1 anchor row plus as many draft rows as its
own confidence product justifies; the *total* is chosen from a measured cost curve, then
distributed by sorting cumulative accept rates (`acc_queue_for_bsz` / `acc_queue_for_seqs_q`).
Kernels consume per-row q lengths; nothing assumes a uniform depth.

**Slipstream.** `QwenTargetVerifyBuffers::liveRowsPerLane` is **one scalar** for the whole batch
(`runtime/model/QwenTarget.hpp:301`), and `Runtime.mm:1151` disables real speculation whenever
`lanes > 1`:

```cpp
buffers.mtpShadow = !(lanes == 1 && mtpProposing(laneEntry(entries, 0)));
```

So 2–4 concurrent requests decode at exactly **1 token per lane per step** while MoE and
attention plans are built for `lanes * 8` physical rows (`MoE::decodePlan` =
`lanes * SPLASH_TARGET_VERIFY_ROWS`, `runtime/ops/MoE.cpp:401`). `retainedRowLimit()` already
computes a per-request limit and `ModelStepResult` already carries per-lane `retained` — only
the *input* shape is global.

**Change.** `std::array<uint32_t, maximumBatchWidth> liveRows` + row offset per lane, then
enable `mtpProposing` for `lanes > 1`. Give each lane `1 + d_i` rows from a shared budget.

**Payoff.** Aggregate throughput for concurrent sessions. Today 4 lanes yield 4 tokens/step at
~6.5 ms GPU. With per-lane depth and roughly 12–16 rows/step, the same step could return ~2×
the tokens: B4 is only 1.7× B1's GPU time precisely because lanes share expert reads, so the
marginal row is cheaper in a batch than in a single stream. Note the win is from *enabling
speculation while batching*, not from shrinking the plan — the B4 number shows dead rows are
nearly free in these weight-bound kernels. Verification is unchanged, so **no quality risk**.

**Cost/risk.** Row indexing is `row % targetVerifyRows` today; per-lane starts need a prefix
sum. Medium-size change in `Runtime.mm` + `Qwen4ExpTarget::addVerify`.

### R2. Choose speculation depth by expected gain per *measured* millisecond

**DeepInfer.** Cost is measured, not modelled:

- `profile_decode_perf` (`deepinfer/worker/model_runner.py:562`) times every captured graph
  size (5 runs, trimmed mean) at warmup and persists the curve (`DecodePerfCache`).
- `ForwardRuntimeEstimator` (`mtp_varlen.hpp:46`) keeps a two-level median of
  `(forward_ms - baseline) / graph_size` over the last 5000 steps, so the marginal cost of a
  row tracks context length and thermal state.
- `MTPVarlen::schedule` maximises `expected_tokens / estimated_ms`, where
  `expected_tokens = 1 + Σ cumprod(confidence)` (`scheduler.hpp:1247` builds the cumprod;
  `llm_engine.py:997` feeds GPU-event timings back every step).

**Slipstream.** The same economics live only in an offline replay
(`bench/depth_controller_replay.py`, `C0/C1/M1` constants) and in fixed env thresholds. The
live controller is bang-bang (rolling accept bits, reject streak) and `docs/tokenspeed-optimizations.md`
records the consequence: wall time is **bimodal, 86–88 vs 110–123 ms**, tracking whether the
step issued 9 or 13 MTP commands.

**Change.** Three small pieces, all hosted in the existing `AdaptiveDraftController`:

1. Persist a measured `rows → ms` table (model + build id), refreshed at startup like
   DeepInfer's curve; it is a handful of synthetic decode steps.
2. Add the online slope, so long-context rows are priced correctly.
3. Replace the threshold ladder with `argmax_d E[kept(d)] / ms(1+d)` using the confidence the
   head already produces, with a 3% hysteresis before switching (avoid the flip-flop).

**Payoff.** Removes the slow mode of the wall-time distribution and prices rows correctly as
context grows. Estimate +2–5% tok/s; one `SPLASH_TRACE` run is enough to predict it offline
before any engine change (no trace file is currently checked in). **No quality risk** — it only
changes how many rows are checked.

### R3. Fit the expert cache to the *tail*, not the mean

**DeepInfer.** `_batch_greedy_indices` (`eplb/policy.py:15`) places experts to minimise
`Σ_steps max_rank(load)`: a min-max objective over history, not total load.

**Slipstream.** `bench/cache_plan.py` scores eviction rules and the slot split by **SSD reads
per step** (mean). The observed step wall is bimodal and `docs/tokenspeed-optimizations.md`
attributes it to expert read waits (15–31 ms of a 75–88 ms step).

**Change.** Re-score the existing `SPLASH_ROUTE_LOG` replays by p95/p99 reads per step (worst
step, then mean), re-fit the per-layer slot split against that objective, and reserve a small
shared "next-step" slot set so one step cannot be forced into 20+ misses. Same total slots →
**no extra RAM**; eviction does not change results → **no quality risk**.

**Payoff.** Lower p95 step time and less SSD stall variance at identical mean hit rate. Cheap:
it is a scoring change in an existing offline tool.

### R4. One-step-ahead scheduling (ZOS), last

**DeepInfer.** Schedules and prepares step N+1 while N is in flight. Fresh rows only need a
compact `remap` (`csrc/engine/zos.cpp:52`); matched rows come from the previous step's GPU
outputs. `schedule_cnt` / `process_cnt` keep accept-rate rounds straight when a sequence is
scheduled ahead (`mtp_varlen.hpp:170`).

**Slipstream.** `Engine::apply` runs only after the model ticket completes, so next-step
encoding never overlaps the previous step. Host staging is 17–20 ms of a 75–78 ms step.

**Change.** Double-buffer the decode arena; pre-encode the resident graph once `NextAnchor` is
known, overlapping the current step's staging/pipeline events.

**Payoff.** Possibly 3–8 ms/step (5–10%). Highest complexity and the only change that touches
state lifetimes — do it only after R1/R2 land and are measured.

## 5. Not worth porting

| Idea | Why not |
|---|---|
| YOCO / SWA bounded prefill replay (`yoco.py`) | Correctness relies on every layer being windowed. Slipstream has 16 full-attention layers, so truncating the replay prefix changes outputs. |
| EPLB redundant expert replicas | Needs multiple ranks to replicate across. One Mac, one copy on disk. |
| Cross-layer share-KV / share-index / candidate-block sharing (`cross_layer.py`) | Architecture-defined by the checkpoint (`layer_info_shared_cfg`), not a free optimisation. |
| CUDA-graph size ladder (`build_graph_batch_sizes`) | Slipstream already has explicit M8/M16/M24/M32 Metal plans. |
| Batch-invariant kernels | A quality-tooling feature that costs throughput; Slipstream's protocol already compares greedy tokens. |
| FlashMLA / DeepGEMM / DeepEP / DeepSelect kernels | CUDA/Ascend, FP8/FP4. No Metal path. |
| Block manager free list, chunked prefill, structured output, deterministic sampling | Already present, at least as capable. |

## 6. Suggested order

1. **R2 offline**: replay `SPLASH_TRACE` through the expected-gain rule with a measured cost
   table. No engine change; go/no-go before touching code.
2. **R1**: per-lane rows + shared budget — the concurrency ceiling.
3. **R3**: tail-aware slot split — cheapest real change.
4. **R4**: only if steps 1–3 leave host staging as the dominant term.

Every step keeps the existing bar: identical greedy tokens on the 10-prompt suite, `speed_suite.py`
under `guarded.py`, and `trace_report.py` for attribution.

## 7. Attribution

DeepInfer is Apache-2.0, same as Slipstream. If any of the scheduling code is copied rather
than re-derived (e.g. the median trackers in `mtp_varlen.hpp`), carry its notice.
