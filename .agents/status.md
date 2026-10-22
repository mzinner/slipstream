# Status — Slipstream (handoff)

**Updated:** 2026-10-22 by Buffy (Freebuff).
**Branch:** `feature/deepinfer-borrows` (off `main`, see DeepInfer review below). Server on 8090 stopped.

## DeepInfer borrow work (2026-10-20)

Three borrows implemented and measured on this branch; full write-up in
`docs/research/deepinfer-review.md` (results table at the top):

- **R1 — speculative drafting for batched decode, `SPLASH_MTP_BATCH=1` (on when served via
  `slipstream serve`, off for a bare `server/server.py`).** Per-lane
  liveRows + per-lane draft chains on the existing lanes\*8 rows. Follow-up review found and
  fixed a page-table binding error in the draft attention pass: it computed lane-specific Q8
  parameters but bound the outer batch's lane-0 page table for every one-lane draft. The draft
  now binds its actual lane's table at kernel batch slot 0. `make -j8` and the full native
  `make -j8 test-engine-cpu test-engine-metal` pass after the fix (re-verified 2026-10-22). A controlled two-lane real-model HTTP A/B shows a throughput gain on the measured workload
  (results below). Greedy text is bit-identical between `SPLASH_MTP_BATCH=0` and `=1` when the
  same prompts are used; the earlier "not token-identical" reading was an artifact of that A/B
  giving each mode different fresh prompts. Sampled decoding is distribution-equivalent by
  construction but **not seed-reproducible** while batched drafting is on (see the parity runs
  below). `install/launcher.py` sets `SPLASH_MTP_BATCH=1` for `slipstream serve` as of
  2026-10-22 (explicit env still wins, `=0` reverts). The draft head now draws from a stream keyed
  by the request's seed and predicted position (`QwenMtpLane::seed` + counter-based draw in
  `runMtpDraft`) instead of a process-global `std::mt19937`; that removes the coupling between
  co-tenants' proposals, but it does **not** make sampled output reproducible — see the
  reproducibility correction below.
- **R2 — measured-cost draft depth.** Neutral-to-negative on the 10-prompt suite; reverted, cost
  curve kept in `cache_plan.py`.
- **R3 — tail-aware expert-cache slot objective.** Refuted on a freshly recorded route log:
  whole-step p95 fit (105.2 mean / 308 p95 reads a step) loses to the mean fit (79.8 / 265.0) and
  to uniform (84.4 / 280.0). Mean objective stays; the tool now reports both.

**2026-10-22 controlled B2 throughput A/B:** ran six guarded server sessions in alternating
`off/on/on/off/off/on` order, each with a 28 GiB expert cache, batch width/concurrency 2, and a
5 GiB memory floor. Each session ran three fresh, unique-tagged pairs of simultaneous prompts
(192 output tokens per request); both modes had zero prefix-cache hits. All 9 samples per mode
recorded B2 decode batches. Median session-weighted aggregate HTTP throughput was **40.64 tok/s
off vs 72.46 tok/s on (+78.3%)**; median individual-pair throughput was **42.89 vs 81.69 tok/s
(+90.4%)**. On-mode draft acceptance was **2,824 / 3,125 (90.4%)**, versus **868 / 12,852
(6.8%)** for shadow probes off. Every response hit the 192-token cap. Important caveat: while
samples within each mode reproduced exact text, the generated sequences differed between modes
for both synthetic prompts. That difference is **explained and was not an MTP defect**: the run
used `unique_prompt_tag_per_pair`, so the two modes generated from different prompts. Superseded
by the parity runs below, which hold the prompts fixed. Full data:
`build/verification/controlled-mtp-ab-results.json`; logs:
`build/verification/controlled-*-session-*.log`. Servers stopped; port 8090 closed.

**2026-10-22 greedy parity under fixed prompts (the safety answer).** Three prompts (integer
list, merge-intervals code, hash-table explanation), 128 output tokens, `temperature=0`, sent as
one concurrent pair into a width-2 batch. Each mode got its own guarded native server
(`SPLASH_MAX_BATCH_WIDTH=SPLASH_MAX_CONCURRENCY=2`, 18 GiB expert cache, 3.5 GiB floor,
variable unset so it is not shadow). Same build `src-b7ee9ea5`, same target model hash.
- Every B2 decode batch was width 2; zero failed requests; Metal healthy; memory pressure normal.
- **`b2-off` vs `b2-on` response sha256 are identical for all 3 prompts, and each pair is
  internally identical**: MTP batch acceptance adds no token-level change to greedy output.
- Acceptance/step: 580 accepted of 940 drafted (61.7%) and 94 decode batches on, versus 3810
drafted with 0 accepted and 381 batches off — a **4.1x reduction in decode steps**, native decode
768 tokens in 363 ms versus 1137 ms.
- Single-prompt repeat of the same design (128 tokens) reproduced this exactly: pair-identical
  text in both modes, identical across modes, 162/460 accepted, 46 versus 127 B2 steps.
- Full data: `build/verification/mtp-parity-b2-multiprompt.json`,
  `build/verification/mtp-parity-b2-singleprompt.json`.

**Caveats this does not cover.** (1) Solo (B1) versus B2 greedy is *not* bit-identical for every
prompt: an earlier run had plain B2 (drafting off) differ from solo on one of three prompts with
identical token counts — the pre-existing batched-kernel numerics, independent of MTP. MTP must
be compared against B2, not against B1. (2) The draft head draws from a stream keyed by the
request's seed and the predicted position (`drawDraftToken` in `runMtpDraft`, fed by
`QwenMtpLane::seed`), so which requests share the batch no longer changes a lane's proposals.
On three fresh servers two identical concurrent seeded requests (`temperature=0.7, top_p=0.9,
seed=847362`) returned the same text every time (`1ab6afce…`, 156/500 accepted, 50 steps each). (3) What still varies with `seed` is batch *composition*, and that is pre-existing:
with drafting off in both runs, the same prompt at width 2 produced different greedy text when
paired with prompt A versus prompt B (`88416109…` vs `e5024573…`), and at width 4 changing only
one lane's output budget (so the batch drops b4→b3) changed all three other prompts' text
(`build/verification/mtp-b4-composition-control.json`). Numerics track batch shape; MTP does not
add a new dependence, it changes which shapes occur because it changes finish times.

**2026-10-22 reproducibility correction (seeded sampled decoding is NOT reproducible).** The
fresh-server result above was a best case, not the general property. On one warm server at width
2 (`SPLASH_MTP_BATCH=0`, `seed=500..502`, 32 tokens): two identical *sequential* requests returned
different text in 2 of 3 trials, and two identical *concurrent* requests differed in 11 of 12
trials (warm cache, cold cache, and 250 ms-staggered admission all failed). The draft-draw fix
only decouples proposals; the sampled token itself still follows the engine's numeric context —
KV page reuse and batch shape perturb logits enough to flip a draw at temperature 0.7. So the
honest statement is: same seed + same prompt reproduces only when the numeric context is also
reproduced (a fresh server, or an untouched lane). Do not promise seed-reproducibility across
retries on a live server. Probe: identical pair/sequential matrix, 3 trials per condition.

**2026-10-22 wider greedy parity (8 prompts, 256 tokens, B2).** Same duplicate-pair design over
math, series, Python, Rust, SQL, systems, prose and JSON prompts: all 8 off pairs and all 8 on
pairs are internally identical and every on response matches off byte for byte. Steps 1756 →
501, draft acceptance 2516/5010 (50.2%), HTTP aggregate 27.63 → 39.60 tok/s (+43%), zero failed
requests. `build/verification/mtp-parity-wide-greedy.json`.

This is now a checked-in harness, `dev/benchmarks/mtp_parity.py`: it runs one server per mode
under `guarded.py`, sends each prompt `--width` times into one batch (a fixed-composition wave),
and requires byte parity for greedy runs or per-mode seeded repeatability for sampled ones:

```zsh
PYTHONPATH=. .venv/bin/python dev/benchmarks/guarded.py --floor-gib 3.5 -- \
  .venv/bin/python dev/benchmarks/mtp_parity.py --model local/prepared \
  --package ~/models/swift-qwen38-flash-next-v3/prepared --binary build/slipstream --width 2
```

Passing runs, all `PASS` with zero failed requests and healthy Metal: width 2 greedy, 8 prompts ×
128 tokens (`mtp-parity-harness-width2.json`); width 4 greedy, 4 prompts × 128 tokens, steps
508 → 127, 1536/2540 accepted (`mtp-parity-harness-width4.json`); width 2 sampled
(`--temperature 0.7 --top-p 0.9 --seed 847362`), 4 prompts, every wave repeatable in both modes,
steps 508 → 124 (`mtp-parity-harness-sampled.json`).

**2026-10-22 B4 benchmark (12 prompts x 192 tokens, three width-4 rounds).** Aggregate HTTP
throughput 17.21 → 20.71 tok/s (+20.3%), native decode 950 → 1363 tok/s, acceptance 1578/3246
(48.6%). The B4 gain is much smaller than B2's because batch-4 steps already amortise the expert
reads MTP is hiding. Token parity at width 4 is only meaningful when the batch composition is
fixed: with 4 identical prompts in lockstep the off and on texts are identical
(`25c90ca0…`) and wall time falls 15.76 → 11.92 s (native decode 1245 → 3192 tok/s). With mixed
prompts, 6 of 12 differ — but the composition control above shows drafting-off differs too once
composition changes, so that is batch-shape sensitivity, not an MTP defect.
`build/verification/mtp-b4-benchmark.json`, `mtp-b4-constant-composition.json`.

**2026-10-22 sampled-distribution check (`dev/benchmarks/mtp_sampling_distribution.py`).** New
harness: 64 seeded samples per prompt at `temperature=0.7, top_p=0.95`, each sample a wave of
`--width` identical requests (the flag only gates *per-lane* drafting, so width 1 runs the same
single-lane path in both modes and compares nothing), re-tokenized through `/tokenize`, compared
with a G-test over the 24-token distribution head against a within-mode split-half null. The
statistic was calibrated offline first: a total-variation version had a noise floor near 0.7 and
missed a 3x head-token bias, which is why the head G-test replaced it (rejects a ~5% absolute
head redistribution, does not reject a faithful sampler).
- Width 1 (control): both modes drafted 6,615/6,613 tokens, g=12.5 vs null 25.9, p=0.63 — the
two servers executed the same code, exactly as expected. This is why the harness now rejects
`--width 1`.
- Width 2 (real comparison): off drafted 38,654, on drafted 13,034 (per-lane chains propose less
per emitted token), g=185.4 against null mean 27.4 / p95 76.7, **p=0.005**, head TV 0.148, head
mass 0.586 vs 0.561. So the sampled distribution does shift measurably between the two paths.
- That shift is **not** attributable to the acceptance/residual math: the two modes also differ in
step structure, and with per-lane drafting the lanes desynchronize, so the numeric context differs
in the same way that changes greedy text with composition. Identical seeded lanes disagreed in
96/96 width-2 waves (and 54/64 at 24 tokens), in both modes. A distribution claim needs the
sampler isolated from engine numerics — a kernel-level statistical test over `accept_sampled_lane`
with fixed logits, not an HTTP-level A/B.
Data: `build/verification/mtp-sampling-distribution.json`.

**2026-10-22 soak (`dev/benchmarks/mtp_soak.py`, width 2, 480 s per mode, 18 GiB cache).** Both
modes ran the same rotating-prompt schedule with streaming responses. `SPLASH_MTP_BATCH=1` is
faster *and* better-tailed on this workload:
- Throughput: 13.05 → 17.19 tok/s (+31.7%) over 70 vs 90 requests / 11,973 vs 16,600 tokens.
- Per-request per-token latency, p50 68.4 → 51.9 ms, p95 88.3 → 70.5 ms, **p99 96.4 → 73.9 ms**,
  max 97.0 → 75.9 ms. TTFT unchanged (p50 1183 → 1133 ms; p99 1750 → 1941 ms, within n=90 noise).
- Tail does not drift over the window: p99 73.6 → 72.8 ms late-vs-early (off: 87.3 → 92.7).
- Draft acceptance falls from 0.76 in the first window to 0.45 in the last (min 0.20) as the KV
  cache fills; drafting-off accepts 0.00 throughout (the batched single-chain path proposes but
  never accepts), which is the whole reason R1 exists.
- Memory: steady-state growth 2.16 GiB off vs 2.48 GiB on — the KV working set from unique prompt
  tags, not a mode-specific leak. No denied reservations, memory pressure `normal` throughout, no
  failed requests, no restarts, Metal healthy.

Two measurement pitfalls worth keeping: SSE events carry however many tokens a step produced, so
client chunk gaps are *not* inter-token latency (an early run reported a nonsense p50 of 0.05 ms);
the harness now uses per-request per-token cost. And a rotating-prompt soak grows the cache in both
modes, so the memory gate compares steady-state growth between modes rather than against a fixed
absolute bound.

**2026-10-22 serving default switched on.** `install/launcher.py` now sets
`SPLASH_MTP_BATCH=1` for `slipstream serve` (env still wins; `SPLASH_MTP_BATCH=0` reverts), and
`docs/swift-v3-usage.md` documents it. Verified against the live launcher, not just the harness:
`ps eww` on the spawned server shows `SPLASH_MTP_BATCH=1` when unset and `=0` when overridden.
Four concurrent greedy requests x 300 tokens (b4-dominated: 72 b4 steps):
- on: 52.83 tok/s aggregate, 22.72 s, draft acceptance 850/1740 (48.9%)
- off (`SPLASH_MTP_BATCH=0`): 40.85 tok/s, 26.93 s, acceptance 9/5427 (0.2%)
- => +29.3% aggregate throughput from the launcher default, zero failed requests, Metal healthy.
Note per-lane drafting only has an effect when several requests decode in the same step, so a
single-client session sees no change (it takes the single-lane drafting path either way).
Logs: `build/verification/serve-default.log`, `serve-off-ab.log`.

Older status (vision merge) below.

**Previously updated:** 2026-10-19 17:05 PDT by antigravity.
**Branch (before this branch):** `main`. Optional vision merged, KV cache image token isolation fixed, and comprehensive diverse benchmark (8 text domains + 6 vision categories) committed to `main`. Server stopped.

## In one line

Optional vision subsystem fully merged into `main` (`603ba300`): real 27-layer vision weights extracted from Swift-Qwen3.8-Flash-Next into `prepared/vision/model.bin` (865 MB), dynamic zero-overhead lazy loading in `Runtime.mm`, Metal kernels compiled, Protocol v5 synchronized, KV prefix cache image token collisions fixed in `Engine.cpp`, all test suites pass 100% green, and diverse benchmarks show 0% text regression (35.9 tok/s text avg vs 39.9 tok/s multimodal avg) with bit-accurate visual perception across all 6 image types. Server on port 8090 stopped.

---

## How to use it

```zsh
# Standard: serve the main Swift model (one server at a time):
./slipstream serve --model ~/models/swift-qwen38-flash-next-v3/prepared --port 8090

# Served id is local/prepared — connect with:
pi --model slipstream/local/prepared --thinking low
omp --model splash-flashnext/local/prepared --thinking=low --tools=read,write,edit,bash,grep,glob,todo --approval-mode=yolo

# Run diverse benchmark (8 text prompts + 6 multimodal vision tasks):
.venv/bin/python dev/benchmarks/run_diverse_benchmarks.py --port 8090

# Run side-by-side text vs vision comparison:
.venv/bin/python dev/benchmarks/benchmark_vision_vs_text.py --port 8090

# Run guarded A/B benchmark comparing Slipstream vs llama.cpp:
python3 dev/tools/compare_slipstream_vs_llamacpp.py --model-dir ~/models/swift-qwen38-flash-next-v3
```

---

## Diverse Benchmark Suite: Text Reasoning vs. Multimodal Vision

Executed via `dev/benchmarks/run_diverse_benchmarks.py` on Apple M5 Pro (64 GB) under memory guard (`SPLASH_EXPERT_CACHE_GIB=28`):

### A. Diverse Text-Only Performance (8 Domains)

| Domain / Prompt ID | Category | TTFT | Decode | Output Tokens | Wall Time |
|---|---|---:|---:|---:|---:|
| `math_arithmetic` | Math Reasoning (Arithmetic) | 2,772.4 ms | **39.7 tok/s** | 193 | 7.61 s |
| `math_derivation` | Symbolic Derivation ($p - q$) | 2,003.4 ms | **39.5 tok/s** | 493 | 14.46 s |
| `logic_puzzle` | Constraint Satisfaction Logic | 1,100.6 ms | **41.2 tok/s** | 447 | 11.94 s |
| `rust_ring_buffer` | Systems Programming (Rust SPSC) | 973.7 ms | **30.3 tok/s** | 512 | 17.86 s |
| `python_algorithms` | Algorithms (Dijkstra / Bellman-Ford) | 833.4 ms | **35.2 tok/s** | 448 | 13.54 s |
| `sql_cohorts` | Data Engineering (Retention Cohorts) | 802.6 ms | **34.3 tok/s** | 448 | 13.85 s |
| `distributed_systems` | Distributed Systems (Raft vs Paxos) | 739.9 ms | **30.8 tok/s** | 447 | 15.20 s |
| `cpp_code_review` | Concurrency & Security Code Review | 1,376.6 ms | **36.2 tok/s** | 512 | 15.51 s |
| **TEXT OVERALL AVG** | **Across all 8 diverse domains** | **1,325.3 ms** | **35.9 tok/s** | **3,500** | **109.96 s** |

### B. Diverse Multimodal (Text + Vision) Performance (6 Image Types)

| Image Task ID | Modality / Category | TTFT | Decode | Output Tokens | Wall Time | Visual Grounding |
|---|---|---:|---:|---:|---:|---|
| `financial_barchart` | Financial Multi-Bar Chart ($k) | 1,813.5 ms | **40.4 tok/s** | 448 | 12.88 s | Verified exact bar extractions |
| `architecture_flowchart`| Microservices Architecture Diagram | 1,331.9 ms | **42.9 tok/s** | 448 | 11.76 s | Correct component path tracing |
| `market_pie_chart` | Cloud Market Share Pie Chart | 1,188.1 ms | **37.4 tok/s** | 384 | 11.44 s | Correct slice & share extraction |
| `geometric_spatial_scatter`| 4-Quadrant Geometric Shapes & Colors | 1,116.6 ms | **43.3 tok/s** | 384 | 9.97 s | Correct quadrant counting |
| `invoice_table_ocr` | Rendered Billing Invoice Table OCR | 1,052.5 ms | **37.7 tok/s** | 384 | 11.22 s | Exact SKU & arithmetic check |
| `scientific_curve_plot`| Damped Oscillator Curve Plot | 1,764.3 ms | **38.0 tok/s** | 384 | 11.84 s | Correct formula & limits |
| **VISION OVERALL AVG** | **Across all 6 diverse image tasks** | **1,377.8 ms** | **39.9 tok/s** | **2,432** | **69.11 s** | **100% accurate grounding** |

---

## Head-to-Head Benchmarks: llama.cpp Fork vs. Slipstream (Flash-Next V3)

Across 6 standard reasoning and coding tasks (temperature 0.0, M5 Pro 64 GB):

| Task | Category | llama.cpp Fork | Slipstream | Speedup | llama.cpp TTFT | Slipstream TTFT |
|---|---|---:|---:|---:|---:|---:|
| `gsm8k_math` | Math Reasoning | 24.0 tok/s | **43.6 tok/s** | **1.82x** | 4,024 ms | **2,337 ms** |
| `math500_series` | Math Derivation ($p - q$) | 24.3 tok/s | **43.1 tok/s** | **1.77x** | 1,655 ms | **1,587 ms** |
| `constraint_logic`| Constraint Logic | 25.4 tok/s | **46.0 tok/s** | **1.81x** | 1,469 ms | **1,042 ms** |
| `python_intervals`| Python Coding | 19.7 tok/s | **35.0 tok/s** | **1.77x** | 1,507 ms | **1,070 ms** |
| `rust_csv` | Systems Coding | 22.7 tok/s | **37.5 tok/s** | **1.65x** | 1,257 ms | **859 ms** |
| `tech_explanation`| Technical Writing | 22.5 tok/s | **39.4 tok/s** | **1.75x** | 1,267 ms | **843 ms** |
| **OVERALL AVG** | Across all 6 tasks | **23.1 tok/s** | **40.8 tok/s** | **1.76x** | **1,863 ms** | **1,290 ms** |

---

## Invariants & Rules

- **One model server at a time.** Two pin more memory than the Mac has and freeze it. Run engine experiments through `dev/benchmarks/guarded.py -- <cmd>`; check `lsof -i :8090` before starting.
- **Do not launch daemon without user request.** Server on port 8090 must remain stopped until explicitly launched.
- **After any engine change:** `make`, `make test-engine-cpu`, `PYTHONPATH=. .venv/bin/python dev/tests/engine/test_protocol_python.py`.
- **Zero-cost text invariant:** No vision weights allocated in unified memory for text requests. Vision loads dynamically only upon receiving an image input.
- **KV cache isolation for vision:** `Cache::lookup` and `publishCommittedBlocks` must never extend past `images.front().offset` to prevent cross-image prefix collisions.
