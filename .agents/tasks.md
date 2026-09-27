# Tasks — qwen4exp port

## Done

- [x] `Qwen4ExpLayout` — layout constants + arithmetic test
- [x] Package loading and validation (`splash-packed-q4-qwen4exp`)
- [x] Expert tiling at StorageN 128
- [x] MoE routing at 512 experts (`RouterWidth` template)
- [x] Hyper-connection kernels (normalize / mix / update)
- [x] QSA indexer scoring kernel
- [x] QSA block selection (exact top-k, deterministic)
- [x] Sparse attention (`Sparse` flag on the Q8 tile + page mask)
- [x] N-gram embedding gather kernel
- [x] Per-layer embedding kernels (gate + convolve)
- [x] Linear attention — reuses the existing GDN path, no new kernels
- [x] Memory plan: resident vs. streamable split
- [x] Affine quantizer (`dev/tools/quantize.py`)
- [x] Package writer (`dev/tools/package_format.py`)
- [x] 27B converter as a format proof (`dev/tools/convert_qwen38.py`)
- [x] qwen4exp converter: layers, experts, hyper-connections, PLE, draft
- [x] `--release-source` for the disk squeeze
- [x] Verified both layer types against real published weights
- [x] Verified a whole composed layer — exact, 0.000e+00
- [x] `write_head` and `write_embedding` in `convert_qwen4exp.py`
- [x] Placeholder vision generator for package manifest compliance
- [x] Converted all 48 layers to `/Users/nitin/models/qwen38-flash-next-splash` (96.61 GiB)
- [x] Kv2Group12 Metal attention kernels (prefill and verify/decode)
- [x] Draft attention shape acceptance for Qwen4Exp placeholder draft
- [x] Forward path (`Qwen4ExpTarget`) wired in `Runtime.mm`
- [x] Metal command chunking by working set to allow models larger than RAM
- [x] Verified end-to-end forward pass on GPU via `decode-profile` (prefill, B1 decode, B4 decode)

- [x] End-to-end generation quality evaluation against reference ("The capital of France is" -> " Paris. The capital of Germany is Berlin. The capital of Italy is Rome.")
- [x] GDN output gate activation fix (`sigmoid` instead of `silu`)
- [x] Both test suites 100% green (`make test-engine-cpu` and `make test-engine-metal`)
- [x] Vectorized hyper-connection normalize & mix kernels ( slashes GPU time from 504 ms to 190 ms)
- [x] Sub-buffer residency isolation (`detachStreamingLayer`) to prevent IOGPU driver 1.42 GiB file residency traps
- [x] Staged MoE expert cache execution on GPU (10x faster MoE GPU dispatch: 13.3 ms/layer vs 134 ms/layer monolithic)
- [x] Persistent per-layer expert cache with LRU slot tracking & GCD parallel miss staging (>10x decode speedup, ~480 ms/tok steady state)
- [x] Fixed CommandGraph input embedding sequencing invariant (`residentGraph = std::move(graph)`)

## Left / Future Improvements

- [x] Prefill staged expert cache integration with automatic monolithic fallback (19x prefill speedup: 11.1s -> 549 ms; warm cache cuts Token 1 decode staging from 6.1s to 159 ms)
- [x] Steady-state decode accelerated to ~160 ms/tok (6.25 tok/s), end-to-end verified across 20 tokens

## Left / Future Improvements

- [x] Asynchronous pipelined prefetch (decode pipeline + lookahead routing)
- [x] Decode at ≥ 30 tok/s consistently (33–45 measured 2026-09-22)
- [x] Server end to end incl. tool calls with MTP drafting
- [x] Prompt path: no fallback to mapped expert files (row ranges)
- [x] Crash guard: loader refuses a cache that does not fit; guarded.py (2026-09-22)
- [x] Waves default + 64K draft vocabulary: 37.6/37.5 tok/s greedy/sampled (2026-09-22)
- [ ] 50 tok/s: needs ~4.1 tokens a step (tree drafting / better draft head)
- [ ] Uneven cache slots per layer (~9% fewer misses in replay)
- [ ] Draft head: process only live rows (~0.5 ms)
- [x] Quality round: Swift-Qwen3.8-27B-Splash-HQ vs Swift-Qwen3.8-Flash-Next-V3 (145 items across 6 domains: AIME 2025, MATH-500, GPQA Diamond, GSM8K, HumanEval, Hard Systems & Logic; Flash-Next leads 70.3% vs 67.6% with identical ~44 tok/s throughput)
- [ ] Code-prompt drift: 79% same pick vs llama.cpp 89%
- [ ] Short-prompt reading speed (87 vs ~110 tok/s)
- [x] 27B 4-bit package prepared: Swift-Qwen3.8-27B converted to Splash format (local/Swift-Qwen3.8-27B, 16.9 GB)

## Deferred by Nitin

- [ ] DFlash 2 draft training. No longer needed for speed: the model's own MTP
      head drafts. The zero placeholder stays because the package format needs one.

## Slipstream

- [x] Phase 1: other models and the tuner removed
- [x] Phase 2: text only
- [x] Phase 3: DFlash placeholder removed (−1.45 GB)
- [x] Phase 4: `models/qwen4exp/` layout, checker rules, lint clean
- [x] Draft head: survey and plan (docs/draft-head-plan.md)
- [x] Draft head pilot: corpus, recording mode, trainer, scoring - no-go (25 vs 39 tok/s)
- [x] Next track: cheaper check steps (Nitin, 2026-09-23)
- [x] Read-ahead 6 + uneven cache slots: +3-4%, outputs identical
- [x] Smarter read-ahead (multi-row consensus): no gain, left off
- [x] Larger expert cache: declined - memory margin too thin for ~2%
- [x] Benchmark round 2, Slipstream: all 9 suites except sessions (2026-09-23)
- [x] Paired vs llama.cpp V3: mmlu, gsm8k, mmlu_pro, needle - all "same"
- [x] Upstream fixes #31/#92/#120 merged; live omp/pi tool calls OK
- [ ] llama.cpp V3: humaneval (may be partial), ifeval, math500
- [ ] Both engines: sessions, then Nitin judges blind (bench.judge)
- [x] 27B HQ round: completed head-to-head against Swift V3 (45.0% on AIME 2025, 92.0% HumanEval, 44.4 vs 43.9 tok/s)
- [ ] Decide default expert cache (34 vs 30 GiB): measure speed cost first
- [ ] Optional: full-precision reference via a hosted API (Nitin's key)
- [ ] Optional: drop `install/` and Homebrew/release packaging
- [ ] Optional: explicit model interface in place of `QwenTarget`

## GGUF Direct Ingestion (slipstream-gguf)

- [x] Sharded GGUF reader with SIMD dequantization via libggml-base (`dev/tools/sharded_gguf_reader.py`)
- [x] Parallel GGUF-to-Slipstream converter (`models/qwen4exp/tools/convert_qwen4exp_gguf.py`)
- [x] Layer composition and weights alignment:
  - [x] Hyper-connection & MTP RMS norm offsets: subtract 1.0 (Metal kernel adds +1.0f)
  - [x] GDN value heads de-permutation: (3, 16) -> (16, 3) across 7 tensors
  - [x] Router & shexp quantization layout: `quantized_q8` group-major transpose
- [x] CLI launcher `--model` auto-detecting GGUF directories + `--port` option (`install/launcher.py`)
- [x] End-to-end 10-prompt benchmark: 40.1 tok/s decode, 10/10 coherent
- [x] HTTP server test `/v1/chat/completions` on `:8090` verified
- [x] Paired A/B benchmark (Slipstream vs llama.cpp on V3 GGUF): 40.76 vs 23.10 tok/s (1.76x speedup, 100% quality parity)
- [x] Reclaim ~97 GB by deleting redundant `~/models/qwen38-flash-next-splash` (confirmed & completed)
- [x] Spliced Swift V3 GGUF (`~/models/swift-qwen38-flash-next-v3`, 95.52 GiB) via HTTP range donor requests (`dev/tools/build_swift_v3_gguf.py`)
- [x] Fast ingestion of Swift V3 into Slipstream package (`~/models/swift-qwen38-flash-next-v3/prepared`, 372.6s, hardlinked `ngram.bin`)
- [x] Paired A/B benchmark on Swift V3 GGUF (Slipstream vs llama.cpp): 41.72 vs 22.84 tok/s (1.83x speedup, 1.51x faster TTFT, 100% quality parity)
- [x] Set maxTokens cap to 16,384 across Pi and Omp configurations and registered `local/swift-qwen38-flash-next-v3`


