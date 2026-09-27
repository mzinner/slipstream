# Status — Slipstream-GGUF (handoff)

**Updated:** 2026-09-26 20:00 PDT by antigravity.
**Branch:** `main` tracking `origin/main` (`github.com/npanj/slipstream.git`). Never push to `github.com/incoai/splash`.

## In one line

`slipstream-gguf` natively ingests and serves both Nitin's daily V3 model (`~/models/qwen38-flash-next-v3`) and the new Swift V3 model (`~/models/swift-qwen38-flash-next-v3`), beating llama.cpp by **1.83x (41.72 vs 22.84 tok/s)**, and in comprehensive head-to-head testing against `Swift-Qwen3.8-27B-Splash-HQ` (145 items across 6 domains), Swift V3 achieved **70.3% vs 67.6%** accuracy with identical decode throughput (**43.9 vs 44.4 tok/s**).

---

## How to use it

```zsh
# Serve Swift V3 directly pointing at GGUF directory:
./splash serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090

# Or serve Nitin's daily V3 model:
./splash serve --model ~/models/qwen38-flash-next-v3 --port 8090

# Run guarded A/B benchmark comparing Slipstream vs llama.cpp:
python3 dev/tools/compare_slipstream_vs_llamacpp.py --model-dir ~/models/swift-qwen38-flash-next-v3

# Run head-to-head benchmark comparing Swift-27B-HQ vs Swift-V3:
dev/benchmarks/run_full_comparison_orchestration.sh
```

---

## Head-to-Head Benchmarks: Swift-27B-Splash-HQ vs Swift-Flash-Next-V3

Completed 2026-09-26 across 145 paired items (seed 1234, temperature 0.0), run sequentially under memory guard:

| Domain / Benchmark | Items | Swift-Flash-Next-V3 | Swift-27B-Splash-HQ | Accuracy Delta | Flash-Next Decode | 27B-Splash Decode | Speed Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| **AIME 2025** | 20 | **45.0% (9/20)** | **45.0% (9/20)** | 0.0% | 44.3 tok/s | 42.8 tok/s | 0.97x |
| **MATH-500 (L4-5)** | 35 | **62.9% (22/35)** | **60.0% (21/35)** | -2.9% | 44.8 tok/s | 46.6 tok/s | 1.04x |
| **GPQA Diamond** | 35 | **54.3% (19/35)** | **45.7% (16/35)** | -8.6% | 44.8 tok/s | 38.5 tok/s | 0.86x |
| **GSM8K** | 25 | **96.0% (24/25)** | **96.0% (24/25)** | 0.0% | 45.6 tok/s | 48.1 tok/s | 1.06x |
| **HumanEval** | 25 | **92.0% (23/25)** | **92.0% (23/25)** | 0.0% | 40.6 tok/s | 48.8 tok/s | 1.20x |
| **Hard Systems & Logic**| 5 | **100.0% (5/5)** | **100.0% (5/5)** | 0.0% | 39.2 tok/s | 35.6 tok/s | 0.91x |
| **TOTAL / OVERALL** | **145** | **70.3% (102/145)** | **67.6% (98/145)** | **-2.8%** | **43.9 tok/s** | **44.4 tok/s** | **1.01x** |

### Key Findings
1. **Mathematical Reasoning Parity**: On AIME 2025, both models solved the exact same 9 out of 20 problems (100% agreement on problem solvability). On GSM8K, both achieved 96.0% accuracy (24/25), missing the exact same single question.
2. **Science Reasoning Advantage**: On GPQA Diamond (PhD-level multi-disciplinary science), Swift-Flash-Next-V3 led 54.3% to 45.7% (+8.6%), showing stronger scientific concept retrieval.
3. **Coding & Systems Parity**: Both models achieved 92.0% on HumanEval (23/25 unit tests passing) and 100% on complex systems reasoning (Acquire-Release vs SeqCst store buffering, memory bandwidth bottleneck explanations, interval merging, and zero-copy Rust CSV parsers).
4. **Throughput Equivalence**: Both models average ~44 tok/s decode (Flash-Next: 43.9 tok/s, 27B-Splash: 44.4 tok/s).
5. **TTFT Difference**: 27B-Splash-HQ achieves 1.70x faster TTFT (659 ms vs 1,119 ms) due to dense Q8 weights vs Flash-Next's 48-layer hyper-connection prefill and n-gram table gather.

---

## Rules that are not optional

- **One model server at a time.** Two pin more memory than the Mac has and freeze it. Run engine experiments through `dev/benchmarks/guarded.py -- <cmd>`; check `pgrep -fl generate-sample` and `lsof -i :8090` before starting.
- **Nothing memory-heavy next to a running engine.**
- **Do not touch:** `~/models/qwen38-flash-next-bf16` (338 GB source), `~/models/qwen38-flash-next-v3` (Nitin's daily llama.cpp source shards).
- **After any engine change:** `make`, `make build/engine-tests/generate-sample`, verify with the 10-prompt check.

---

## Done in this Session

| What | Where |
|---|---|
| Reclaimed 97 GB by deleting redundant old package | `/Users/nitin/models/qwen38-flash-next-splash` deleted |
| Spliced Swift V3 GGUF (3 shards, 95.52 GiB) via HTTP range donor requests | `dev/tools/build_swift_v3_gguf.py` -> `~/models/swift-qwen38-flash-next-v3` |
| Ingested Swift V3 into Slipstream package with hardlinked `ngram.bin` (0 bytes extra) | `models/qwen4exp/tools/convert_qwen4exp_gguf.py` -> `~/models/swift-qwen38-flash-next-v3/prepared` |
| Generalized paired A/B benchmark harness for multi-model evaluation | `dev/tools/compare_slipstream_vs_llamacpp.py` |
| Completed end-to-end Swift V3 benchmark: 41.72 vs 22.84 tok/s (1.83x) | `dev/benchmarks/comparison_swift-qwen38-flash-next-v3_results.json` |
| Capped maxTokens to 16,384 and registered Swift V3 across Pi & Omp | `~/.pi/agent/models.json`, `~/.omp/agent/models.yml`, `~/.pi/agent/settings.json` |
| Executed comprehensive 145-item benchmark between Swift-27B-HQ and Swift-V3 | `dev/benchmarks/swift_benchmark_results/BENCHMARK_SCORECARD.md` |
| Benchmarked & optimized speculative drafting on Swift V3 (20 items; 68.6% tokens speculative, precomputed rotary frequencies) | `dev/benchmarks/benchmark_drafting.py`, `models/qwen4exp/Qwen4ExpTarget.cpp` |

---

## Pending / Active Track (Option B: Tree Drafting)
1. Implement 2D tree attention masking in Metal shaders (`runtime/metal/kernels/decode/attention_q8.metal`) for tree-based speculative drafting.
2. Update MTP draft proposer to emit branching candidate trees.
3. Update verification and tree acceptance path in `Runtime.mm`.
4. Validate throughput gain (target: 50–60+ tok/s).
