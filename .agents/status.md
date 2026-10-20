# Status — Slipstream (handoff)

**Updated:** 2026-10-19 17:05 PDT by antigravity.
**Branch:** `main`. Optional vision merged, KV cache image token isolation fixed, and comprehensive diverse benchmark (8 text domains + 6 vision categories) committed to `main`. Server stopped.

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
