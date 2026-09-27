# Head-to-Head Benchmark: Swift-Qwen3.8-27B-Splash-HQ vs Swift-Qwen3.8-Flash-Next-V3

**Date:** 2026-09-26 11:26 PDT

**Platform:** Apple Silicon (Unified Memory), Single-Engine Sequential Execution

**Total Evaluated Items:** 145 items across 6 rigorous domains


---

## 1. Executive Summary & Scorecard

| Domain / Benchmark | Items | Swift-Flash-Next-V3 Accuracy | Swift-27B-Splash-HQ Accuracy | Accuracy Delta | Flash-Next Speed | 27B-Splash Speed | Speed Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| **AIME 2025** | 20 | 45.0% (9/20) | 45.0% (9/20) | 0.0% | 44.3 tok/s | 42.8 tok/s | 0.97x |
| **MATH-500** | 35 | 62.9% (22/35) | 60.0% (21/35) | -2.9% | 44.8 tok/s | 46.6 tok/s | 1.04x |
| **GPQA Diamond** | 35 | 54.3% (19/35) | 45.7% (16/35) | -8.6% | 44.8 tok/s | 38.5 tok/s | 0.86x |
| **GSM8K** | 25 | 96.0% (24/25) | 96.0% (24/25) | 0.0% | 45.6 tok/s | 48.1 tok/s | 1.06x |
| **HumanEval** | 25 | 92.0% (23/25) | 92.0% (23/25) | 0.0% | 40.6 tok/s | 48.8 tok/s | 1.20x |
| **Hard Systems & Logic** | 5 | 100.0% (5/5) | 100.0% (5/5) | 0.0% | 39.2 tok/s | 35.6 tok/s | 0.91x |
| **TOTAL / OVERALL** | **145** | **70.3% (102/145)** | **67.6% (98/145)** | **-2.8%** | **43.9 tok/s** | **44.4 tok/s** | **1.01x** |

---

## 2. Latency & Efficiency Metrics

| Metric | Swift-Flash-Next-V3 | Swift-27B-Splash-HQ | Comparison |
|---|---:|---:|---|
| **Average Decode Throughput** | **43.9 tok/s** | **44.4 tok/s** | **1.01x speed** |
| **Average Time-To-First-Token (TTFT)** | **1118.9 ms** | **659.4 ms** | **1.70x faster** |
| **Total Generated Tokens** | 128,925 tokens | 127,396 tokens | -1.2% tokens |