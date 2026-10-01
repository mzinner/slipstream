I've been working on getting the 95.5 GiB Qwen3.8-Flash-Next model to run fast on a single 64GB Mac. In my earlier posts, I shared a custom expert-streaming fork of llama.cpp and native 8-bit benchmarks on Splash. It worked, but decode capped out around ~23–27 tok/s and slowed down as context grew.

Today I'm releasing **Slipstream**: a compiled C++ Metal inference engine with native SSD expert streaming and speculative drafting for Apple Silicon.

**The main result:**
If you already downloaded my original V3 model (34k+ downloads), you don't need to re-download anything. You can run that exact checkpoint on Slipstream for a **1.76x speedup: 41–52 tok/s** (up from 23.1 tok/s in llama.cpp) on the same 64GB Mac.

Even better: decode speed doesn't collapse at long context. Across 3,086 live requests in real coding sessions, it stays flat at **33–44 tok/s all the way out to 130,000 tokens**.

Previous posts for context:
- [Qwen3.8-Flash-Next in llama.cpp on r/Qwen_AI](https://www.reddit.com/r/Qwen_AI/comments/1wkd7pm/qwen38flashnext_955_gib_on_a_64gb_mac_at_27_toks/)
- [Splash Engine native 8-bit on r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1wmbbf9/splash_engine_qwen3827b_in_native_8bit_at_3755/)

Open source resources:
- [Slipstream Engine on GitHub](https://github.com/npanj/slipstream)
- [Original V3 Model on Hugging Face (34k+ downloads)](https://huggingface.co/nitinpanj/qwen38-flash-next-v3)
- [Optional Swift KV-Sparse Model on Hugging Face](https://huggingface.co/nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF)

---

## 1. How to run your existing V3 model on Slipstream

If you have the model from the last post (`~/models/qwen38-flash-next-v3`), you can point Slipstream directly at it.

### Step 1: Clone & build (under 1 minute)
```zsh
git clone https://github.com/npanj/slipstream.git
cd slipstream
make -j4
```

### Step 2: Download the model (if you don't already have it)
```zsh
# Downloads the 3 GGUF shards + MTP draft head (~95.5 GiB total)
huggingface-cli download nitinpanj/qwen38-flash-next-v3 \
    --local-dir ~/models/qwen38-flash-next-v3
```

### Step 3: Raise wired GPU memory limit & serve
```zsh
# Raise wired GPU memory limit once per boot (required on 64 GB Macs):
sudo sysctl iogpu.wired_limit_mb=59392

# Serve your existing model:
./slipstream serve --model ~/models/qwen38-flash-next-v3 --port 8090
```

> **First Run Note:** On first launch, Slipstream detects the multi-shard GGUF files and prepares optimized streaming package files into `<model>/prepared/` (~5–7 minutes). Subsequent launches load in **~10–15 seconds**.

The server exposes a standard OpenAI-compatible API (`http://127.0.0.1:8090/v1/chat/completions`) ready for curl, Oh My Pi (omp), Claude Code, or OpenCode.

---

## 2. Speed: llama.cpp Fork vs. Slipstream (Same V3 Checkpoint)

Here is a direct head-to-head comparison running the exact same 95.5 GiB model files across 6 reasoning and coding tasks on the same M5 Pro (64 GB unified memory, temperature 0.0):

| Domain / Task | Prompt Task | llama.cpp Fork | Slipstream | Speedup | llama.cpp TTFT | Slipstream TTFT |
|---|---|---:|---:|---:|---:|---:|
| **Math Reasoning** | GSM8K (eggs problem) | 24.0 tok/s | **43.6 tok/s** | **1.82x** | 4,024 ms | **2,337 ms** |
| **Math Derivation** | MATH-500 series ($p - q$) | 24.3 tok/s | **43.1 tok/s** | **1.77x** | 1,655 ms | **1,587 ms** |
| **Constraint Logic** | 3-chair deduction | 25.4 tok/s | **46.0 tok/s** | **1.81x** | 1,469 ms | **1,042 ms** |
| **Python Coding** | `merge_intervals` ($O(N \log N)$) | 19.7 tok/s | **35.0 tok/s** | **1.77x** | 1,507 ms | **1,070 ms** |
| **Systems Coding** | Rust CSV parser | 22.7 tok/s | **37.5 tok/s** | **1.65x** | 1,257 ms | **859 ms** |
| **Tech Writing** | Multi-head attention | 22.5 tok/s | **39.4 tok/s** | **1.75x** | 1,267 ms | **843 ms** |
| **AVERAGE** | Across all 6 tasks | **23.1 tok/s** | **40.8 tok/s** | **1.76x** | **1,863 ms** | **1,290 ms** |

*(Insert Image 1 here: 1_slipstream_vs_llamacpp_throughput.png)*

### What made Slipstream faster:
1. **Asynchronous layer-ahead prefetch (`fcntl(F_RDADVISE)`)**: In llama.cpp, synchronous page reads for missed expert matrices stalled the GPU on NVMe latency (~475 ms per chunk). In Slipstream, non-blocking read-ahead hints stream upcoming expert layers from SSD into RAM while the GPU is still executing the previous layer, cutting prefill staging latency by 28%.
2. **Hybrid MTP + Prompt Lookup speculation**: During tool calls and code generation, Prompt Lookup Decoding (PLD) matches prompt anchors in under 50 ns with 0 allocations, preventing draft rejections. This lifted tool-calling decode from 5.6 tok/s to over 45 tok/s.
3. **Metal GPU-mapped n-gram tables**: llama.cpp faulted on the 26.8 GiB n-gram table during prefill. Slipstream maps and gathers n-gram embeddings directly in Metal kernels.

---

## 3. Context Scaling: Real Telemetry up to 130,000 Tokens

On standard Transformers, decode slows down sharply as context grows because the KV cache swells and memory bandwidth saturates.

Qwen3.8-Flash-Next avoids that through its hybrid architecture:
- **48 recurrent linear DeltaNet layers** (fixed $128 \times 128$ hidden state, $O(1)$ memory growth with context).
- **Only 16 full-attention layers**.

Here is actual telemetry collected across **3,086 live requests** during real agent coding sessions on my M5 Pro (64 GB):

| Context Range (Tokens) | Live Runs | Average Decode | Median (p50) | Peak Decode | Average TTFT | Notes |
|---|---:|---:|---:|---:|---:|---|
| **< 1,000** | 314 | **41.5 tok/s** | 41.9 tok/s | 59.8 tok/s | 2.16 s | Short baseline |
| **1k – 4,000** | 21 | **41.0 tok/s** | 42.5 tok/s | 64.5 tok/s | 5.26 s | Small documents |
| **4k – 8,000** | 58 | **43.6 tok/s** | 43.2 tok/s | 67.2 tok/s | 7.36 s | Code review turns |
| **8k – 16,000** | 117 | **43.6 tok/s** | 44.6 tok/s | 58.2 tok/s | 7.91 s | Multi-file context |
| **16k – 32,000** | 562 | **38.2 tok/s** | 40.9 tok/s | 58.0 tok/s | 13.59 s | Deep agent session |
| **32k – 64,000** | 1,029 | **35.0 tok/s** | 37.5 tok/s | 55.6 tok/s | 13.24 s | Large repo refactor |
| **64k – 96,000** | 650 | **32.4 tok/s** | 34.7 tok/s | 53.9 tok/s | 12.81 s | Multi-turn transcript |
| **96k – 130,000** | 364 | **32.9 tok/s** | 33.3 tok/s | 43.8 tok/s | 7.95 s | Cache-hit deep turns |

*(Insert Image 2 here: 2_context_scaling_130k_telemetry.png)*

**Takeaway:** Decode speed stays between **33 and 44 tok/s** all the way out to 130k tokens. Even at 130k context, it generates tokens faster than stock llama.cpp did on a 500-token prompt.

---

## 4. Optional: Swift KV-Sparse Model Variant

If you want higher reasoning accuracy and lower KV cache memory, I also put together an optional Swift variant of this model: [Swift-Qwen3.8-Flash-Next-V3](https://huggingface.co/nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF).

### What Swift changes:
- **KV-Sparse Attention:** Replaces standard dense attention with KV-sparse layers distilled from Swift-1.5, cutting down RAM pressure at long contexts.
- **Spliced Q8 Donor Backbones:** Slices 686 high-precision Q8 donor tensors into resident backbone layers for sharper representations.
- **Concise Reasoning:** Distilled to eliminate repetitive thinking loops in deep contexts.

Both models run on Slipstream using the exact same engine command. Here is how they compare across 145 paired evaluation problems (temperature 0.0, seed 1234):

| Domain / Benchmark | Items | Original Flash-Next V3 | Swift-Flash-Next V3 | Accuracy Delta | Original Decode | Swift Decode |
|---|---:|---:|---:|---:|---:|---:|
| **AIME 2025** | 20 | 45.0% (9/20) | **45.0% (9/20)** | 0.0% | 44.3 tok/s | **44.3 tok/s** |
| **MATH-500 (L4–5)** | 35 | 60.0% (21/35) | **62.9% (22/35)** | **+2.9%** | 44.8 tok/s | **44.8 tok/s** |
| **GPQA Diamond** | 35 | 45.7% (16/35) | **54.3% (19/35)** | **+8.6%** | 44.8 tok/s | **44.8 tok/s** |
| **GSM8K** | 25 | 96.0% (24/25) | **96.0% (24/25)** | 0.0% | 45.6 tok/s | **45.6 tok/s** |
| **HumanEval** | 25 | 92.0% (23/25) | **92.0% (23/25)** | 0.0% | 40.6 tok/s | **40.6 tok/s** |
| **Hard Systems Logic** | 5 | 100.0% (5/5) | **100.0% (5/5)** | 0.0% | 39.2 tok/s | **39.2 tok/s** |
| **OVERALL** | **145** | **67.6% (98/145)** | **70.3% (102/145)** | **+2.8%** | **43.9 tok/s** | **44.4 tok/s** |

*(Insert Image 3 here: 3_swift_v3_quality_benchmarks.png)*

To run the Swift model instead:
```zsh
huggingface-cli download nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF \
    --local-dir ~/models/swift-qwen38-flash-next-v3

./slipstream serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090
```

---

## 5. Foundation for Qwen4

The core primitives in Slipstream:
- 512-route sparse MoE streaming with SSD prefetch
- QSA (Quasi-Sparse Attention) indexer & selection kernels
- Hyper-connection mixing and per-layer embedding gathers
- Metal GPU-mapped n-gram table gathers
- Single-lane speculative verification with PLD & MTP

...were built around this hybrid architecture. If **Qwen4** adopts a similar blueprint (hybrid linear recurrence + sparse attention + routed MoE experts), Slipstream should be able to run Qwen4 locally on consumer unified memory hardware on day one.

---

## 6. Hardware Tested & Porting to NVIDIA / AMD

- **Hardware tested:** All testing and benchmarking were done on an **Apple MacBook Pro (M5 Pro, 64 GB unified memory, 2 TB SSD)**.
- **CUDA / ROCm ports:** I don't have access to modern NVIDIA or AMD GPU hardware, so I can't build or test CUDA/ROCm backends myself.
- **If you have hardware and want to help port this:** If anyone in the community has NVIDIA or AMD hardware and wants to help bring expert streaming and hybrid speculation to Linux/Windows, **I'm happy to help collaborate on the port**. Feel free to open an issue on the repo or DM me.

---

## 7. Credits & Upstream

- **Splash Team (Incoai)**: Full credit to the creators of [Splash](https://github.com/incoai/splash). Their C++ Metal speculative decoding design and memory architecture provided the foundation for this work. I will prepare a clean PR proposing these Flash-Next and SSD streaming extensions to the Splash upstream repo in case they want to incorporate them.
- **ds4 Team**: For their insights on Metal router numerical precision (Taylor polynomial softplus expansion) and streaming scheduling designs.
- **Qwen Team**: For training Qwen3.8-Flash-Next and releasing the hybrid linear MTP architecture.
- **ukisai**: For the Swift-1.5 distillation work enabling KV-sparse reasoning.
- **bartowski & unsloth**: For donor quants and quantization tooling.
- **mihailescu2m**: For the initial expert streaming work in llama.cpp.
