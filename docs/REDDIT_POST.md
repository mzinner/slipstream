# [Release] Running 95.5 GiB Qwen3.8-Flash-Next at 41–52 tok/s on a 64GB Mac: Introducing Slipstream, Swift KV-Sparsity, and 130k Context Scaling

Following up on my two previous posts:
1. **[Splash Engine native 8-bit on r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1wmbbf9/splash_engine_qwen3827b_in_native_8bit_at_3755/)** (extending Splash to native Q8 and analyzing 256k context scaling).
2. **[Qwen3.8-Flash-Next on r/Qwen_AI](https://www.reddit.com/r/Qwen_AI/comments/1wkd7pm/qwen38flashnext_955_gib_on_a_64gb_mac_at_27_toks/)** (running 95.5 GiB on a 64GB Mac at ~27 tok/s using a custom expert-streaming fork of llama.cpp).

Since those posts, I built and tuned **Slipstream**—a dedicated C++ Metal inference engine designed to stream big models from SSD and speculate ahead.

The takeaway first:
- **Slipstream runs Qwen3.8-Flash-Next V3 at 41–52 tok/s** on an M5 Pro Mac (64 GB unified memory). That is a **1.76x speedup** over the llama.cpp fork (23.1 -> 40.8 tok/s average across standard benchmarks), with 31% lower time-to-first-token.
- **I am releasing the Swift version of this model**: `Swift-Qwen3.8-Flash-Next-V3`. It incorporates KV-sparsity and distilled concise reasoning, pushing MATH-500 from 60.0% to 62.9% and GPQA Diamond from 45.7% to 54.3% (+8.6%) on our 145-item evaluation suite, while decoding up to 52–67 tok/s on structured tasks.
- **Context scaling does not collapse**: Across 3,086 real telemetry points, decode speed stays between **33 and 44 tok/s all the way out to 130,000 tokens**.

Everything is open source:
- **Engine Repo:** [github.com/npanj/slipstream](https://github.com/npanj/slipstream)
- **Base Model (GGUF):** [huggingface.co/nitinpanj/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF](https://huggingface.co/nitinpanj/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF)
- **Swift KV-Sparse Model (GGUF):** [huggingface.co/nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF](https://huggingface.co/nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF)

*(Note on naming: This is the codebase previously developed as slipstream-v2, now published cleanly as Slipstream. The earlier experimental tree has been archived locally as slipstream-orig).*

---

## 1. Head-to-Head: llama.cpp Fork vs. Slipstream

In my last post, my llama.cpp fork topped out around 27.6 tok/s on short prompts and slowed to ~20.6 tok/s at 29k context. 

Here is a direct, sequential run of the exact same 95.5 GiB checkpoint (`qwen38-flash-next-v3`) across 6 standardized reasoning and coding tasks on the same M5 Pro (64 GB Unified Memory, temperature 0.0):

| Domain / Task | Prompt Task | llama.cpp Fork | Slipstream | Speedup | llama.cpp TTFT | Slipstream TTFT |
|---|---|---:|---:|---:|---:|---:|
| **Math Reasoning** | GSM8K (eggs problem) | 24.0 tok/s | **43.6 tok/s** | **1.82x** | 4,024 ms | **2,337 ms** |
| **Math Derivation** | MATH-500 series ($p - q$) | 24.3 tok/s | **43.1 tok/s** | **1.77x** | 1,655 ms | **1,587 ms** |
| **Constraint Logic** | 3-chair deduction | 25.4 tok/s | **46.0 tok/s** | **1.81x** | 1,469 ms | **1,042 ms** |
| **Python Coding** | `merge_intervals` ($O(N \log N)$) | 19.7 tok/s | **35.0 tok/s** | **1.77x** | 1,507 ms | **1,070 ms** |
| **Systems Coding** | Rust CSV parser | 22.7 tok/s | **37.5 tok/s** | **1.65x** | 1,257 ms | **859 ms** |
| **Tech Writing** | Multi-head attention | 22.5 tok/s | **39.4 tok/s** | **1.75x** | 1,267 ms | **843 ms** |
| **AVERAGE** | Across all 6 tasks | **23.1 tok/s** | **40.8 tok/s** | **1.76x** | **1,863 ms** | **1,290 ms** |

### Why is Slipstream so much faster?
1. **Predicted SSD Read-Ahead**: Rather than waiting for the GPU to request routed experts, Slipstream predicts which experts are required for upcoming layers during speculative steps and streams them from SSD into a unified Metal buffer ahead of execution (+4.4% net throughput gain).
2. **PLD-First Speculative Verification**: Slipstream pairs neural MTP (Multi-Token Prediction) draft heads with Prompt Lookup Decoding (PLD). During structured tool-calling and code synthesis, PLD queries prompt anchors in under 50 ns with 0 allocations, lifting tool decode throughput from 5.6 tok/s to over 45 tok/s.
3. **Metal GPU-Mapped N-Gram Gathering**: In llama.cpp, the 26.8 GiB n-gram embedding table caused page faults and CPU stalls during prompt ingestion. Slipstream executes gather operations directly on Metal with memory-mapped tables, keeping TTFT low.

---

## 2. Plain Base vs. Swift KV-Sparse Model

Alongside base Flash-Next V3, I am releasing the **Swift version** (`Swift-Qwen3.8-Flash-Next-V3`). 

Swift replaces standard dense attention layers with KV-sparse layers (distilled from Swift-1.5) and splices 686 high-precision Q8 donor tensors into resident backbone blocks.

Both run on Slipstream using the exact same engine command. Here is how they compare across 145 rigorous evaluation items (temperature 0.0, seed 1234):

| Domain / Benchmark | Items | Plain Flash-Next V3 | Swift-Flash-Next V3 | Accuracy Delta | Plain Decode | Swift Decode |
|---|---:|---:|---:|---:|---:|---:|
| **AIME 2025** | 20 | 45.0% (9/20) | **45.0% (9/20)** | 0.0% | 44.3 tok/s | **44.3 tok/s** |
| **MATH-500 (L4–5)** | 35 | 60.0% (21/35) | **62.9% (22/35)** | **+2.9%** | 44.8 tok/s | **44.8 tok/s** |
| **GPQA Diamond** | 35 | 45.7% (16/35) | **54.3% (19/35)** | **+8.6%** | 44.8 tok/s | **44.8 tok/s** |
| **GSM8K** | 25 | 96.0% (24/25) | **96.0% (24/25)** | 0.0% | 45.6 tok/s | **45.6 tok/s** |
| **HumanEval** | 25 | 92.0% (23/25) | **92.0% (23/25)** | 0.0% | 40.6 tok/s | **40.6 tok/s** |
| **Hard Systems Logic** | 5 | 100.0% (5/5) | **100.0% (5/5)** | 0.0% | 39.2 tok/s | **39.2 tok/s** |
| **OVERALL** | **145** | **67.6% (98/145)** | **70.3% (102/145)** | **+2.8%** | **43.9 tok/s** | **44.4 tok/s** |

### Why use the Swift variant?
- **Better reasoning density**: Swift was distilled to avoid repetitive, circular thinking in deep contexts. It scores **+8.6% higher on GPQA Diamond** and **+2.9% on MATH-500**.
- **Smaller KV footprint**: The KV-sparse attention heads allow running deeper context windows with less RAM pressure on a 64 GB Mac.
- **Top speed**: On structured extraction and code generation tasks, Swift reaches up to **52–67 tok/s** under Smart PLD drafting.

---

## 3. Context Scaling: Live Telemetry up to 130k Tokens

Standard Transformers suffer from an attention decode collapse as context expands: the KV cache swells and memory bandwidth saturates.

Qwen3.8-Flash-Next uses a hybrid architecture:
- **48 recurrent linear DeltaNet layers** (fixed $128 \times 128$ hidden state, $O(1)$ memory growth with context).
- **Only 16 full-attention layers** (further sparsified in the Swift build).

Below is empirical telemetry across **3,086 live requests** run during real agent sessions on an M5 Pro (64 GB):

| Context Range (Tokens) | Live Runs | Average Decode | Median (p50) | Peak Decode | Average TTFT | Notes |
|---|---:|---:|---:|---:|---:|---|
| **< 1,000** | 314 | **41.5 tok/s** | 41.9 tok/s | 59.8 tok/s | 2.16 s | Short baseline |
| **1k – 4,000** | 21 | **41.0 tok/s** | 42.5 tok/s | 64.5 tok/s | 5.26 s | Small documents |
| **4k – 8,000** | 58 | **43.6 tok/s** | 43.2 tok/s | 67.2 tok/s | 7.36 s | Code review turns |
| **8k – 16,000** | 117 | **43.6 tok/s** | 44.6 tok/s | 58.2 tok/s | 7.91 s | Multi-file context |
| **16k – 32,000** | 562 | **38.2 tok/s** | 40.9 tok/s | 58.0 tok/s | 13.59 s | Deep agent session |
| **32k – 64,000** | 1,029 | **35.0 tok/s** | 37.5 tok/s | 55.6 tok/s | 13.24 s | Large repository refactor |
| **64k – 96,000** | 650 | **32.4 tok/s** | 34.7 tok/s | 53.9 tok/s | 12.81 s | Multi-turn transcript |
| **96k – 130,000** | 364 | **32.9 tok/s** | 33.3 tok/s | 43.8 tok/s | 7.95 s | Cache-hit deep turns |

*(See benchmark and context scaling plot in the repo: `docs/context_scaling_and_benchmark.png`)*

**The key finding:** Decode throughput **never collapses**. It stays between **33 and 44 tok/s** across the entire 130k context window. At 130k context, it is still generating tokens faster than stock llama.cpp could generate on a 500-token prompt.

---

## 4. Foundation for Qwen4

The core primitives implemented in Slipstream:
- 512-route sparse MoE streaming with SSD prefetch
- QSA (Quasi-Sparse Attention) indexer & selection kernels
- Hyper-connection mixing and per-layer embedding gathers
- Metal GPU-mapped n-gram table gathers
- Single-lane speculative verification with PLD & MTP

...were engineered to match the upcoming model generation. Assuming **Qwen4** follows the Flash-Next architectural blueprint (hybrid linear recurrence + sparse attention + routed MoE experts), Slipstream can serve as a direct foundation to run Qwen4 locally on consumer unified memory hardware on day one.

---

## 5. Hardware Tested & Porting to NVIDIA / AMD GPUs

- **Testing Hardware:** All development and benchmarking were done exclusively on an **Apple MacBook Pro (M5 Pro, 64 GB Unified Memory, 2 TB SSD)**.
- **Porting to CUDA / ROCm:** Because I do not have access to modern NVIDIA or AMD GPU hardware, I cannot test or build CUDA/ROCm backends for this architecture myself.
- **Collaboration offer:** If anyone in the community has NVIDIA (CUDA) or AMD (ROCm) hardware available and wants to bring these expert-streaming and hybrid-speculation gains to those platforms, **I am happy to collaborate and help with the port**. Feel free to open an issue or reach out via DM/GitHub.

---

## 6. How to Run It

### 1. Clone & Build
```zsh
git clone https://github.com/npanj/slipstream.git
cd slipstream
make -j4
```

### 2. Download Model (Swift or Base)
```zsh
# Option A: Swift KV-sparse model (recommended for reasoning & agent use)
huggingface-cli download nitinpanj/Swift-Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF \
    --local-dir ~/models/swift-qwen38-flash-next-v3

# Option B: Plain base model
huggingface-cli download nitinpanj/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-GGUF \
    --local-dir ~/models/qwen38-flash-next-v3
```

### 3. Launch Server
```zsh
# Launch Swift V3
./slipstream serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090
```
On first launch, Slipstream automatically packs the multi-shard GGUF files into native streaming pages (`prepared/`). The server exposes a standard OpenAI-compatible API (`http://127.0.0.1:8090/v1/chat/completions`) ready for curl, Oh My Pi (omp), or your preferred coding agent.

---

## 7. Credits & Acknowledgments

- **Splash Team (Incoai)**: Full credit to the creators of Splash ([github.com/incoai/splash](https://github.com/incoai/splash)). Their C++ Metal speculative decoding design and memory architecture provided the foundation for this work. I will prepare a clean PR/patch proposing these Flash-Next and SSD streaming extensions to the Splash upstream repo in case they would like to incorporate them.
- **ds4 Team**: For their valuable insights on Metal router numerical precision (Taylor polynomial softplus expansion) and streaming scheduling designs.
- **Qwen Team**: For training Qwen3.8-Flash-Next and open-sourcing the hybrid linear MTP architecture.
- **ukisai**: For the Swift-1.5 distillation work enabling KV-sparse reasoning.
- **bartowski & unsloth**: For donor quants and quantization tooling.
- **mihailescu2m**: For initial expert streaming concepts in llama.cpp.
