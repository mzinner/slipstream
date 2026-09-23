# Decisions — qwen4exp port

## The layer order, verified exactly

This is the single most valuable fact in this directory. A layer composed this
way, from real layer-0 weights, matches `Qwen4ExpTextGatedResidual` from
transformers 5.16.1 with **max abs diff 0.000e+00**:

```
attention hyper-connection  (norm -> mix-down -> mix-up)
  -> mixer block (GDN, or full attention every 4th layer)
  -> inject back into the residual
mlp hyper-connection        (norm -> mix-down -> mix-up)
  -> expert block
  -> inject back into the residual
```

Residual width is 10240 = 4 streams of 2560. The check script is
`models/qwen4exp/tools/checks/check_layer_composition.py` — re-run it if the forward path
disagrees with the reference.

**Why this matters:** every kernel was already verified against its own
reference in isolation. None of those could catch a wiring error — wrong order,
wrong residual stream carried, injection in the wrong place. That is where every
interface bug in this port actually lived.

## Norm gain is `(1 + weight)`

`Qwen4ExpTextRMSNorm` computes `output * (1.0 + self.weight.float())`, and the
weight is zero-initialised. Using the stored gain directly doubles every gain.
Real gains run -5.94 to +6.63, so this is badly wrong, not subtly wrong.

## Q4 layouts are not interchangeable

- **Matmul operands are tiled**: `[tile][group][row][64 codes]`.
- **Lookup tables are plain row-major**, three separately aligned runs.
  Embeddings are gathered a row at a time, never multiplied as a tile.

## Quantizer anchors at the larger-magnitude extreme

Not at the minimum. That value lands on code zero and the error falls on the
smaller side, which is why scales can be negative. Codes are quantized against
the **bf16** stored scale/bias, not the float32 ones — the reader only ever sees
the stored values. Getting this wrong gave a 55% code mismatch that still loaded
and ran.

## Tiling and group sizes

- Experts tile at **StorageN 128** (`kQ4ExpertStorageN`), not 256 — experts are
  640 wide. Chosen by Nitin over padding to 256.
- N-gram rows are 160 wide, so they use **group 32** (`kQ4FineGroupElements`).
- Router is padded to 256 regardless of live expert count. Do not conflate
  `params.experts` with the router width — that broke `moe-metal` once.
  Handled via a `RouterWidth` template with entries at 256 and 512.

## Linear attention needed no new kernels

GDN already compiles for `{16, 48, 128, 10240, 16640}`. This was the single
biggest scope reduction in the port.

## Sparse attention is a loop head plus a mask term

Attention already reads through a page table and takes each page's position from
its logical index. Walking a *list* of logical pages instead of a range keeps
position, causality and the softmax exactly as they were. Two bugs found here:
splits must partition whatever list is walked (the first version had every split
traverse the whole selection, 65x the work), and selection order must come from
a prefix sum, not an atomic slot claim, or it is non-deterministic.

## Memory: accounting, not machinery

`WeightStore` already maps weights `PROT_READ / MAP_SHARED`, so they are
file-backed and reclaimable. What refused to start the model was
`fixedRuntimeBytes()` counting all 96.61 GiB as held. `ModelMemoryFootprint` now
splits resident from streamable. Result: 6.98 GiB resident + ~38 GiB cache,
against the llama.cpp fork's 5 GiB + 36 GiB at 25 tok/s.

**Trap:** `streamableWeightsBytes` and `streamCacheBytes` are at the **end** of
`ModelMemoryFootprint` on purpose. Three call sites use positional brace-init;
inserting fields mid-struct silently shifts every later value, compiles clean,
and passes the CPU suite.

## The draft is a placeholder

> **Superseded (2026-09-22):** the placeholder still ships (the format needs one) but never runs: the model's own MTP head drafts, up to 5 guesses a step, output exact.

The package format requires a DFlash 2 draft. Flash-Next has no such draft, only
an MTP head. A zero-filled placeholder is written so packages validate. Expect
no speculation speedup until a real draft is trained. Nitin deferred that.

## Kv2Group12 Attention Kernels

Qwen4Exp attention uses 24 query heads and 2 KV heads (head dimension 128), which
is group size 12 (`Kv2Group12`). Dedicated prefill and decode (verify) projections,
gates, and Q8 split/reduce kernels were instantiated and wired into `PagedAttention`
to support this layout cleanly without padding.

## Command Buffer Chunking for Models Exceeding Device RAM

A single Metal command buffer cannot reference resources whose sum exceeds the
device working set ceiling (~58 GiB on 64 GiB Apple Silicon). Attempting to commit
all 48 layers (96.61 GiB) in a single command buffer triggers
`kIOGPUCommandBufferCallbackErrorOutOfMemory`. `MetalBackend::submitCommandAsync`
now chunks dispatches into batches bounded by `recommendedMaxWorkingSetBytes / 2`.
All chunks commit in order to the serial command queue, enabling models larger
than physical RAM to stream on demand without exhausting GPU driver residency limits.

## GDN Output Gate Activation is Sigmoid, Not SiLU

Upstream `transformers` (`modeling_qwen4_exp.py`) uses `Qwen4ExpTextRMSNormGated` with `output_gate_type: "sigmoid"`, computing `rms_norm(x) * sigmoid(gate)`. Splash's `gdn_primitives.h` had originally implemented `gate * sigmoid(gate)` (SiLU), which inflated GDN hidden states by ~2.8x. Changing `gdn_gate_phase` to use `1.0f / (1.0f + fast::exp2(-1.44269504089f * gate))` matched PyTorch output to bf16 precision and resolved text generation quality. Synthetic unit tests `gdn_metal_test.mm` and `gdn_decode_metal_test.mm` were updated to use `sigmoid` accordingly.

## Vectorized Hyper-Connection Kernels (4.7x GPU Speedup)

Dispatch profiling of the full 48-layer model revealed that `hyper_connection_normalize` (641.7 ms) and `hyper_connection_mix` (200.0 ms) accounted for 94% (841.8 ms out of 893 ms) of total GPU time due to unvectorized 16-bit scalar loads and low threadgroup occupancy. Vectorizing both kernels to 64-bit `bfloat4` aligned loads and caching the full 320-element low-rank vector in threadgroup SRAM reduced per-dispatch time by >3.15x and total GPU decode time from 504 ms to 190 ms across all 48 layers.

## Cyclic LRU Paging Thrashing & Expert Working Set Isolation

> **Superseded (2026-09-22):** no layer is resident now; every expert streams through per-layer caches read with direct preads (F_NOCACHE), so page-cache thrashing no longer applies.

On a 64 GB Mac, model files total 96.6 GiB (68.2 GiB in 48 layer files, 26.8 GiB n-gram, 1.6 GiB head/embed). Because `WeightFile` creates a zero-copy `MTLBuffer` for the entire 1.42 GiB layer file, the macOS IOGPU driver enforces residency on the full file on commit, even though a token only touches 10 experts (26 MiB) in that layer. This causes cyclic LRU thrashing: by layer 48, layers 0..15 are evicted, forcing every token to re-read ~45–60 GiB from SSD (~7.4 s/tok).
- Usable RAM allows 37 layers to remain 100% resident in physical RAM.
- Calling `madvise(MADV_DONTNEED)` on the expert weights of the remaining streaming layers (37..47) at the end of each decode step ensures the OS reclaims streaming pages first, preventing eviction of resident layers.
- For true sub-second decode (~200-250 ms), streaming layers must bind and stage only the 10 active experts (26 MiB/layer = 286 MiB total) via an in-memory expert cache rather than binding the full 1.42 GiB file.

## Sub-Buffer Residency Isolation (`detachStreamingLayer`)

Binding even a single 5 KB norm from a 1.42 GiB `WeightFile` forces Apple's IOGPU driver to enforce GPU residency on the entire 1.42 GiB buffer on commit. `detachStreamingLayer()` detaches all non-expert tensors (attention projections/norms, GDN weights/norms, indexer, MLP hyper-connection, router, shared expert) into standalone, dedicated `MTLBuffer`s (~35 MB per layer). The 1.42 GiB file buffer is NEVER bound to Metal during decode, keeping the active GPU working set under 12 GiB for all 48 layers combined.

## Persistent Per-Layer Expert Cache with LRU Slot Tracking

> **Superseded (2026-09-22):** capacity is now 288 per layer (llama.cpp's 36 GiB budget, minus a 144-expert prompt staging buffer), eviction is least-used first, caches are mlock'd and in a GPU residency set.

To eliminate copying 20+ active experts on every token step, each layer maintains an in-memory `Qwen4ExpLayerExpertCache` with dedicated `cacheGate`, `cacheUp`, and `cacheDown` `MTLBuffer`s of configurable capacity (default 64 experts per layer = 9.4 GiB across all 48 layers).
- Active experts are mapped to persistent cache slots using `expertToSlot` and `slotToExpert` arrays.
- Cache hits touch `lruTime[slot]` with zero copy overhead (0 ns).
- Cache misses allocate the next free slot or evict the least-recently-used slot not used in the current step.
- Misses are staged concurrently via GCD `dispatch_apply` across CPU performance cores.
- Hit rates exceed 92% in steady state (reducing misses to ~1.7 per layer), dropping staging latency from 6,162 ms to ~300 ms across all 48 layers.

## CommandGraph Move Invariant in Multi-Command Execution

In Splash's pipeline, `Runtime.mm` encodes `encodeBatchVerifyInput` and `encodeBatchEmbedding` into `graph` before invoking `targetModel.addVerify(graph, ...)`. Any engine path that internally splits the execution graph into multiple command buffers must begin with `metal::CommandGraph residentGraph = std::move(graph);` rather than constructing a default-initialized `CommandGraph`. Otherwise, layer 0 dispatches execute on GPU before input token embeddings are written to `buffers.hidden[0]`, corrupting the first layer's activations.

## Prefill Staged Streaming Expert Cache Integration

> **Superseded (2026-09-22):** the monolithic fallback stalled the GPU for ~10 s on page faults and is gone; prompts run expert-first waves (see below).

Prefill previously bound all 48 layer files (68.2 GiB) into a single monolithic command graph, causing Apple's IOGPU driver to demand-page gigabytes of weight files from SSD on cold start (~11.1 seconds for 5 prompt tokens) and dispatch 512-expert MoE kernels.
- `Qwen4ExpTarget::addPrefill` is now integrated with the per-layer staged expert cache: only the ~20-30 active experts per layer selected by the router across the prompt tokens are staged into `layer[L].expertCache`.
- MoE execute dispatches against the 64-expert cache buffer rather than the monolithic 512-expert file.
- Prefill latency dropped from 11.1s down to **549 ms (19x speedup)**.
- **Warm Decode Cache Hand-off:** Prefill leaves the prompt's active domain experts warm in `layer[L].expertCache`, eliminating the cold-cache penalty on decode token 1 (staging time dropped from 6,162 ms to **159 ms**).
- **Graceful Monolithic Fallback:** If an exceptionally long prompt activates more unique experts in a single layer than `cache.capacity` (64), `stageActiveExperts` returns `false` and automatically falls back to `weights.layers[L].ffn` for that layer, guaranteeing zero numerical regression or memory overflow regardless of sequence length.

## Everything but the routed experts is 8-bit (2026-09-22)

A fake-quantized reference run (`reference_logits.py --quant`) reproduced the
engine's error to 0.1%, so the long-context drift was rounding, not a bug.
4-bit mixers, head and embedding cost most of it. Now: routed experts 4-bit;
mixers, output head, embedding and hyper-connection weights 8-bit; router 8-bit.
Code prompt 81% -> 91% same top pick. Rejected: 3-bit experts (83%), expert
groups of 32 and error-minimizing ranges (both worse, unexplained).

## Hyper-connection weights use the tiled 8-bit matrix layout

So prompts run them as matrix products (they were 53% of prompt GPU time in
scalar kernels) while decode keeps its own kernels over the same bytes. The
down projection is padded from 320 to 512 outputs to fill whole tiles.

## Prompts run experts in waves, each expert read once per layer per chunk

Stage 0 runs cached experts; missing ones among the chunk's most-used go into
cache slots (so decode inherits them), the rest into a 144-expert staging
buffer whose two halves alternate so reads overlap GPU work. Waits use a
GPU-only "stage done" event: the shared pipeline event is also raised by the
host, which let a wait return before the GPU had finished (a real race).

## Prompt chunks are per package

`SPLASH_PREFILL_TOKEN_BUDGET` (4096) is the built maximum; the scheduler's
chunk comes from the manifest's `prefill_token_budget`. qwen4exp uses 4096:
bigger chunks re-read fewer experts per token.

## Dense attention past 2,048 tokens stays

The reference model drops tokens through a sparse indexer past 2,048. On real
text it then predicts worse (perplexity 10.3 vs 4.5 past 3K), so matching it
would make Splash worse. The indexer kernels exist but stay unwired.

## Decode hand-offs stay on the host

Each layer hands control to the CPU (~164 us) to pick and load experts. Moving
routing to the GPU would need the CPU to signal a running command buffer;
Metal doesn't make host writes visible there (probe: handoff_latency.mm).


## Draft head scores the 64K most common tokens (2026-09-22)

Its guesses are checked by the full head, so answers can't change. 64K cut
the draft 12.3 -> 8.4 ms a step with the same tokens per step; 32K made the
head's confidence (softmax over fewer tokens) too high, so it guessed further
and more guesses were rejected. Supersedes the earlier note that shrinking
the draft vocabulary would cost more than it saves.

## Expert cache must fit in free memory at load (2026-09-22)

The loader refuses when free memory < cache + 2 GiB. A 10% margin (like the
runtime governor) refused the normal 34 GiB setup, which leaves ~4 GiB.

## Hyper-connection kernels stay two launches (2026-09-22)

Merging down and up-mix needs every threadgroup to wait on the others
mid-launch. Metal does not promise they all run at once, so it can deadlock
the GPU. Three safe variants measured no faster.

## 2026-09-22 — Model code lives in `models/<name>/` (Slipstream phase 4)

A model folder holds its layout, loader, forward pass, own kernels, ABI headers,
converter, checks and benchmarks. It may launch its own kernels; it may not
depend on the engine or another model. Shared code reaches it only through four
files (`ModelDescriptor.hpp`, `ModelFactory.hpp`, `QwenTarget.cpp`, `Runtime.mm`).
`check_architecture.py` enforces this; the build fingerprint hashes `models/`.
Kept `runtime/` as the shared root (no rename to `core/`): the rename adds churn
and no checkable boundary. The installer and Homebrew packaging stay for now:
they are wired into 13 tests, so removing them is its own step.
