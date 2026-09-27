# Slipstream GGUF Loading Plan (Direct V3 Ingestion)

**Goal:** Enable Slipstream to load and serve sharded GGUF models directly—specifically Nitin's daily `~/models/qwen38-flash-next-v3` (`Qwen3.8-Flash-Next-Q4_0-Q8out-v3-*.gguf`)—without requiring offline package conversion via `convert_qwen4exp.py`.

---

## 1. Problem & Motivation

* **Current Status**: Slipstream runs `qwen4exp` (Qwen3.8-Flash-Next) at ~41+ tok/s using bespoke Metal kernels (4-stream hyper-connections, QSA indexer, 512 MoE experts, n-gram gather).
* **The Friction**: Slipstream currently only accepts pre-converted Splash packages (`splash-packed-q4-qwen4exp` at `/Users/nitin/models/qwen38-flash-next-splash`). Creating this required:
  1. A massive 338 GB unquantized BF16 source checkpoint.
  2. Running `convert_qwen4exp.py` to quantize and tile into a separate 96.6 GB package.
  3. Maintaining redundant copies on disk.
* **The Opportunity**: Upstream Splash (`incoai/splash#132–#134`) proved that GGUF weights can be repacked on-the-fly via a GPU compute shader into memory-mapped zero-copy planes (`MDGG0001` layout).
* **The Target Model**: Nitin's daily model `~/models/qwen38-flash-next-v3` is already fully quantized:
  - `Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00001-of-00003.gguf` (40.5 GB)
  - `Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00002-of-00003.gguf` (40.5 GB)
  - `Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00003-of-00003.gguf` (21.5 GB)
  - Total: 102.5 GB across 3 shards.
  - Formats: `Q4_0` for linear projections and routed experts; `Q8_0` for the output head; `F32` for norms and embeddings; embedded MTP draft layer in block 48.

---

## 2. Architecture & Pipeline

```
  [ ~/models/qwen38-flash-next-v3/*.gguf ] (3 Shards)
                       │
                       ▼
  ┌──────────────────────────────────────────────┐
  │ 1. ShardedGgufFile (Multi-file Reader)        │
  │    - Scans headers across shards 1..3        │
  │    - Builds unified tensor catalog:          │
  │      name -> (shard_fd, byte_offset, shape)  │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ 2. Q4_0 / Q8_0 GPU Repack Shader             │
  │    - Adds Q4_0 format to QuantFormat.h       │
  │    - gguf_repack reorganizes nibbles & half-  │
  │      scales into coalesced GPU planes        │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ 3. Qwen4Exp GGUF Target Mapper               │
  │    - Maps GGUF names to Slipstream operators:│
  │      * blk.{i}.hyper_connection.*            │
  │      * blk.{i}.attn_qsa.*                    │
  │      * blk.{i}.ffn_gate_exps.* (512 experts) │
  │      * ngram.weight (20M x 2560 table)       │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ 4. Embedded MTP Drafter                      │
  │    - Extracts block 48 directly from GGUF    │
  │    - No external DFlash2 download required   │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
  [ Slipstream Metal Engine (~41 tok/s with LRU Expert Cache) ]
```

---

## 3. Implementation Phases

### Phase 1: Port Core GGUF Headers & Repack Engine
* Import the following from `splash-main`:
  * `runtime/model/GgufFile.hpp`, `GgufFile.cpp`
  * `runtime/model/GgufImage.hpp`, `GgufImage.cpp`
  * `runtime/metal/abi/QuantFormat.h`, `runtime/metal/abi/GgufRepack.h`
  * `runtime/metal/kernels/shared/gguf_repack.metal`

### Phase 2: Add `Q4_0` Quantization Support
Upstream Splash implemented K-quants (`Q4_K`, `Q5_K`, `Q6_K`) and `Q8_0`, but omitted `Q4_0`. Since V3 uses `Q4_0`:
* Add `GGUF_FMT_Q40` to `QuantFormat.h`:
  * Block elements: 32
  * Block bytes: 18 (one `half` float scale `d` + 16 bytes of 4-bit nibbles)
  * Meta: 2 bytes per 32-element group (`half` scale)
  * Plane0: 16 bytes per 32-element group (32 nibbles)
* Implement `q40_dequantize_32` in `runtime/metal/kernels/common/quant_formats.h`.
* Add repack kernel specialization in `runtime/metal/kernels/shared/gguf_repack.metal`.

### Phase 3: Implement Sharded GGUF Reader (`ShardedGgufFile`)
* Standard GGUF sharding convention puts full metadata and kv-pairs in shard 1, with tensors distributed across shards 1, 2, and 3.
* Create `ShardedGgufFile`:
  * Accepts a directory path or list of file paths (e.g. `Qwen3.8-Flash-Next-Q4_0-Q8out-v3-*.gguf`).
  * Opens all shard file descriptors.
  * Collects tensor records from each shard header into a single hash map.
  * Provides zero-copy `pread` / `mmap` slices for any requested tensor regardless of which shard it resides in.

### Phase 4: `Qwen4ExpGgufTarget` Mapper
* Wire the repacked GGUF planes to the existing `qwen4exp` model structure:
  * **Layer pattern**: 48 layers (3 GDN layers followed by 1 full attention layer).
  * **512 Routed Experts**: Map `blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight` into Slipstream's LRU expert cache manager.
  * **Hyper-Connections**: Map 4-stream low-rank weights (`blk.{i}.hyper_connection.*`).
  * **QSA Indexer**: Map sparse attention indexer projections (`blk.{i}.attn_qsa.*`).
  * **N-Gram Table**: Direct mapping of the 20M x 2560 table.
  * **Output Head**: Extract `output.weight` as `Q8_0`.

### Phase 5: Direct MTP Drafter Wiring
* The V3 GGUF already contains the Multi-Token Prediction layer (block 48).
* Ingest block 48 weights directly into Slipstream's MTP drafting module (`models/qwen4exp/kernels/mtp_pick.metal`), eliminating the need for an external draft checkpoint.

### Phase 6: Validation & Verification
1. **Arithmetic Check**: Compare repacked tensor values bitwise against `llama.cpp` dequantization oracle.
2. **First Token Match**: Verify that prompt evaluation on standard prompts matches the existing packed package logits bit-for-bit.
3. **Throughput Parity**: Verify that decode throughput stays at or above **41 tok/s** on the M5 Pro.

---

## 4. Expected Benefits
1. **Zero Storage Redundancy**: Drops the requirement for `/Users/nitin/models/qwen38-flash-next-splash` (saving ~97 GB of disk).
2. **Instant Model Swaps**: Test new GGUF quants or checkpoints directly from `llama.cpp` builds without writing conversion scripts.
3. **Daily Workflow Continuity**: Point `slipstream serve` straight at `~/models/qwen38-flash-next-v3` for `omp` and `pi`.
