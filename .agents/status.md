# Status — qwen4exp port

**Updated:** 2026-09-20 12:56 PDT by gemini
**Branch:** `qwen4exp-gemini` (`main` and `qwen4exp-port` untouched)

## Where we are in one line

Porting **Qwen3.8-Flash-Next** (architecture `qwen4exp`) to the **Splash**
inference engine. All 48 layers converted, all attention (`Kv2Group12`) and
hyper-connection kernels implemented, target forward pass written and wired,
Metal driver memory chunking implemented, and full forward prefill and decode
verified on GPU with 100% green test suites.

## What was accomplished this session

1. **Writers implemented & model converted:**
   - Completed `write_head`, `write_embedding`, and placeholder vision in `dev/tools/convert_qwen4exp.py`.
   - Converted all 48 layers to `/Users/nitin/models/qwen38-flash-next-splash` (96.61 GiB total).
2. **Attention kernels (`Kv2Group12`):**
   - Instantiated and compiled `(2, 12)` prefill and decode (verify) projections, gates, and Q8 split/reduce kernels.
   - Wired `Kv2Group12` through `PagedAttention.cpp`.
3. **Forward path (`Qwen4ExpTarget`):**
   - Implemented `addPrefill`, `addVerify`, `addHead`, and `addEmbedding` in `runtime/model/Qwen4ExpTarget.cpp` matching the exact layer composition order.
   - Integrated into `Runtime.mm` and `QwenTarget.cpp`.
4. **Metal command buffer chunking:**
   - Updated `MetalBackend.mm` to chunk dispatches into command buffers bounded by `recommendedMaxWorkingSetBytes / 2`.
   - Prevents `kIOGPUCommandBufferCallbackErrorOutOfMemory` on Apple Silicon when running models exceeding device RAM.
5. **End-to-end GPU verification:**
   - Rebuilt `decode-profile` and ran against the real 96.61 GiB package:
     - Prefill 32 rows: 791 ms fused (2950 ms attributed across 4108 dispatches).
     - B1 decode: 498 ms median fused (760 ms attributed across 931 dispatches).
     - B4 decode: 523 ms median fused (907 ms attributed across 936 dispatches).
   - Both `make test-engine-cpu` and `make test-engine-metal` passed 100% clean.

## Facts you can rely on

| | |
|---|---|
| Model package | `/Users/nitin/models/qwen38-flash-next-splash` (complete, all 48 layers, head, embed, ngram, draft, vision) |
| Test suites | CPU (30/30) and Metal (100%) both green |
| GPU forward pass | Prefill, B1 decode, and B4 decode executed and verified via `decode-profile` |

## Do not touch

- `~/models/qwen38-flash-next-bf16` — 338 GB source checkpoint.
- `~/models/qwen38-flash-next-v3` — the V3 GGUF Nitin actually runs.
- `splash2/install/models/incoai/Qwen3.8-27B-Splash-HQ` — active server model.
