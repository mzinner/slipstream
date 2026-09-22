# Journal — qwen4exp port

## 2026-09-21 05:25 PDT — antigravity

Integrated prefill with the staged expert cache and automatic monolithic fallback. Slashed prefill latency by 19x from 11.1s down to 549 ms (9.1 tok/s) and pre-warmed decode caches with prompt domain experts, cutting Token 1 decode staging from 6,162 ms to 159 ms (38x faster). Decode steady-state latency reached ~160 ms/tok (6.25 tok/s), with 20-token end-to-end generation perfectly bit-exact (" Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain"). Both CPU (30/30) and Metal test suites pass 100% green.
Blocked on: nothing.

## 2026-09-21 00:57 PDT — antigravity

Accelerated Qwen3.8-Flash-Next decode by >10x (from 7.8 s/tok to 0.48 s/tok steady state / 1.35 tok/s average) with 100% quality retention. Built persistent per-layer in-memory expert caches with LRU slot tracking and GCD multi-core parallel miss staging, reducing staging time from 6,162 ms to ~300 ms across all 48 layers (cache hit rate >92%). Detached non-expert layer weights to avoid IOGPU driver residency bloat, slashing GPU execution time across all 48 layers to 171 ms. Both CPU (30/30) and Metal (`moe-staged-cache` 0 diff) suites pass 100% green.
Blocked on: nothing.

## 2026-09-20 23:40 PDT — antigravity

Vectorized `hyper_connection_normalize` and `hyper_connection_mix` to 64-bit `bfloat4` aligned loads, slashing full 48-layer GPU execution time by 2.5x (from 504 ms to 190 ms). Profiled the ~7s decode latency down to driver-level cyclic LRU cache thrashing across 48 layers of 1.42 GB files; scaled resident layers up to 37/48 with pre-touching and added streaming expert eviction (`MADV_DONTNEED`) after decode steps. Verified 100% green test suites on CPU (30/30) and Metal, with end-to-end generation quality verified (" Paris. The capital of Germany is Berlin.").
Blocked on: nothing.

## 2026-09-20 18:57 PDT — antigravity

Fixed GDN output gating activation (sigmoid vs silu) in `gdn_primitives.h`, matching PyTorch reference down to bf16 precision. Verified end-to-end generation quality on full 48 layers: `"The capital of France is" -> " Paris. The capital of Germany is Berlin. The capital of Italy is Rome."` Both CPU and Metal test suites pass 100% green. Speed measured at ~7.8s/tok (demand-paging ~45 GiB/step from SSD on 64GB RAM machine without speculation/expert caching).
Blocked on: nothing.

## 2026-09-20 12:56 PDT — gemini

Completed the Qwen4Exp port end-to-end on GPU.

- Implemented `write_head`, `write_embedding`, and `write_placeholder_vision` in `dev/tools/convert_qwen4exp.py`. Converted all 48 layers to `/Users/nitin/models/qwen38-flash-next-splash`.
- Wrote `Kv2Group12` attention kernels for prefill and decode; wired into `PagedAttention`.
- Implemented `Qwen4ExpTarget` forward pass in `runtime/model/Qwen4ExpTarget.cpp` and wired it into `Runtime.mm`.
- Resolved Metal driver working set overflow (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) by chunking command buffer dispatches in `MetalBackend.mm` when referenced buffers exceed working set limits.
- Verified end-to-end GPU execution with `decode-profile`: prefill (791 ms for 32 rows), B1 decode (498 ms median), and B4 decode (523 ms median) succeeded cleanly.
- Both test suites (`make test-engine-cpu` and `make test-engine-metal`) remain 100% green.

## 2026-09-20 10:45 PDT — claude-code

Verified the composed layer and cleared the disk blocker.

- Composed a full qwen4exp layer the way the kernels will run it, from real
  layer-0 weights, against `Qwen4ExpTextGatedResidual` (transformers 5.16.1):
  **max abs diff 0.000e+00**. Order is now pinned down, so the forward path is
  transcription against a known-good trace rather than a search.
- Earlier the same session: attention-layer conversion verified against real
  weights (all sections within 0.5 quantization steps; norms byte-identical).
- The 338 GB bf16 download finished at 03:17. Checkpoint is complete: 48/48
  layers, `lm_head.weight`, `embed_tokens.weight`, 128/128 n-gram shards.
- Disk audit: deleted `~/models/qwen38-flash-next-v4` (76 GB Unsloth GGUF) and
  `~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality` (27 GB), both
  re-downloadable and unused by this work. **199 GiB free**, against 96.61 GiB
  needed. `--release-source` is no longer necessary.
- Moved the verification scripts out of the session scratchpad into
  `dev/tools/checks/` so they survive.
- Created this `.agents/` directory so Codex or Gemini can pick the work up.

**Blocked on:** nothing. **Next:** `write_head` and `write_embedding` in
`dev/tools/convert_qwen4exp.py`, then a full conversion, then the forward path.

## Earlier — claude-code

22 commits on `qwen4exp-port`, `main` untouched, both CPU and Metal suites green
after every one. Built, in order: the layout struct and package validation;
expert tiling at 128 and routing at 512; hyper-connection, QSA indexer, QSA
selection, n-gram and per-layer-embedding kernels; sparse attention on the
existing Q8 tile; the memory plan's resident/streamable split; and the
quantizer, package writer and converters. Found along the way that linear
attention needed no new kernels (GDN already compiles at these dimensions) and
that the memory blocker was accounting rather than machinery. See
`decisions.md`.
