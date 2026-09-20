# Journal — qwen4exp port

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
