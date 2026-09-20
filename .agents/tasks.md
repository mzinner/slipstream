# Tasks — qwen4exp port

## Done

- [x] `Qwen4ExpLayout` — layout constants + arithmetic test
- [x] Package loading and validation (`splash-packed-q4-qwen4exp`)
- [x] Expert tiling at StorageN 128
- [x] MoE routing at 512 experts (`RouterWidth` template)
- [x] Hyper-connection kernels (normalize / mix / update)
- [x] QSA indexer scoring kernel
- [x] QSA block selection (exact top-k, deterministic)
- [x] Sparse attention (`Sparse` flag on the Q8 tile + page mask)
- [x] N-gram embedding gather kernel
- [x] Per-layer embedding kernels (gate + convolve)
- [x] Linear attention — reuses the existing GDN path, no new kernels
- [x] Memory plan: resident vs. streamable split
- [x] Affine quantizer (`dev/tools/quantize.py`)
- [x] Package writer (`dev/tools/package_format.py`)
- [x] 27B converter as a format proof (`dev/tools/convert_qwen38.py`)
- [x] qwen4exp converter: layers, experts, hyper-connections, PLE, draft
- [x] `--release-source` for the disk squeeze
- [x] Verified both layer types against real published weights
- [x] Verified a whole composed layer — exact, 0.000e+00
- [x] `write_head` and `write_embedding` in `convert_qwen4exp.py`
- [x] Placeholder vision generator for package manifest compliance
- [x] Converted all 48 layers to `/Users/nitin/models/qwen38-flash-next-splash` (96.61 GiB)
- [x] Kv2Group12 Metal attention kernels (prefill and verify/decode)
- [x] Draft attention shape acceptance for Qwen4Exp placeholder draft
- [x] Forward path (`Qwen4ExpTarget`) wired in `Runtime.mm`
- [x] Metal command chunking by working set to allow models larger than RAM
- [x] Verified end-to-end forward pass on GPU via `decode-profile` (prefill, B1 decode, B4 decode)

## Left

- [ ] End-to-end generation quality evaluation against reference


## Deferred by Nitin

- [ ] DFlash 2 draft training (speculative decoding). A zero-filled placeholder
      draft is written today because the format requires one. Expect no speedup
      from speculation until this exists.
- [ ] Expert streaming beyond the current demand-paging
