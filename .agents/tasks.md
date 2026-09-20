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

## Left

- [ ] **`write_head`** in `convert_qwen4exp.py` — `lm_head.weight` is downloaded
- [ ] **`write_embedding`** — `model.language_model.embed_tokens.weight` is downloaded
- [ ] **Full conversion** of all 48 layers (needs disk freed first)
- [ ] **Load the real package** in the engine and confirm it validates
- [ ] **Forward path** — `Qwen4ExpTarget`, currently throws in `Runtime.mm`
- [ ] **End-to-end generation** + quality check against the llama.cpp fork

## Deferred by Nitin

- [ ] DFlash 2 draft training (speculative decoding). A zero-filled placeholder
      draft is written today because the format requires one. Expect no speedup
      from speculation until this exists.
- [ ] Expert streaming beyond the current demand-paging
