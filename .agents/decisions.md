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
`dev/tools/checks/check_layer_composition.py` — re-run it if the forward path
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

The package format requires a DFlash 2 draft. Flash-Next has no such draft, only
an MTP head. A zero-filled placeholder is written so packages validate. Expect
no speculation speedup until a real draft is trained. Nitin deferred that.
