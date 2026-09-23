"""Stage 1: does Splash's packed data equal the MLX source bytes?

Answers the two open questions: nibble order, and the pad convention where an
output dimension rounds up to StorageN=256.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from splashpack import QWEN38, qwen38_layer_sections, read_sections

L = QWEN38
SRC = Path(sys.argv[1])  # mlx checkpoint
PKG = Path(sys.argv[2])  # installed splash package

index = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
_open = {}


def tensor(name):
    """Return (raw bytes, shape, dtype) for one safetensors tensor."""
    shard = index[name]
    if shard not in _open:
        f = open(SRC / shard, "rb")
        (n,) = struct.unpack("<Q", f.read(8))
        _open[shard] = (f, json.loads(f.read(n)), 8 + n)
    f, hdr, base = _open[shard]
    e = hdr[name]
    s, t = e["data_offsets"]
    f.seek(base + s)
    return f.read(t - s), e["shape"], e["dtype"]


P = "language_model.model.layers.0."
type_, sizes = qwen38_layer_sections(L, 0)
magic, layer, ftype, sec = read_sections(PKG / "target/layer-0.bin", sizes)
names = [
    "input-norm",
    "gdn-input",
    "conv",
    "decay",
    "time-bias",
    "mixer-norm",
    "gdn-output",
    "post-norm",
    "mlp-gate",
    "mlp-up",
    "mlp-down",
]
S = dict(zip(names, sec))


def verdict(label, a, b, extra=""):
    mark = "MATCH" if a == b else "differ"
    print(f"  {label:<26} {mark:>6}   {len(a)} vs {len(b)} bytes {extra}")
    return a == b


print("layer-0.bin  header:", magic, layer, ftype, f"({len(sizes)} sections)\n")
print("bf16 vectors (no packing question):")
verdict("input_layernorm", S["input-norm"], tensor(P + "input_layernorm.weight")[0])
verdict(
    "post_attention_layernorm",
    S["post-norm"],
    tensor(P + "post_attention_layernorm.weight")[0],
)
verdict("linear_attn.norm", S["mixer-norm"], tensor(P + "linear_attn.norm.weight")[0])
verdict("conv1d.weight", S["conv"], tensor(P + "linear_attn.conv1d.weight")[0])
verdict("dt_bias", S["time-bias"], tensor(P + "linear_attn.dt_bias")[0])

print("\ndecay (Splash stores f32; source A_log is bf16):")
a_raw, a_shape, a_dt = tensor(P + "linear_attn.A_log")
src = np.frombuffer(a_raw, dtype=np.uint16).astype(np.uint32) << 16
print(
    f"  A_log {a_shape} {a_dt} -> widened bf16==f32 : "
    f"{'MATCH' if S['decay'] == src.astype('<u4').tobytes() else 'differ'}"
)

print("\nQ4 projections (weights | scales | biases as three runs):")
for label, src_name, out, inp in [
    ("mlp.gate_proj", P + "mlp.gate_proj", L["intermediateSize"], L["hiddenSize"]),
    ("mlp.down_proj", P + "mlp.down_proj", L["hiddenSize"], L["intermediateSize"]),
    (
        "linear_attn.out_proj",
        P + "linear_attn.out_proj",
        L["hiddenSize"],
        L["attentionWidth"],
    ),
]:
    key = {
        "mlp.gate_proj": "mlp-gate",
        "mlp.down_proj": "mlp-down",
        "linear_attn.out_proj": "gdn-output",
    }[label]
    blob = S[key]
    e = out * inp
    w, sc, bi = (
        blob[: e // 2],
        blob[e // 2 : e // 2 + e // 32],
        blob[e // 2 + e // 32 :],
    )
    sw, shape, dt = tensor(src_name + ".weight")
    ss, _, _ = tensor(src_name + ".scales")
    sb, _, _ = tensor(src_name + ".biases")
    print(f"  [{label}]  source weight shape {shape} {dt}")
    verdict("    weights run", w, sw)
    verdict("    scales run", sc, ss)
    verdict("    biases run", bi, sb)

print("\ngdn-input: concat order and StorageN padding")
parts = [
    ("in_proj_qkv", L["convolutionDimension"]),
    ("in_proj_z", L["attentionWidth"]),
    ("in_proj_a", 48),
    ("in_proj_b", 48),
]
actual = sum(p[1] for p in parts)
print(
    f"  actual width {actual}  padded width {L['packedGdnWidth']}  "
    f"pad rows {L['packedGdnWidth'] - actual}"
)
e = L["packedGdnWidth"] * L["hiddenSize"]
blob = S["gdn-input"]
w = blob[: e // 2]
cat = b"".join(tensor(P + "linear_attn." + n + ".weight")[0] for n, _ in parts)
print(f"  concatenated source weights: {len(cat)} bytes vs packed run {len(w)}")
if w[: len(cat)] == cat:
    tail = w[len(cat) :]
    print(f"  prefix MATCH; {len(tail)} pad bytes, all zero: {not any(tail)}")
else:
    first = next((i for i in range(min(len(w), len(cat))) if w[i] != cat[i]), None)
    print(f"  prefix differs, first at byte {first}")
