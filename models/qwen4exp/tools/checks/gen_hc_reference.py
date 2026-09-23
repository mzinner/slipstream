"""Reference vectors for the qwen4exp gated residual, from the real module."""

import struct
import sys

import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

torch.manual_seed(20260919)
HIDDEN, HC, LOWRANK, ROWS = 2560, 4, 320, 8
cfg = Qwen4ExpTextConfig(
    hidden_size=HIDDEN, hc_count=HC, hc_lowrank=LOWRANK, rms_norm_eps=1e-6
)
mod = Qwen4ExpTextGatedResidual(cfg).to(torch.float32).eval()
H = HC * HIDDEN

with torch.no_grad():
    for p in mod.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
    mod.hc_norm.weight.copy_(1.0 + torch.randn(H) * 0.05)
    x = (torch.randn(ROWS, H) * 0.7).to(torch.bfloat16).to(torch.float32)
    mixed, saved, inj = mod(x)
    # the residual update the layer performs after the block runs
    y = (torch.randn(ROWS, HIDDEN) * 0.3).to(torch.bfloat16).to(torch.float32)
    updated = saved + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)


def bf16(t):
    u = t.contiguous().to(torch.float32).view(torch.int32)
    # round-to-nearest-even, matching Metal's bfloat conversion
    r = ((u >> 16) & 1) + 0x7FFF
    return (((u + r) >> 16) & 0xFFFF).to(torch.int32).numpy().astype("<u2").tobytes()


out = open(sys.argv[1], "wb")
out.write(b"HCREF001")
out.write(struct.pack("<4I", ROWS, HIDDEN, HC, LOWRANK))
for name, t in [
    ("x", x),
    ("gain", mod.hc_norm.weight),
    ("down", mod.input_mix_weight_down.weight),
    ("up", mod.input_mix_weight_up.weight),
    ("inject", mod.block_inject_weight.weight),
    ("y", y),
    ("mixed", mixed),
    ("inj", inj),
    ("updated", updated),
]:
    blob = bf16(t)
    out.write(struct.pack("<Q", len(blob)))
    out.write(blob)
    print(f"  {name:8} {tuple(t.shape)} -> {len(blob)} bytes")
out.close()
print(
    "eps",
    cfg.rms_norm_eps,
    "group",
    mod.hc_norm.group_size if hasattr(mod.hc_norm, "group_size") else "?",
)
