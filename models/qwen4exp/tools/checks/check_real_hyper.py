"""Does the transcription hold on the model's real weights, not random ones?"""

import glob
import json
import os
import struct

import numpy as np
import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

D = os.path.expanduser("~/models/qwen38-flash-next-bf16")
where = {}
for shard in sorted(glob.glob(D + "/*.safetensors")):
    try:
        f = open(shard, "rb")
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    except Exception:
        continue
    h.pop("__metadata__", None)
    for k in h:
        where[k] = shard


def tensor(name):
    shard = where[name]
    f = open(shard, "rb")
    (n,) = struct.unpack("<Q", f.read(8))
    h = json.loads(f.read(n))
    e = h[name]
    s, t = e["data_offsets"]
    f.seek(8 + n + s)
    raw = np.frombuffer(f.read(t - s), dtype=np.uint16)
    return torch.from_numpy(
        (raw.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"]).copy()
    )


P = "model.language_model.layers.0.attn_hyper_connection."
HID, HC, LR = 2560, 4, 320
cfg = Qwen4ExpTextConfig(hidden_size=HID, hc_count=HC, hc_lowrank=LR, rms_norm_eps=1e-6)
mod = Qwen4ExpTextGatedResidual(cfg).to(torch.float32).eval()
with torch.no_grad():
    mod.hc_norm.weight.copy_(tensor(P + "hc_norm.weight"))
    mod.input_mix_weight_down.weight.copy_(tensor(P + "input_mix_weight_down.weight"))
    mod.input_mix_weight_up.weight.copy_(tensor(P + "input_mix_weight_up.weight"))
    mod.block_inject_weight.weight.copy_(tensor(P + "block_inject_weight.weight"))

    torch.manual_seed(1)
    x = (torch.randn(8, HC * HID) * 0.7).to(torch.bfloat16).to(torch.float32)
    mixed, saved, inj = mod(x)

    # the transcription the kernels implement
    xn = x.reshape(8, HC, HID)
    xn = (xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + 1e-6)).flatten(-2)
    xn = xn * (1.0 + mod.hc_norm.weight)
    low = torch.nn.functional.silu(xn @ mod.input_mix_weight_down.weight.T / HC)
    w = torch.sigmoid(low @ mod.input_mix_weight_up.weight.T)
    mine = (w.reshape(8, HC, HID) * xn.reshape(8, HC, HID)).mean(1)
    inj2 = 2 * torch.sigmoid(xn @ mod.block_inject_weight.weight.T / HC)

print("weights           : layer 0, from the published checkpoint")
print(
    f"gain range        : {float(mod.hc_norm.weight.min()):+.4f} .. "
    f"{float(mod.hc_norm.weight.max()):+.4f}"
)
print(f"mixed max abs diff: {float((mine - mixed).abs().max()):.3e}")
print(f"inject max abs dif: {float((inj2 - inj).abs().max()):.3e}")
