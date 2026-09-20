"""Does a whole layer, composed the way the kernels will be, match the module?

Every kernel has been checked against its own reference in isolation. This
checks the thing none of those can: that they compose in the right order,
with the residual carried the right way between them.
"""
import glob, json, os, struct, sys
import numpy as np
import torch

D = os.path.expanduser("~/models/qwen38-flash-next-bf16")
where = {}
for shard in sorted(glob.glob(D + "/*.safetensors")):
    try:
        f = open(shard, "rb"); n, = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    except Exception:
        continue
    h.pop("__metadata__", None)
    for k in h: where[k] = shard

def tensor(name):
    shard = where[name]
    f = open(shard, "rb"); n, = struct.unpack("<Q", f.read(8)); h = json.loads(f.read(n))
    e = h[name]; s, t = e["data_offsets"]; f.seek(8 + n + s)
    raw = np.frombuffer(f.read(t - s), dtype=np.uint16)
    return torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32)
                            .reshape(e["shape"]).copy())

HID, HC, LR, EPS = 2560, 4, 320, 1e-6
P = "model.language_model.layers.0."

def gated_residual(x, prefix, with_inject=True):
    """The transcription the three hyper-connection kernels implement."""
    xn = x.reshape(-1, HC, HID)
    xn = (xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + EPS)).flatten(-2)
    xn = xn * (1.0 + tensor(prefix + "hc_norm.weight"))
    low = torch.nn.functional.silu(
        xn @ tensor(prefix + "input_mix_weight_down.weight").T / HC)
    w = torch.sigmoid(low @ tensor(prefix + "input_mix_weight_up.weight").T)
    mixed = (w.reshape(-1, HC, HID) * xn.reshape(-1, HC, HID)).mean(1)
    if not with_inject:
        return mixed, None
    inj = 2 * torch.sigmoid(xn @ tensor(prefix + "block_inject_weight.weight").T / HC)
    return mixed, inj

def inject(x, y, weights):
    return x + (y.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)

torch.manual_seed(20260920)
rows = 6
x = (torch.randn(rows, HC * HID) * 0.6).to(torch.bfloat16).to(torch.float32)

with torch.no_grad():
    # --- the composition, as the engine will run it ---
    block_in, inj_a = gated_residual(x, P + "attn_hyper_connection.")
    # The mixer is exercised by its own test; here it stands for any block of
    # the right shape, because what is under test is the wiring around it.
    mixer_out = block_in @ torch.eye(HID)
    after_attention = inject(x, mixer_out, inj_a)

    block_in2, inj_b = gated_residual(after_attention, P + "mlp_hyper_connection.")
    mlp_out = block_in2 @ torch.eye(HID)
    composed = inject(after_attention, mlp_out, inj_b)

    # --- the same thing through the module ---
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextGatedResidual)
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpTextConfig)
    cfg = Qwen4ExpTextConfig(hidden_size=HID, hc_count=HC, hc_lowrank=LR,
                             rms_norm_eps=EPS)
    def module(prefix, use_combine=True):
        m = Qwen4ExpTextGatedResidual(cfg, use_combine=use_combine).to(torch.float32).eval()
        m.hc_norm.weight.copy_(tensor(prefix + "hc_norm.weight"))
        m.input_mix_weight_down.weight.copy_(tensor(prefix + "input_mix_weight_down.weight"))
        m.input_mix_weight_up.weight.copy_(tensor(prefix + "input_mix_weight_up.weight"))
        if use_combine:
            m.block_inject_weight.weight.copy_(tensor(prefix + "block_inject_weight.weight"))
        return m

    a = module(P + "attn_hyper_connection.")
    b = module(P + "mlp_hyper_connection.")
    mixed1, saved1, w1 = a(x)
    step1 = saved1 + (mixed1.unsqueeze(-2) * w1.unsqueeze(-1)).flatten(-2)
    mixed2, saved2, w2 = b(step1)
    want = saved2 + (mixed2.unsqueeze(-2) * w2.unsqueeze(-1)).flatten(-2)

print("weights          : layer 0, published checkpoint")
print(f"residual width   : {x.shape[1]} ({HC} streams of {HID})")
print(f"after attention  : max abs diff {float((after_attention - step1).abs().max()):.3e}")
print(f"whole layer      : max abs diff {float((composed - want).abs().max()):.3e}")
