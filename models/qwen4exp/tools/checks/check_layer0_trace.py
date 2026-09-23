import glob
import json
import os
import struct

import numpy as np
import torch

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

_open_files = {}


def tensor(name):
    shard = where[name]
    if shard not in _open_files:
        f = open(shard, "rb")
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
        _open_files[shard] = (f, h, 8 + n)
    f, h, base = _open_files[shard]
    e = h[name]
    s, t = e["data_offsets"]
    f.seek(base + s)
    raw = np.frombuffer(f.read(t - s), dtype=np.uint16)
    return torch.from_numpy(
        (raw.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"]).copy()
    )


HID, HC, LR, EPS = 2560, 4, 320, 1e-6
prompt_tokens = [760, 6511, 314, 9338, 369]

print("Loading embedding...")
embed_w = tensor("model.language_model.embed_tokens.weight")
embeds = embed_w[prompt_tokens]  # [5, 2560]
print(
    f"Embeds shape {embeds.shape} mean={embeds.mean():.6f} std={embeds.std():.6f} first4={embeds[4, :4].tolist()}"
)

# Broadcast to 4 streams:
h0 = embeds.repeat(1, HC)  # [5, 10240]
print(
    f"H0 shape {h0.shape} mean={h0.mean():.6f} std={h0.std():.6f} row4_first4={h0[4, :4].tolist()}"
)

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextSparseMoeBlock,
)

with open(os.path.join(D, "config.json")) as f:
    cfg_dict = json.load(f)
cfg = Qwen4ExpTextConfig(**cfg_dict["text_config"])

print("\nRunning attn_hyper_connection...")
attn_hc = Qwen4ExpTextGatedResidual(cfg, use_combine=True).eval()
P = "model.language_model.layers.0."
with torch.no_grad():
    attn_hc.hc_norm.weight.copy_(tensor(P + "attn_hyper_connection.hc_norm.weight"))
    attn_hc.input_mix_weight_down.weight.copy_(
        tensor(P + "attn_hyper_connection.input_mix_weight_down.weight")
    )
    attn_hc.input_mix_weight_up.weight.copy_(
        tensor(P + "attn_hyper_connection.input_mix_weight_up.weight")
    )
    attn_hc.block_inject_weight.weight.copy_(
        tensor(P + "attn_hyper_connection.block_inject_weight.weight")
    )

    mixed, saved, inj_w = attn_hc(h0)
print(
    f"Attn HC mixed shape {mixed.shape} mean={mixed.mean():.6f} std={mixed.std():.6f} row4_first4={mixed[4, :4].tolist()}"
)
print(
    f"Attn HC inj_w shape {inj_w.shape} mean={inj_w.mean():.6f} std={inj_w.std():.6f} row4={inj_w[4].tolist()}"
)

print("\nRunning GDN linear_attn...")
gdn = Qwen4ExpTextGatedDeltaNet(cfg, 0).eval()
with torch.no_grad():
    for name, param in gdn.named_parameters():
        full_name = P + "linear_attn." + name
        if full_name in where:
            param.copy_(tensor(full_name))
        else:
            print(f"MISSING {full_name}")

    gdn_out = gdn(mixed.unsqueeze(0)).squeeze(0)  # [5, 2560]
print(
    f"GDN out shape {gdn_out.shape} mean={gdn_out.mean():.6f} std={gdn_out.std():.6f} row4_first4={gdn_out[4, :4].tolist()}"
)

# Attention injection
inj_attn = (gdn_out.unsqueeze(-2) * inj_w.unsqueeze(-1)).flatten(-2)
h_after_attn = h0 + inj_attn
print(
    f"After attn residual shape {h_after_attn.shape} mean={h_after_attn.mean():.6f} std={h_after_attn.std():.6f} row4_first4={h_after_attn[4, :4].tolist()}"
)

print("\nRunning mlp_hyper_connection...")
mlp_hc = Qwen4ExpTextGatedResidual(cfg, use_combine=True).eval()
with torch.no_grad():
    mlp_hc.hc_norm.weight.copy_(tensor(P + "mlp_hyper_connection.hc_norm.weight"))
    mlp_hc.input_mix_weight_down.weight.copy_(
        tensor(P + "mlp_hyper_connection.input_mix_weight_down.weight")
    )
    mlp_hc.input_mix_weight_up.weight.copy_(
        tensor(P + "mlp_hyper_connection.input_mix_weight_up.weight")
    )
    mlp_hc.block_inject_weight.weight.copy_(
        tensor(P + "mlp_hyper_connection.block_inject_weight.weight")
    )

    mlp_mixed, _, mlp_inj_w = mlp_hc(h_after_attn)
print(
    f"MLP HC mixed shape {mlp_mixed.shape} mean={mlp_mixed.mean():.6f} std={mlp_mixed.std():.6f} row4_first4={mlp_mixed[4, :4].tolist()}"
)
print(
    f"MLP HC inj_w shape {mlp_inj_w.shape} mean={mlp_inj_w.mean():.6f} std={mlp_inj_w.std():.6f} row4={mlp_inj_w[4].tolist()}"
)

print("\nRunning MoE mlp...")
moe = Qwen4ExpTextSparseMoeBlock(cfg).eval()
with torch.no_grad():
    for name, param in moe.named_parameters():
        full_name = P + "mlp." + name
        if full_name in where:
            param.copy_(tensor(full_name))
        else:
            print(f"MoE MISSING {full_name}")

    moe_out = moe(mlp_mixed.unsqueeze(0)).squeeze(0)
print(
    f"MoE out shape {moe_out.shape} mean={moe_out.mean():.6f} std={moe_out.std():.6f} row4_first4={moe_out[4, :4].tolist()}"
)

inj_moe = (moe_out.unsqueeze(-2) * mlp_inj_w.unsqueeze(-1)).flatten(-2)
h_layer0_out = h_after_attn + inj_moe
print(
    f"\nFinal Layer 0 out shape {h_layer0_out.shape} mean={h_layer0_out.mean():.6f} std={h_layer0_out.std():.6f} row4_first4={h_layer0_out[4, :4].tolist()}"
)

print("\nRunning final hyper_connection_mixer and lm_head...")
final_mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False).eval()
P_mix = "model.language_model.hyper_connection_mixer."
with torch.no_grad():
    final_mixer.hc_norm.weight.copy_(tensor(P_mix + "hc_norm.weight"))
    final_mixer.input_mix_weight_down.weight.copy_(
        tensor(P_mix + "input_mix_weight_down.weight")
    )
    final_mixer.input_mix_weight_up.weight.copy_(
        tensor(P_mix + "input_mix_weight_up.weight")
    )

    fh_h0 = final_mixer(h0)
    lm_head_w = tensor("lm_head.weight")
    lg_h0 = fh_h0 @ lm_head_w.T
    top_h0 = torch.topk(lg_h0[4], 5)
    print(
        f"h0 finalHidden row4 mean={fh_h0[4].mean():.6f} std={fh_h0[4].std():.6f} first4={fh_h0[4, :4].tolist()}"
    )
    print("h0 top 5 logits at row 4:")
    for v, t in zip(top_h0.values.tolist(), top_h0.indices.tolist()):
        print(f"  token {t} = {v:.6f}")

    fh_l0 = final_mixer(h_layer0_out)
    lg_l0 = fh_l0 @ lm_head_w.T
    top_l0 = torch.topk(lg_l0[4], 5)
    print(
        f"\nLayer 0 finalHidden row4 mean={fh_l0[4].mean():.6f} std={fh_l0[4].std():.6f} first4={fh_l0[4, :4].tolist()}"
    )
    print("Layer 0 top 5 logits at row 4:")
    for v, t in zip(top_l0.values.tolist(), top_l0.indices.tolist()):
        print(f"  token {t} = {v:.6f}")
