import glob
import json
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

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


HID, HC = 2560, 4
prompt_tokens = [760, 6511, 314, 9338, 369]

embed_w = tensor("model.language_model.embed_tokens.weight")
embeds = embed_w[prompt_tokens]
h0 = embeds.repeat(1, HC)

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
)

with open(os.path.join(D, "config.json")) as f:
    cfg_dict = json.load(f)
cfg = Qwen4ExpTextConfig(**cfg_dict["text_config"])

P = "model.language_model.layers.0."
attn_hc = Qwen4ExpTextGatedResidual(cfg, use_combine=True).eval()
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
    mixed, _, inj_w = attn_hc(h0)

gdn = Qwen4ExpTextGatedDeltaNet(cfg, 0).eval()
with torch.no_grad():
    for name, param in gdn.named_parameters():
        param.copy_(tensor(P + "linear_attn." + name))

with torch.no_grad():
    x = mixed.unsqueeze(0)  # [1, 5, 2560]
    qkv = gdn.in_proj_qkv(x)
    z = gdn.in_proj_z(x)
    b = gdn.in_proj_b(x)
    a = gdn.in_proj_a(x)

    print(
        "qkv shape",
        qkv.shape,
        "mean",
        qkv.mean().item(),
        "std",
        qkv.std().item(),
        "row4 first4",
        qkv[0, 4, :4].tolist(),
    )
    print(
        "z shape",
        z.shape,
        "mean",
        z.mean().item(),
        "std",
        z.std().item(),
        "row4 first4",
        z[0, 4, :4].tolist(),
    )
    print(
        "b shape",
        b.shape,
        "mean",
        b.mean().item(),
        "std",
        b.std().item(),
        "row4 first4",
        b[0, 4, :4].tolist(),
    )
    print(
        "a shape",
        a.shape,
        "mean",
        a.mean().item(),
        "std",
        a.std().item(),
        "row4 first4",
        a[0, 4, :4].tolist(),
    )

    # conv1d
    conv_out = F.conv1d(
        qkv.transpose(1, 2),
        weight=gdn.conv1d.weight,
        bias=None,
        padding=gdn.conv_kernel_size - 1,
        groups=gdn.conv_dim,
    )[:, :, :5].transpose(1, 2)
    conv_silu = F.silu(conv_out)
    print(
        "conv_silu shape",
        conv_silu.shape,
        "mean",
        conv_silu.mean().item(),
        "std",
        conv_silu.std().item(),
        "row4 first4",
        conv_silu[0, 4, :4].tolist(),
    )

    q, k, v = torch.split(conv_silu, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
    q = q.reshape(1, 5, -1, gdn.head_k_dim)
    k = k.reshape(1, 5, -1, gdn.head_k_dim)
    v = v.reshape(1, 5, -1, gdn.head_v_dim)

    # q, k rmsnorm
    # Note: transformers does l2norm with use_qk_l2norm_in_kernel=True
    # Let's check l2norm on q and k:
    # inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    # x * inv_norm
    q_norm = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
    k_norm = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
    print("q_norm row4 head0 first4:", q_norm[0, 4, 0, :4].tolist())
    print("k_norm row4 head0 first4:", k_norm[0, 4, 0, :4].tolist())
    print("v row4 head0 first4:", v[0, 4, 0, :4].tolist())

    beta = b.sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    decay = g.exp()
    print("beta row4 first4:", beta[0, 4, :4].tolist())
    print("decay row4 first4:", decay[0, 4, :4].tolist())

    # Also run torch_chunk_gated_delta_rule or torch_recurrent_gated_delta_rule to see core_attn_out
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        torch_recurrent_gated_delta_rule,
    )

    q_raw = q
    k_raw = k
    if gdn.num_v_heads // gdn.num_k_heads > 1:
        q_raw = q_raw.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)
        k_raw = k_raw.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)
    core_attn_out, _ = torch_recurrent_gated_delta_rule(
        q_raw, k_raw, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
    )
    print("core_attn_out row4 head0 first4:", core_attn_out[0, 4, 0, :4].tolist())
    print("core_attn_out row4 head0 norm:", core_attn_out[0, 4, 0].norm().item())

    # Now norm and gate
    z_h = z.reshape(1, 5, -1, gdn.head_v_dim)
    normed = gdn.norm(
        core_attn_out.reshape(-1, gdn.head_v_dim), z_h.reshape(-1, gdn.head_v_dim)
    )
    normed = normed.reshape(1, 5, -1, gdn.head_v_dim)
    print("normed (gdn_hidden) row4 head0 first4:", normed[0, 4, 0, :4].tolist())

    gdn_out = gdn.out_proj(normed.reshape(1, 5, -1))
    print(
        "full gdn_out row4 first4:",
        gdn_out[0, 4, :4].tolist(),
        "mean",
        gdn_out.mean().item(),
        "std",
        gdn_out.std().item(),
    )
