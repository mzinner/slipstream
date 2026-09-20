"""Verify a transcription of the PLE gate and convolution."""
import math, torch
import torch.nn.functional as F
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

torch.manual_seed(5)
HID, HC, SEQ, EMB = 128, 4, 10, 64
cfg = Qwen4ExpTextConfig(hidden_size=HID, hc_count=HC, ple_embed_dim=EMB,
                         ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=4,
                         ngram_vocab_size_base=1009,
                         make_ngram_vocab_size_divisible_by=8,
                         vocab_size=512, eos_token_id=2, seed=0, rms_norm_eps=1e-6)
ple = Qwen4ExpTextPLELayer(cfg, 0, 0).to(torch.float32).eval()
W = HC * HID
with torch.no_grad():
    for p in ple.parameters(): p.copy_(torch.randn_like(p) * 0.05)
    emb = torch.randn(1, SEQ, EMB) * 0.4
    hidden = torch.randn(1, SEQ, W) * 0.5

    # reference body, with the embedding supplied directly
    key_normed = ple.norm_key(ple.key_proj(emb)).unflatten(-1, (HC, HID))
    value = ple.value_proj(emb)
    query_normed = ple.norm_query(hidden).unflatten(-1, (HC, HID))
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(HID)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_normed = ple.norm_conv(gated.flatten(-2))
    gated = gated.flatten(-2)
    state = (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size
    x = gated_normed.transpose(1, 2)
    x = F.pad(x, (state, 0))
    want = gated + F.silu(ple.conv1d(x)).transpose(1, 2)

    # --- transcription, the way the kernels compute it ---
    def rms_per_stream(t, gain):
        s = t.reshape(1, SEQ, HC, HID)
        s = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)
        return s.flatten(-2) * (1.0 + gain)
    k = rms_per_stream(ple.key_proj(emb), ple.norm_key.weight).reshape(1, SEQ, HC, HID)
    q = rms_per_stream(hidden, ple.norm_query.weight).reshape(1, SEQ, HC, HID)
    g = (k * q).sum(-1) / math.sqrt(HID)
    g = torch.sign(g) * torch.sqrt(torch.clamp(g.abs(), min=1e-6))
    mine = (torch.sigmoid(g).unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)
    mine_normed = rms_per_stream(mine, ple.norm_conv.weight)
    padded = torch.cat([torch.zeros(1, state, W), mine_normed], dim=1)
    w = ple.conv1d.weight.squeeze(1)                       # [W, taps]
    conv = torch.zeros(1, SEQ, W)
    for tap in range(cfg.ple_conv_kernel_size):
        conv += padded[:, tap * cfg.ngram_size : tap * cfg.ngram_size + SEQ] * w[:, tap]
    mine = mine + F.silu(conv)

print("gate+conv max abs diff:", float((mine - want).abs().max()))
