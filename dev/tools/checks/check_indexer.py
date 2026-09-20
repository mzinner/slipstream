"""Verify a transcription of the QSA indexer against the real module."""
import math, torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextQSAIndexer, apply_rotary_pos_emb)
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

torch.manual_seed(7)
HID, NH, KVH, HD, BUDGET, RATIO, SEQ = 256, 4, 1, 32, 16, 4, 24
# rotary_dim must equal the indexer head dim: head_dim * partial_rotary_factor
cfg = Qwen4ExpTextConfig(hidden_size=HID, num_attention_heads=4,
                         num_key_value_heads=1, head_dim=2*HD,
                         indexer_n_heads=NH, indexer_kv_heads=KVH,
                         indexer_head_dim=HD, indexer_budget=BUDGET,
                         indexer_compress_ratio=RATIO, rms_norm_eps=1e-6,
                         rope_parameters={"rope_type": "default",
                                          "rope_theta": 10000.0,
                                          "partial_rotary_factor": 0.5})
ix = Qwen4ExpTextQSAIndexer(cfg, 0).to(torch.float32).eval()
with torch.no_grad():
    for p in ix.parameters(): p.copy_(torch.randn_like(p) * 0.05)

x = torch.randn(1, SEQ, HID) * 0.5
cos = torch.randn(1, SEQ, HD) * 0.3 + 1.0
sin = torch.randn(1, SEQ, HD) * 0.3
causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))[None, None]

with torch.no_grad():
    want = ix(x, (cos, sin), causal, None)

    # --- transcription ---
    qk = ix.index_qk_proj(x)
    q, token_k = torch.split(qk, [NH*HD, KVH*HD], dim=-1)
    q = ix.q_layernorm(q.reshape(1, SEQ, NH, HD))
    q = apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)
    raw_keys = token_k.reshape(1, SEQ, HD)
    block_topk = BUDGET // RATIO

    got = torch.zeros_like(want)
    for query in range(SEQ):
        visible = torch.nonzero(causal[0, 0, query]).flatten()
        blocks = visible.shape[-1] // RATIO
        chosen = []
        if blocks > 0:
            idx = visible[: blocks*RATIO].view(blocks, RATIO)
            pooled = raw_keys[0].index_select(0, idx.flatten()).view(blocks, RATIO, HD)
            pooled = ix.k_layernorm(pooled.float().mean(dim=1))
            bk = apply_rotary_pos_emb(pooled.unsqueeze(1),
                                      cos=cos[0].index_select(0, idx[:, 0]),
                                      sin=sin[0].index_select(0, idx[:, 0])).squeeze(1)
            scores = torch.matmul(q[0, query].float(), bk.float().transpose(-1, -2)).transpose(-1, -2)
            scores = torch.relu(scores).sum(dim=-1) / math.sqrt(HD)
            pick = scores.topk(min(block_topk, blocks), dim=0).indices
            chosen.append(idx.index_select(0, pick).flatten())
        chosen.append(visible[blocks*RATIO:])
        sel = torch.cat(chosen).to(torch.int64) if chosen else torch.tensor([], dtype=torch.int64)
        got[0, 0, query, sel] = True

print("mask exact match:", bool(torch.equal(got, want)),
      " differing entries:", int((got != want).sum()))
