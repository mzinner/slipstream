"""Verify a transcription of the n-gram id hashing against the real module."""
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

torch.manual_seed(11)
VOCAB, NGRAM, HEADS_PER, DIM, SEQ = 512, 3, 4, 64, 12
cfg = Qwen4ExpTextConfig(vocab_size=VOCAB, ngram_size=NGRAM, heads_per_ngram=HEADS_PER,
                         ngram_vocab_size_base=1009, make_ngram_vocab_size_divisible_by=8,
                         hidden_size=256, eos_token_id=2, seed=0)
mod = Qwen4ExpTextNGramEmbedding(cfg, DIM, layer_idx=0, ple_layer_index=0).eval()
heads = (NGRAM - 1) * HEADS_PER
with torch.no_grad():
    mod.ngram_embedding.weight.copy_(torch.randn_like(mod.ngram_embedding.weight))
    ids = torch.randint(0, VOCAB, (1, SEQ))
    want = mod(ids, None)                       # [1, SEQ, heads*head_dim]

    # --- transcription ---
    ctx = NGRAM - 1
    history = torch.cat([ids.new_full((1, ctx), cfg.eos_token_id), ids], dim=-1)
    shifted = [mod._shift_right_ignore_eos(history, s) for s in range(NGRAM)]
    mult, sizes, offs = mod.layer_multipliers, mod.ngram_heads_vocab_sizes, mod.ngram_heads_offsets
    blocks = []
    for n in range(2, NGRAM + 1):
        lo = (n - 2) * HEADS_PER
        mixed = shifted[0] * mult[0]
        for pos in range(1, n):
            mixed = torch.bitwise_xor(mixed, shifted[pos] * mult[pos])
        block = torch.remainder(mixed.unsqueeze(-1), sizes[lo:lo+HEADS_PER].view(1, 1, -1))
        blocks.append(block + offs[lo:lo+HEADS_PER].view(1, 1, -1))
    got_ids = torch.cat(blocks, dim=-1)[:, -SEQ:]
    got = mod.ngram_embedding(got_ids).flatten(-2)

print("head count            :", heads, " head dim:", DIM // heads if DIM % heads == 0 else DIM)
print("gathered rows match   :", bool(torch.equal(got, want)))
print("per-head vocab sizes  :", sizes.tolist()[:4], "...")
