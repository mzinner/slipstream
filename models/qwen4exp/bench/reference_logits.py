#!/usr/bin/env python3
"""Full-precision reference for qwen4exp, one layer at a time.

This is the ground truth the quantized engines are scored against: the
published bf16 checkpoint run through the transformers reference code. The
checkpoint is 338 GB, far more than memory, so the model is never built
whole. Each decoder layer is constructed on its own, loaded from the
checkpoint with strict=True (a missing or misnamed weight is an error, not a
silent zero), run over the whole passage, and freed.

The n-gram table is 102 GB and never loaded. Its rows are gathered straight
out of the checkpoint files for just the ids the passage hashes to.

Compute is float32 on the CPU: slower than bf16, and more accurate than the
reference would be in its own dtype, which is what a ground truth should be.

Output, per passage: the logits for every position, as bf16, [tokens x vocab].
"""

import argparse
import gc
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as M
from torch import nn
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

CHECKPOINT = Path.home() / "models/qwen38-flash-next-bf16"
PREFIX = "model.language_model."


class Checkpoint:
    """Tensor lookup across the shards by reading their headers only."""

    def __init__(self, root: Path):
        self.root = root
        self.where = {}
        for shard in sorted(root.glob("*.safetensors")):
            with open(shard, "rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
            header.pop("__metadata__", None)
            for name, entry in header.items():
                self.where[name] = (shard, 8 + length, entry)

    def memmap(self, name):
        shard, base, entry = self.where[name]
        assert entry["dtype"] == "BF16", (name, entry["dtype"])
        start, end = entry["data_offsets"]
        raw = np.memmap(
            shard,
            dtype=np.uint16,
            mode="r",
            offset=base + start,
            shape=((end - start) // 2,),
        )
        return raw.reshape(entry["shape"])

    def tensor(self, name, dtype=torch.float32):
        shard, base, entry = self.where[name]
        start, end = entry["data_offsets"]
        with open(shard, "rb") as handle:
            handle.seek(base + start)
            data = handle.read(end - start)
        kind = {
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "I64": torch.int64,
            "F16": torch.float16,
        }[entry["dtype"]]
        value = torch.frombuffer(bytearray(data), dtype=kind).reshape(entry["shape"])
        return value.to(dtype) if value.is_floating_point() else value

    def state(self, prefix, dtype=torch.float32, skip=".ngram_embedding.shard_"):
        # The n-gram shards are skipped here, not after: together they are
        # 102 GB, and reading them before filtering is what memory cannot hold.
        return {
            name[len(prefix) :]: self.tensor(name, dtype)
            for name in self.where
            if name.startswith(prefix) and skip not in name
        }


class ShardedRows(nn.Module):
    """Stands in for the n-gram nn.Embedding: gathers rows from disk."""

    def __init__(self, checkpoint, prefix, shards):
        super().__init__()
        self.tables = [
            checkpoint.memmap(f"{prefix}shard_{i}.weight") for i in range(shards)
        ]
        self.rows = self.tables[0].shape[0]
        # The reference reads .weight.device to place its ids; a CPU stand-in.
        self.weight = torch.empty(0)

    def forward(self, ids):
        flat = ids.reshape(-1).numpy()
        out = np.empty((flat.size, self.tables[0].shape[1]), dtype=np.uint16)
        shard, row = np.divmod(flat, self.rows)
        for s in np.unique(shard):
            pick = shard == s
            out[pick] = self.tables[s][row[pick]]
        values = torch.from_numpy(out.astype(np.uint32) << 16).view(torch.float32)
        return values.reshape(*ids.shape, -1)


# Weight groups --quant can round to an engine format, to measure what each
# quantization choice costs against this reference. Names are relative to a
# layer ("head" and "embed" are the model's own).
QUANT_GROUPS = {
    "experts": ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
    "router": ("mlp.gate.weight",),
    "shared": (
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
    ),
    "attn": (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ),
    "gdn_in": ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight"),
    "gdn_ab": ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"),
    "gdn_out": ("linear_attn.out_proj.weight",),
    "ple": ("ple.key_proj.weight", "ple.value_proj.weight"),
    "hyper": (
        "attn_hyper_connection.input_mix_weight_down.weight",
        "attn_hyper_connection.input_mix_weight_up.weight",
        "mlp_hyper_connection.input_mix_weight_down.weight",
        "mlp_hyper_connection.input_mix_weight_up.weight",
    ),
}


def fake_quant(weight, bits, group=64, search=False):
    """Round to the converter's affine format (tools/quantize.py) and back:
    per group of `group` inputs, a bf16 bias at the larger-magnitude end and
    a bf16 scale, codes 0..2^bits-1. search=True instead picks, per group, the
    range (min/max pulled in by up to 30%) with the least squared error -
    same storage, smarter rounding."""
    shape = weight.shape
    rows = weight.reshape(-1, shape[-1])
    out = torch.empty_like(rows)
    levels = (1 << bits) - 1
    for begin in range(0, rows.shape[0], 8192):
        blocks = rows[begin : begin + 8192].reshape(-1, shape[-1] // group, group)
        low = blocks.amin(dim=2, keepdim=True)
        high = blocks.amax(dim=2, keepdim=True)
        best, best_error = None, None
        for shrink_low in (1.0, 0.9, 0.8, 0.7) if search else (1.0,):
            for shrink_high in (1.0, 0.9, 0.8, 0.7) if search else (1.0,):
                lo, hi = low * shrink_low, high * shrink_high
                if search:
                    bias, other = lo, hi  # plain min-based affine
                else:
                    anchor_low = lo.abs() > hi.abs()
                    bias = torch.where(anchor_low, lo, hi)
                    other = torch.where(anchor_low, hi, lo)
                scale = ((other - bias) / levels).bfloat16().float()
                bias = bias.bfloat16().float()
                safe = torch.where(scale == 0, torch.ones_like(scale), scale)
                codes = torch.clamp(torch.round((blocks - bias) / safe), 0, levels)
                rebuilt = codes * scale + bias
                if not search:
                    best = rebuilt
                    break
                error = ((rebuilt - blocks) ** 2).sum(dim=2, keepdim=True)
                if best is None:
                    best, best_error = rebuilt, error
                else:
                    better = error < best_error
                    best = torch.where(better, rebuilt, best)
                    best_error = torch.where(better, error, best_error)
        out[begin : begin + 8192] = best.reshape(-1, shape[-1])
    return out.reshape(shape)


def parse_quant(spec):
    """ "experts:4,router:8" -> {"experts": (4, 64, False), ...}. A group may
    add its group size and "mse": experts:4:32, experts:4:64:mse."""
    result = {}
    for item in filter(None, (spec or "").split(",")):
        name, bits, *rest = item.split(":")
        if name not in QUANT_GROUPS and name not in ("head", "embed"):
            raise SystemExit(f"unknown quant group {name}")
        group = int(rest[0]) if rest else None
        result[name] = (int(bits), group, "mse" in rest)
    return result


@torch.no_grad()
def run(config, checkpoint, ids, out_path, log, no_ple=False, quant=None):
    quant = quant or {}
    tokens = torch.tensor([ids])
    T = tokens.shape[1]
    embed = checkpoint.tensor(PREFIX + "embed_tokens.weight")
    if "embed" in quant:
        bits, group, search = quant["embed"]
        embed = fake_quant(embed, bits, group=group or 32, search=search)
    hidden = embed[tokens]
    del embed

    position_ids = torch.arange(T).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids, position_ids = position_ids[0], position_ids[1:]
    mask_kwargs = dict(
        config=config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=text_position_ids,
        allow_is_causal_skip=False,
    )
    causal = M.create_causal_mask(**mask_kwargs)
    conv_mask = M.create_recurrent_attention_mask(**mask_kwargs)
    ple_ids = tokens
    if conv_mask is not None:
        eos = (
            config.eos_token_id[0]
            if isinstance(config.eos_token_id, list)
            else config.eos_token_id
        )
        ple_ids = torch.where(conv_mask.bool(), ple_ids, eos)
    rotary = M.Qwen4ExpTextRotaryEmbedding(config=config)
    position_embeddings = rotary(hidden, position_ids)
    hidden = hidden.repeat(1, 1, config.hc_count)

    for index in range(config.num_hidden_layers):
        started = time.time()
        with torch.device("meta"):
            layer = M.Qwen4ExpTextDecoderLayer(config, index)
        prefix = f"{PREFIX}layers.{index}."
        state = checkpoint.state(prefix)
        for group, (bits, size, search) in quant.items():
            for name in QUANT_GROUPS.get(group, ()):
                if name in state:
                    state[name] = fake_quant(
                        state[name], bits, group=size or 64, search=search
                    )
        if layer.ple is not None:
            # The table is swapped for a disk gather; everything else loads.
            embedding = layer.ple.ple_embedding
            embedding.ngram_embedding = ShardedRows(
                checkpoint, prefix + "ple.ple_embedding.ngram_embedding.", 128
            )
            state = {
                k: v for k, v in state.items() if ".ngram_embedding.shard_" not in k
            }
            # The hash multipliers are stored; they must also match what the
            # reference derives from the seed, or the gather is meaningless.
            derived = M._build_layer_multipliers(
                config.vocab_size,
                config.ngram_size,
                embedding.ple_layer_index,
                config.seed,
            )
            if not torch.equal(state["ple.ple_embedding.layer_multipliers"], derived):
                raise RuntimeError("stored n-gram multipliers disagree with the seed")
        missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
        missing = [m for m in missing if ".ngram_embedding." not in m]
        if missing or unexpected:
            raise RuntimeError(
                f"layer {index}: missing {missing[:5]} unexpected {unexpected[:5]}"
            )
        layer = layer.float().eval()
        if no_ple:
            # Diagnostic only: what the model does without its n-gram input.
            layer.ple = None
        hidden = layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal,
            conv_mask=conv_mask,
            past_key_values=None,
            ple_input_ids=ple_ids,
        )
        del layer, state
        gc.collect()
        log(f"  layer {index:2d}  {time.time() - started:5.1f}s")

    with torch.device("meta"):
        mixer = M.Qwen4ExpTextGatedResidual(config, use_combine=False)
    mixer.load_state_dict(
        checkpoint.state(PREFIX + "hyper_connection_mixer."), strict=True, assign=True
    )
    hidden = mixer.float()(hidden)[0]
    head = checkpoint.tensor("lm_head.weight")
    if "head" in quant:
        bits, size, search = quant["head"]
        head = fake_quant(head, bits, group=size or 64, search=search)
    with open(out_path, "wb") as out:
        for begin in range(0, T, 256):
            logits = hidden[begin : begin + 256] @ head.T
            bits = logits.contiguous().view(torch.int32).numpy().astype(np.uint32)
            # Round to nearest bf16, as the engine's buffer holds.
            rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            rounded.tofile(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", required=True, help="json: {name: {ids: [...]}}")
    parser.add_argument("--only")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--quant",
        help="round weight groups to the engine's "
        "format first, e.g. experts:4,attn:8 (groups: "
        + ", ".join(list(QUANT_GROUPS) + ["head", "embed"])
        + ")",
    )
    parser.add_argument("--tag", help="output name suffix (default: reference)")
    parser.add_argument(
        "--no-ple",
        action="store_true",
        help="diagnostic: drop the per-layer n-gram embedding",
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    config = Qwen4ExpTextConfig(
        **json.loads((CHECKPOINT / "config.json").read_text())["text_config"]
    )
    config._attn_implementation = "sdpa"
    checkpoint = Checkpoint(CHECKPOINT)
    passages = json.loads(Path(args.passages).read_text())
    names = args.only.split(",") if args.only else list(passages)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        ids = passages[name]["ids"]
        print(f"{name}: {len(ids)} tokens", flush=True)
        started = time.time()
        suffix = args.tag or ("no-ple" if args.no_ple else "reference")
        run(
            config,
            checkpoint,
            ids,
            out_dir / f"{name}.{suffix}.bin",
            lambda text: print(text, flush=True),
            no_ple=args.no_ple,
            quant=parse_quant(args.quant),
        )
        print(f"{name}: done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
