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
from torch import nn

import transformers.models.qwen4_exp.modeling_qwen4_exp as M
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
        raw = np.memmap(shard, dtype=np.uint16, mode="r", offset=base + start,
                        shape=((end - start) // 2,))
        return raw.reshape(entry["shape"])

    def tensor(self, name, dtype=torch.float32):
        shard, base, entry = self.where[name]
        start, end = entry["data_offsets"]
        with open(shard, "rb") as handle:
            handle.seek(base + start)
            data = handle.read(end - start)
        kind = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64,
                "F16": torch.float16}[entry["dtype"]]
        value = torch.frombuffer(bytearray(data), dtype=kind).reshape(entry["shape"])
        return value.to(dtype) if value.is_floating_point() else value

    def state(self, prefix, dtype=torch.float32, skip=".ngram_embedding.shard_"):
        # The n-gram shards are skipped here, not after: together they are
        # 102 GB, and reading them before filtering is what memory cannot hold.
        return {name[len(prefix):]: self.tensor(name, dtype)
                for name in self.where if name.startswith(prefix) and skip not in name}


class ShardedRows(nn.Module):
    """Stands in for the n-gram nn.Embedding: gathers rows from disk."""

    def __init__(self, checkpoint, prefix, shards):
        super().__init__()
        self.tables = [checkpoint.memmap(f"{prefix}shard_{i}.weight") for i in range(shards)]
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


@torch.no_grad()
def run(config, checkpoint, ids, out_path, log, no_ple=False):
    tokens = torch.tensor([ids])
    T = tokens.shape[1]
    embed = checkpoint.tensor(PREFIX + "embed_tokens.weight")
    hidden = embed[tokens]
    del embed

    position_ids = torch.arange(T).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids, position_ids = position_ids[0], position_ids[1:]
    mask_kwargs = dict(config=config, inputs_embeds=hidden, attention_mask=None,
                       past_key_values=None, position_ids=text_position_ids,
                       allow_is_causal_skip=False)
    causal = M.create_causal_mask(**mask_kwargs)
    conv_mask = M.create_recurrent_attention_mask(**mask_kwargs)
    ple_ids = tokens
    if conv_mask is not None:
        eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
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
        if layer.ple is not None:
            # The table is swapped for a disk gather; everything else loads.
            embedding = layer.ple.ple_embedding
            embedding.ngram_embedding = ShardedRows(
                checkpoint, prefix + "ple.ple_embedding.ngram_embedding.", 128)
            state = {k: v for k, v in state.items() if ".ngram_embedding.shard_" not in k}
            # The hash multipliers are stored; they must also match what the
            # reference derives from the seed, or the gather is meaningless.
            derived = M._build_layer_multipliers(
                config.vocab_size, config.ngram_size, embedding.ple_layer_index, config.seed)
            if not torch.equal(state["ple.ple_embedding.layer_multipliers"], derived):
                raise RuntimeError("stored n-gram multipliers disagree with the seed")
        missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
        missing = [m for m in missing if ".ngram_embedding." not in m]
        if missing or unexpected:
            raise RuntimeError(f"layer {index}: missing {missing[:5]} unexpected {unexpected[:5]}")
        layer = layer.float().eval()
        if no_ple:
            # Diagnostic only: what the model does without its n-gram input.
            layer.ple = None
        hidden = layer(hidden, position_embeddings=position_embeddings,
                       attention_mask=causal, conv_mask=conv_mask,
                       past_key_values=None, ple_input_ids=ple_ids)
        del layer, state
        gc.collect()
        log(f"  layer {index:2d}  {time.time() - started:5.1f}s")

    with torch.device("meta"):
        mixer = M.Qwen4ExpTextGatedResidual(config, use_combine=False)
    mixer.load_state_dict(checkpoint.state(PREFIX + "hyper_connection_mixer."), strict=True, assign=True)
    hidden = mixer.float()(hidden)[0]
    head = checkpoint.tensor("lm_head.weight")
    with open(out_path, "wb") as out:
        for begin in range(0, T, 256):
            logits = hidden[begin:begin + 256] @ head.T
            bits = logits.contiguous().view(torch.int32).numpy().astype(np.uint32)
            # Round to nearest bf16, as the engine's buffer holds.
            rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            rounded.tofile(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", required=True, help="json: {name: {ids: [...]}}")
    parser.add_argument("--only")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-ple", action="store_true",
                        help="diagnostic: drop the per-layer n-gram embedding")
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    config = Qwen4ExpTextConfig(**json.loads((CHECKPOINT / "config.json").read_text())["text_config"])
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
        suffix = "no-ple" if args.no_ple else "reference"
        run(config, checkpoint, ids, out_dir / f"{name}.{suffix}.bin",
            lambda text: print(text, flush=True), no_ple=args.no_ple)
        print(f"{name}: done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
