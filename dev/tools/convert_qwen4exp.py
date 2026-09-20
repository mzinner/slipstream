#!/usr/bin/env python3
"""Convert a Qwen3.8-Flash-Next checkpoint into a Splash package.

Unlike the 27B path this one quantizes: the published checkpoint is bf16, so
every projection is taken through the affine scheme in tools/quantize before
it is tiled. Three widths are in play and they are not interchangeable.

    most projections      groups of 64, tiled 256 wide
    experts and indexer   groups of 64, tiled 128 wide - their outputs are
                          640 and 640, neither a multiple of 256
    per-layer embedding   groups of 32, row-major - its rows are 160 wide,
                          which is not a whole number of 64-element groups

With --dry-run the weights are not read at all: each file is written with its
header and sized by the layout, leaving the body a hole. That costs a few
kibibytes for a package whose apparent size is ninety-seven gibibytes, and it
is enough for the engine to load, validate and plan, which is what proves the
section arithmetic here agrees with the reader.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.package_format import (  # noqa: E402
    ALIGNMENT,
    BF16,
    EXPERT_STORAGE_N,
    FINE_GROUP,
    GROUP,
    STORAGE_N,
    StreamingWeightFile,
    WeightFile,
    align,
    pad_rows,
    q4_bytes,
    tile_q4,
)
from tools.quantize import from_bf16, quantize_affine  # noqa: E402


class Checkpoint:
    """Reads tensors by name from a directory of safetensors shards.

    The index is not required: every shard's header names what it holds, so a
    partially downloaded checkpoint can still be converted as far as it goes.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._where: dict[str, str] = {}
        self._open: dict[str, tuple] = {}
        for shard in sorted(glob.glob(str(self.root / "*.safetensors"))):
            try:
                with open(shard, "rb") as handle:
                    (length,) = struct.unpack("<Q", handle.read(8))
                    header = json.loads(handle.read(length))
            except (OSError, ValueError, json.JSONDecodeError):
                continue  # still being written
            header.pop("__metadata__", None)
            for name in header:
                self._where[name] = shard

    def has(self, name: str) -> bool:
        return name in self._where

    def shards_for(self, names) -> set[str]:
        return {self._where[name] for name in names if name in self._where}

    def release(self, keep: set[str]) -> int:
        """Delete shards no remaining tensor needs, returning bytes freed.

        The converted package is about a third the size of the checkpoint it
        came from, so releasing a shard once nothing else reads it frees
        space faster than the output fills it. Without this the two cannot
        both be on disk: 335 GiB of bf16 and a 96.61 GiB package need more
        room than converting them one after the other ever does.
        """
        freed = 0
        for shard in sorted(set(self._where.values()) - keep):
            handle = self._open.pop(shard, None)
            if handle:
                handle[0].close()
            try:
                freed += os.path.getsize(shard)
                os.remove(shard)
            except OSError:
                continue
        self._where = {
            name: shard for name, shard in self._where.items() if shard in keep
        }
        return freed

    def tensor(self, name: str) -> np.ndarray:
        shard = self._where[name]
        if shard not in self._open:
            handle = open(shard, "rb")
            (length,) = struct.unpack("<Q", handle.read(8))
            self._open[shard] = (handle, json.loads(handle.read(length)), 8 + length)
        handle, header, base = self._open[shard]
        entry = header[name]
        start, end = entry["data_offsets"]
        handle.seek(base + start)
        payload = handle.read(end - start)
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name} is {entry['dtype']}, expected BF16")
        values = from_bf16(np.frombuffer(payload, dtype=np.uint16))
        return values.reshape(entry["shape"])

    def raw(self, name: str) -> bytes:
        """The stored bytes, for tensors the package carries as bf16."""
        shard = self._where[name]
        if shard not in self._open:
            handle = open(shard, "rb")
            (length,) = struct.unpack("<Q", handle.read(8))
            self._open[shard] = (handle, json.loads(handle.read(length)), 8 + length)
        handle, header, base = self._open[shard]
        start, end = header[name]["data_offsets"]
        handle.seek(base + start)
        return handle.read(end - start)


def quantized_tile(values, storage_n=STORAGE_N, group=GROUP, pad_to=None):
    codes, scales, biases = quantize_affine(values, group=group)
    if pad_to is not None:
        codes, scales, biases = pad_rows(codes, scales, biases, pad_to)
    return tile_q4(codes, scales, biases, storage_n=storage_n, group=group)


LAYER_MAGIC = b"MDFN0001"
HEAD_MAGIC = b"MDFN0002"
EMBEDDING_MAGIC = b"MDFN0003"
NGRAM_MAGIC = b"MDFN0004"
DRAFT_MAGIC = b"MDFD0004"
VISION_MAGIC = b"MDFV0001"

# Qwen4ExpLayout, from runtime/model/Qwen4Exp.hpp.
LAYOUT = {
    "layers": 48,
    "hidden": 2560,
    "vocabulary": 248320,
    "packed_gdn": 16640,
    "packed_full": 13312,
    "convolution": 10240,
    "value_heads": 48,
    "head_dimension": 128,
    "attention_width": 6144,
    "attention_head_dimension": 256,
    "full_attention_period": 4,
    "experts": 512,
    "expert_intermediate": 640,
    "hyper_count": 4,
    "hyper_low_rank": 320,
    "indexer_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dimension": 128,
    "ngram_vocabulary": 320_001_536,
    "ngram_heads": 16,
    "ngram_head_dimension": 160,
    "ngram_size": 3,
    "ple_taps": 4,
    "ple_layer": 1,
    "ngram_shards": 128,
}


def hyper_width() -> int:
    return LAYOUT["hidden"] * LAYOUT["hyper_count"]


def indexer_width() -> int:
    return (LAYOUT["indexer_heads"] + LAYOUT["indexer_kv_heads"]) * LAYOUT[
        "indexer_head_dimension"
    ]


def q8_bytes(out_size: int, in_size: int) -> int:
    """A byte per weight, plus a bf16 scale and bias per group of 64."""
    elements = out_size * in_size
    return elements + 2 * (elements // GROUP) * BF16


def hyper_sections(with_inject: bool) -> list[tuple[int, str]]:
    width = hyper_width() * BF16
    mix = LAYOUT["hyper_low_rank"] * width
    out = [(width, "hyper-norm"), (mix, "hyper-mix-down"), (mix, "hyper-mix-up")]
    # The final mixer collapses the streams before the head and never writes a
    # block result back, so it carries no injection weight.
    if with_inject:
        out.append((LAYOUT["hyper_count"] * width, "hyper-inject"))
    return out


def expert_sections() -> list[tuple[int, str]]:
    hidden = LAYOUT["hidden"]
    inter = LAYOUT["expert_intermediate"]
    experts = LAYOUT["experts"]
    return [
        (q8_bytes(experts, hidden), "router"),
        (experts * q4_bytes(inter, hidden), "experts-gate"),
        (experts * q4_bytes(inter, hidden), "experts-up"),
        (experts * q4_bytes(hidden, inter), "experts-down"),
        (q4_bytes(inter, hidden), "shared-gate"),
        (q4_bytes(inter, hidden), "shared-up"),
        (q4_bytes(hidden, inter), "shared-down"),
        (q8_bytes(STORAGE_N, hidden), "shared-scalar-gate"),
    ]


def layer_sections(index: int) -> tuple[int, list[tuple[int, str]]]:
    hidden = LAYOUT["hidden"]
    full = (index + 1) % LAYOUT["full_attention_period"] == 0
    entries = hyper_sections(True)
    if full:
        entries += [
            (q4_bytes(LAYOUT["packed_full"], hidden), "attention-input"),
            (LAYOUT["attention_head_dimension"] * BF16, "query-norm"),
            (LAYOUT["attention_head_dimension"] * BF16, "key-norm"),
            (q4_bytes(hidden, LAYOUT["attention_width"]), "attention-output"),
            (q4_bytes(indexer_width(), hidden), "indexer-qk"),
            (LAYOUT["indexer_head_dimension"] * BF16, "indexer-query-norm"),
            (LAYOUT["indexer_head_dimension"] * BF16, "indexer-key-norm"),
        ]
    else:
        entries += [
            (q4_bytes(LAYOUT["packed_gdn"], hidden), "gdn-input"),
            (LAYOUT["convolution"] * 4 * BF16, "gdn-convolution"),
            (LAYOUT["value_heads"] * 4, "gdn-decay"),
            (LAYOUT["value_heads"] * BF16, "gdn-time-bias"),
            (LAYOUT["head_dimension"] * BF16, "gdn-norm"),
            (q4_bytes(hidden, LAYOUT["attention_width"]), "gdn-output"),
        ]
    entries += hyper_sections(True) + expert_sections()
    return (1 if full else 0), entries


def head_sections() -> list[tuple[int, str]]:
    return (
        hyper_sections(False)
        + [(LAYOUT["hidden"] * BF16, "final-norm")]
        + [(q4_bytes(LAYOUT["vocabulary"], LAYOUT["hidden"]), "logits")]
    )


def embedding_sections() -> list[tuple[int, str]]:
    elements = LAYOUT["vocabulary"] * LAYOUT["hidden"]
    parameters = elements // GROUP * BF16
    return [
        (elements // 2, "embedding-weights"),
        (parameters, "embedding-scales"),
        (parameters, "embedding-biases"),
    ]


def ngram_sections() -> list[tuple[int, str]]:
    elements = LAYOUT["ngram_vocabulary"] * LAYOUT["ngram_head_dimension"]
    parameters = elements // FINE_GROUP * BF16
    heads = LAYOUT["ngram_heads"] * 8
    width = hyper_width() * BF16
    return [
        (elements // 2, "ngram-weights"),
        (parameters, "ngram-scales"),
        (parameters, "ngram-biases"),
        (heads, "ngram-head-offsets"),
        (heads, "ngram-head-vocabulary-sizes"),
        (LAYOUT["ngram_size"] * 8, "ngram-layer-multipliers"),
        (q4_bytes(hyper_width(), LAYOUT["hidden"]), "ple-key"),
        (q4_bytes(LAYOUT["hidden"], LAYOUT["hidden"]), "ple-value"),
        (width, "ple-key-norm"),
        (width, "ple-query-norm"),
        (width, "ple-conv-norm"),
        (width * LAYOUT["ple_taps"], "ple-convolution"),
    ]


def write_hyper(packed, source: Checkpoint, prefix: str, with_inject: bool):
    """Order matches readHyperConnection: norm, mix down, mix up, inject."""
    packed.section(source.raw(prefix + ".hc_norm.weight"))
    packed.section(source.raw(prefix + ".input_mix_weight_down.weight"))
    packed.section(source.raw(prefix + ".input_mix_weight_up.weight"))
    if with_inject:
        packed.section(source.raw(prefix + ".block_inject_weight.weight"))


def write_attention(packed, source: Checkpoint, prefix: str) -> None:
    """Order matches readQwenMixer then readIndexer.

    The packed projection is query, key, value, and the query is twice the
    head width because this model gates its attention output - the same
    doubling the 27B has, which is why packed_full is 13312 and not 7168.
    """
    parts = [
        source.tensor(prefix + ".q_proj.weight"),
        source.tensor(prefix + ".k_proj.weight"),
        source.tensor(prefix + ".v_proj.weight"),
    ]
    packed.section(quantized_tile(np.vstack(parts), pad_to=LAYOUT["packed_full"]))
    packed.section(source.raw(prefix + ".q_norm.weight"))
    packed.section(source.raw(prefix + ".k_norm.weight"))
    packed.section(quantized_tile(source.tensor(prefix + ".o_proj.weight")))
    # The indexer tiles 128 wide: its projection is 640 out, not a multiple
    # of 256.
    packed.section(
        quantized_tile(
            source.tensor(prefix + ".indexer.index_qk_proj.weight"),
            storage_n=EXPERT_STORAGE_N,
        )
    )
    packed.section(source.raw(prefix + ".indexer.q_layernorm.weight"))
    packed.section(source.raw(prefix + ".indexer.k_layernorm.weight"))


def write_experts(packed, source: Checkpoint, prefix: str) -> None:
    """Expert slabs, 128 wide, with gate and up split out of one tensor.

    The checkpoint fuses them as [experts, 2 * intermediate, hidden] and the
    reference chunks the product in two, so the first half is the gate and
    the second the up. They are separate slabs here because the operator
    reads them as separate projections.
    """
    experts = LAYOUT["experts"]
    inter = LAYOUT["expert_intermediate"]
    narrow = EXPERT_STORAGE_N

    router = source.tensor(prefix + ".gate.weight")
    packed.section(quantized_q8(router))

    fused = source.tensor(prefix + ".experts.gate_up_proj")
    for half in (slice(0, inter), slice(inter, 2 * inter)):
        packed.section(
            b"".join(
                quantized_tile(fused[index][half], storage_n=narrow)
                for index in range(experts)
            )
        )
    del fused
    down = source.tensor(prefix + ".experts.down_proj")
    packed.section(
        b"".join(
            quantized_tile(down[index], storage_n=narrow) for index in range(experts)
        )
    )
    del down

    for name in ("gate_proj", "up_proj", "down_proj"):
        packed.section(
            quantized_tile(
                source.tensor(f"{prefix}.shared_expert.{name}.weight"),
                storage_n=narrow,
            )
        )
    gate = source.tensor(prefix + ".shared_expert_gate.weight")
    padded = np.zeros((STORAGE_N, gate.shape[1]), dtype=np.float32)
    padded[0] = gate[0]
    packed.section(quantized_q8(padded))


def quantized_q8(values) -> bytes:
    """Eight-bit affine, in the runs readQ8Projection expects."""
    out, inp = values.shape
    groups = inp // GROUP
    blocks = np.ascontiguousarray(values, dtype=np.float32).reshape(out, groups, GROUP)
    low = blocks.min(axis=2)
    high = blocks.max(axis=2)
    anchor_low = np.abs(low) > np.abs(high)
    bias = np.where(anchor_low, low, high)
    other = np.where(anchor_low, high, low)
    scale = (other - bias) / 255.0
    from tools.quantize import to_bf16

    scale_bits, bias_bits = to_bf16(scale), to_bf16(bias)
    stored_scale = from_bf16(scale_bits)[..., None]
    stored_bias = from_bf16(bias_bits)[..., None]
    safe = np.where(stored_scale == 0, 1.0, stored_scale)
    codes = np.clip(np.rint((blocks - stored_bias) / safe), 0, 255).astype(np.uint8)

    # [quant group][row] for the parameters, as the router kernel indexes them.
    def parameters(values_):
        return values_.T.reshape(-1).tobytes()

    return (
        codes.reshape(out, inp).tobytes()
        + parameters(scale_bits)
        + parameters(bias_bits)
    )


def write_layer(source: Checkpoint, index: int, destination: Path) -> int:
    prefix = f"model.language_model.layers.{index}"
    kind, _ = layer_sections(index)
    packed = WeightFile(destination / f"layer-{index}.bin", LAYER_MAGIC, index, kind)
    write_hyper(packed, source, prefix + ".attn_hyper_connection", True)

    if kind == 1:
        write_attention(packed, source, prefix + ".self_attn")
        write_hyper(packed, source, prefix + ".mlp_hyper_connection", True)
        write_experts(packed, source, prefix + ".mlp")
        return packed.finish()

    linear = prefix + ".linear_attn"
    parts = [
        source.tensor(linear + ".in_proj_qkv.weight"),
        source.tensor(linear + ".in_proj_z.weight"),
        source.tensor(linear + ".in_proj_b.weight"),
        source.tensor(linear + ".in_proj_a.weight"),
    ]
    packed.section(quantized_tile(np.vstack(parts), pad_to=LAYOUT["packed_gdn"]))
    packed.section(source.raw(linear + ".conv1d.weight"))
    logarithm = source.tensor(linear + ".A_log")
    packed.section((-np.exp(logarithm)).astype("<f4").tobytes())
    packed.section(source.raw(linear + ".dt_bias"))
    packed.section(source.raw(linear + ".norm.weight"))
    packed.section(quantized_tile(source.tensor(linear + ".out_proj.weight")))

    write_hyper(packed, source, prefix + ".mlp_hyper_connection", True)
    write_experts(packed, source, prefix + ".mlp")
    return packed.finish()


def write_per_layer_embedding(source: Checkpoint, destination: Path) -> int:
    """The per-layer embedding, streamed because its table is 29.8 GiB.

    The table is 128 shards of identical shape whose rows total exactly the
    padded vocabulary, so it is a concatenation in shard order - the shards
    are uniform slices of one table, not one chunk per head. The per-head
    vocabulary sizes and offsets travel with it as hashing metadata; they do
    not decide the order.

    Its rows are 160 wide, which is not a whole number of 64-element groups,
    so it quantizes in groups of 32 and stays row-major: it is gathered a row
    at a time, never multiplied as a tile.
    """
    prefix = f"model.language_model.layers.{LAYOUT['ple_layer']}.ple"
    embedding = prefix + ".ple_embedding"
    shards = LAYOUT["ngram_shards"] if "ngram_shards" in LAYOUT else 128
    names = [f"{embedding}.ngram_embedding.shard_{i}.weight" for i in range(shards)]
    missing = [name for name in names if not source.has(name)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {shards} table shards are not downloaded yet"
        )

    packed = StreamingWeightFile(
        destination / "ngram.bin",
        NGRAM_MAGIC,
        shards,
        LAYOUT["ngram_head_dimension"],
    )
    # Weights, then scales, then biases: three runs over the whole table, so
    # each shard is quantized once and its three parts held until the run they
    # belong to is being written.
    parts = []
    for name in names:
        codes, scales, biases = quantize_affine(source.tensor(name), group=FINE_GROUP)
        flat = codes.reshape(-1)
        parts.append(
            (
                (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8).tobytes(),
                scales.tobytes(),
                biases.tobytes(),
            )
        )
    for run in range(3):
        packed.begin()
        for part in parts:
            packed.write(part[run])
    del parts

    for name in ("ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        packed.section(source.raw(f"{embedding}.{name}"))
    packed.section(source.raw(f"{embedding}.layer_multipliers"))
    packed.section(quantized_tile(source.tensor(prefix + ".key_proj.weight")))
    packed.section(quantized_tile(source.tensor(prefix + ".value_proj.weight")))
    for name in ("norm_key", "norm_query", "norm_conv"):
        packed.section(source.raw(f"{prefix}.{name}.weight"))
    packed.section(source.raw(prefix + ".conv1d.weight"))
    return packed.finish()


def write_placeholder_draft(destination: Path) -> int:
    """A draft of the right shape and no content.

    The package format requires a DFlash 2 draft and the loader reads one
    unconditionally, but this model has none: it ships an MTP head, which is
    a different architecture that these files cannot describe. Until a draft
    is trained for this target, the package carries zeros - enough for the
    engine to load, validate and plan, and useless for proposing tokens.
    Nothing can reach it in the meantime, because execution is refused a step
    earlier.
    """
    layer, model, _ = draft_sections()
    total = 0
    for index in range(5):
        total += sized_file(
            destination / f"layer-{index}.bin", DRAFT_MAGIC, index, 0, layer
        )
    total += sized_file(destination / "model.bin", DRAFT_MAGIC, 5, 1, model)
    return total


def sized_file(path: Path, magic: bytes, layer: int, kind: int, entries) -> int:
    """Header and a hole: the file is the size the layout implies, no body."""
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = 16
    for size, _ in entries:
        offset = align(offset) + size
    total = align(offset)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<8sII", magic, layer, kind))
        os.ftruncate(handle.fileno(), total)
    return total


def draft_sections() -> list[tuple[int, str]]:
    # Provisional, matching qwen4expDraftLayout in ModelDescriptor.mm.
    draft = {
        "layers": 5,
        "hidden": 2560,
        "dynamic": 768,
        "qkv": 3072,
        "attention": 2048,
        "intermediate": 8704,
        "head_dimension": 128,
        "selector_rank": 256,
        "vocabulary": LAYOUT["vocabulary"],
    }
    hidden = draft["hidden"] * BF16
    conv = 4 * draft["hidden"] * BF16
    head_norm = draft["head_dimension"] * BF16
    layer = [
        (hidden, "input-norm"),
        (conv, "attention-convolution"),
        (q4_bytes(draft["dynamic"], draft["hidden"]), "attention-dynamic"),
        (q4_bytes(draft["qkv"], draft["hidden"]), "qkv"),
        (head_norm, "query-norm"),
        (head_norm, "key-norm"),
        (q4_bytes(draft["hidden"], draft["attention"]), "attention-output"),
        (hidden, "post-attention-norm"),
        (conv, "mlp-convolution"),
        (q4_bytes(draft["dynamic"], draft["hidden"]), "mlp-dynamic"),
        (q4_bytes(draft["intermediate"], draft["hidden"]), "mlp-gate"),
        (q4_bytes(draft["intermediate"], draft["hidden"]), "mlp-up"),
        (q4_bytes(draft["hidden"], draft["intermediate"]), "mlp-down"),
    ]
    codebook = draft["vocabulary"] * draft["selector_rank"] * BF16
    model = [
        (q4_bytes(draft["hidden"], LAYOUT["hidden"] * 5), "context-projection"),
        (hidden, "hidden-norm"),
        (hidden, "final-norm"),
        (q4_bytes(draft["selector_rank"], draft["hidden"]), "selector"),
        (codebook, "predecessor-codebook"),
        (codebook, "successor-codebook"),
    ]
    return layer, model, draft


def write_manifest(root: Path) -> None:
    layer_types = [
        "attention" if (index + 1) % LAYOUT["full_attention_period"] == 0 else "gdn"
        for index in range(LAYOUT["layers"])
    ]
    manifest = {
        "model": "Qwen3.8-Flash-Next",
        "schema_version": 5,
        "format": {
            "name": "splash-packed-q4-qwen4exp",
            "q4_bits": 4,
            "q8_bits": 8,
            "quant_group_size": GROUP,
            "storage_n": STORAGE_N,
            "expert_storage_n": EXPERT_STORAGE_N,
            "section_alignment_bytes": ALIGNMENT,
            "target_layer_magic": LAYER_MAGIC.decode(),
            "draft_layer_magic": DRAFT_MAGIC.decode(),
            "vision_magic": VISION_MAGIC.decode(),
        },
        # These mirror ExecutionLimits and the KV constants; the loader
        # checks every one of them against the engine it is loading into,
        # which is how the first guess here was caught.
        "execution_geometry": {
            "allocation_extent_target_bytes": 134217728,
            "draft_proposal_tokens": 7,
            "draft_query_rows": 8,
            "draft_sliding_window": 2048,
            "maximum_batch_width": 4,
            "prefill_token_budget": 2048,
            "target_kv_block_tokens": 32,
            "target_verify_rows": 8,
        },
        "target": {
            "architecture": "qwen4exp",
            "layers": LAYOUT["layers"],
            "hidden_size": LAYOUT["hidden"],
            "vocabulary_size": LAYOUT["vocabulary"],
            "gdn_actual_width": LAYOUT["convolution"]
            + LAYOUT["attention_width"]
            + 2 * LAYOUT["value_heads"],
            "gdn_packed_width": LAYOUT["packed_gdn"],
            "attention_packed_width": LAYOUT["packed_full"],
            "experts": LAYOUT["experts"],
            "experts_per_token": 10,
            "moe_intermediate_size": LAYOUT["expert_intermediate"],
            "shared_expert_intermediate_size": LAYOUT["expert_intermediate"],
            "hyper_connection_count": LAYOUT["hyper_count"],
            "hyper_connection_low_rank": LAYOUT["hyper_low_rank"],
            "indexer_heads": LAYOUT["indexer_heads"],
            "indexer_kv_heads": LAYOUT["indexer_kv_heads"],
            "indexer_head_dim": LAYOUT["indexer_head_dimension"],
            "ngram_layer": LAYOUT["ple_layer"],
            "ngram_vocabulary_size": LAYOUT["ngram_vocabulary"],
            "ngram_embedding_size": LAYOUT["hidden"],
            "layer_types": layer_types,
        },
        "draft": {
            "architecture": "DFlash2DraftModel",
            "layers": 5,
            "hidden_size": 2560,
            "intermediate_size": 8704,
            "sliding_window": 2048,
            "block_size": 8,
            "dynamic_conv_group_size": 16,
            "dynamic_conv_kernel_size": 2,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_capture_layers": [3, 13, 23, 33, 43],
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tokenizer = root / "tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "hidden_size": LAYOUT["hidden"],
                    "vocab_size": LAYOUT["vocabulary"],
                    "max_position_embeddings": 262144,
                }
            }
        )
    )


def dry_run(root: Path) -> int:
    target = root / "target"
    total = 0
    for index in range(LAYOUT["layers"]):
        kind, entries = layer_sections(index)
        total += sized_file(
            target / f"layer-{index}.bin", LAYER_MAGIC, index, kind, entries
        )
    total += sized_file(
        target / "head.bin", HEAD_MAGIC, LAYOUT["layers"], 2, head_sections()
    )
    total += sized_file(
        target / "embedding.bin",
        EMBEDDING_MAGIC,
        LAYOUT["vocabulary"],
        LAYOUT["hidden"],
        embedding_sections(),
    )
    total += sized_file(
        target / "ngram.bin",
        NGRAM_MAGIC,
        128,
        LAYOUT["ngram_head_dimension"],
        ngram_sections(),
    )
    layer, model, _ = draft_sections()
    for index in range(5):
        sized_file(root / "draft" / f"layer-{index}.bin", DRAFT_MAGIC, index, 0, layer)
    sized_file(root / "draft" / "model.bin", DRAFT_MAGIC, 5, 1, model)
    write_manifest(root)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--release-source",
        action="store_true",
        help="delete each shard once no remaining layer reads it; needed when "
        "the checkpoint and the package will not both fit",
    )
    arguments = parser.parse_args()

    if not arguments.dry_run and not arguments.source:
        parser.error("--source is required unless --dry-run is given")
    if arguments.dry_run:
        total = dry_run(arguments.destination)
        on_disk = sum(
            path.stat().st_blocks * 512
            for path in arguments.destination.rglob("*")
            if path.is_file()
        )
        print(f"target weights  {total / 2**30:.2f} GiB apparent")
        print(f"on disk         {on_disk / 1024:.0f} KiB (headers only)")
        print(f"package at      {arguments.destination}")
        return 0

    source = Checkpoint(arguments.source)
    if arguments.release_source:
        # Deleting shards out from under a running download would lose data
        # and confuse the fetcher, so refuse unless the checkpoint is whole.
        index = arguments.source / "model.safetensors.index.json"
        if not index.exists():
            parser.error(
                "--release-source needs a complete checkpoint; its index is "
                "not downloaded yet"
            )
        wanted = set(json.loads(index.read_text())["weight_map"].values())
        present = {path.name for path in arguments.source.glob("*.safetensors")}
        if wanted - present:
            parser.error(
                f"--release-source needs a complete checkpoint; "
                f"{len(wanted - present)} of {len(wanted)} shards are missing"
            )
        if any(arguments.source.glob("*.incomplete")):
            parser.error("a download is still in progress in that directory")

    target = arguments.destination / "target"
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    for index in range(LAYOUT["layers"]):
        try:
            written = write_layer(source, index, target)
        except KeyError as missing:
            print(f"  layer-{index}.bin needs {missing}, not downloaded yet")
            return 1
        total += written
        note = ""
        if arguments.release_source:
            still_needed = set()
            for later in range(index + 1, LAYOUT["layers"]):
                still_needed |= source.shards_for(
                    [
                        name
                        for name in source._where
                        if f".layers.{later}." in name
                        or ".ple." in name
                        or ".layers." not in name
                    ]
                )
            freed = source.release(still_needed)
            if freed:
                note = f"  (released {freed / 2**30:.1f} GiB of source)"
        print(f"  layer-{index}.bin {written / 2**30:.2f} GiB{note}")
    total += write_per_layer_embedding(source, target)
    draft = write_placeholder_draft(arguments.destination / "draft")
    print(f"\ntarget weights  {total / 2**30:.2f} GiB")
    print(f"draft           {draft / 2**30:.2f} GiB of zeros - this model has")
    print("                no DFlash 2 draft, only an MTP head")
    print("head and embedding are not wired up yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
