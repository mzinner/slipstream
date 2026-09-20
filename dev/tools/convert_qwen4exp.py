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
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.package_format import (  # noqa: E402
    ALIGNMENT,
    BF16,
    EXPERT_STORAGE_N,
    FINE_GROUP,
    GROUP,
    STORAGE_N,
    align,
    q4_bytes,
)

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
    arguments = parser.parse_args()

    if not arguments.dry_run and not arguments.source:
        parser.error("--source is required unless --dry-run is given")
    if not arguments.dry_run:
        parser.error(
            "converting real weights needs the checkpoint; only --dry-run is "
            "wired up so far"
        )

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


if __name__ == "__main__":
    raise SystemExit(main())
