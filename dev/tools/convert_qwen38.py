#!/usr/bin/env python3
"""Convert an mlx-community Qwen3.8-27B checkpoint into a Splash package.

This path is a repack, not a requantization: that checkpoint already carries
affine 4-bit groups of 64, which is what the package format holds, so the
weights are reordered rather than recomputed. Run it against the published
package to check the reordering:

    dev/tools/convert_qwen38.py --source <mlx checkpoint> \\
        --verify install/models/incoai/Qwen3.8-27B-Splash --layers 0,1,2,3

Every section is expected to match byte for byte except the GDN decay, which
is stored as -exp(A_log) and differs from the published package by one unit
in the last place on about a third of its values - the original converter
computed that exponential with a different implementation, most likely on the
GPU. It is 192 bytes per linear-attention layer and a relative difference of
about 1e-7; the tool reports it rather than hiding it.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.package_format import (  # noqa: E402
    BF16,
    WeightFile,
    pad_rows,
    plain_q4,
    q4_bytes,
    read_sections,
    tile_q4,
)

LAYER_MAGIC = b"MDFL0006"
HEAD_MAGIC = b"MDFL0002"
EMBEDDING_MAGIC = b"MDFE0001"

# Qwen3_8Layout, from runtime/model/Qwen3_8.hpp.
LAYOUT = {
    "layers": 64,
    "hidden": 5120,
    "vocabulary": 248320,
    "packed_gdn": 16640,
    "packed_full": 14336,
    "convolution": 10240,
    "value_heads": 48,
    "head_dimension": 128,
    "attention_width": 6144,
    "intermediate": 17408,
    "attention_head_dimension": 256,
    "full_attention_period": 4,
}


class Checkpoint:
    def __init__(self, root: Path):
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        self.map = json.loads(index.read_text())["weight_map"]
        self._open: dict[str, tuple] = {}

    def raw(self, name: str):
        shard = self.map[name]
        if shard not in self._open:
            handle = open(self.root / shard, "rb")
            (length,) = struct.unpack("<Q", handle.read(8))
            self._open[shard] = (handle, json.loads(handle.read(length)), 8 + length)
        handle, header, base = self._open[shard]
        entry = header[name]
        start, end = entry["data_offsets"]
        handle.seek(base + start)
        return handle.read(end - start), entry["shape"]

    def bf16(self, name: str) -> bytes:
        return self.raw(name)[0]

    def q4(self, name: str):
        packed, shape = self.raw(name + ".weight")
        out, inp = shape[0], shape[1] * 8
        words = np.frombuffer(packed, dtype="<u4").reshape(out, inp // 8)
        codes = np.empty((out, inp), dtype=np.uint8)
        for nibble in range(8):
            codes[:, nibble::8] = (words >> (4 * nibble)) & 0xF
        groups = inp // 64
        scales = np.frombuffer(self.raw(name + ".scales")[0], dtype=np.uint16)
        biases = np.frombuffer(self.raw(name + ".biases")[0], dtype=np.uint16)
        return codes, scales.reshape(out, groups), biases.reshape(out, groups)


def layer_sections(index: int) -> tuple[int, list[int], list[str]]:
    hidden = LAYOUT["hidden"] * BF16
    full = (index + 1) % LAYOUT["full_attention_period"] == 0
    if full:
        mixer = [
            (q4_bytes(LAYOUT["packed_full"], LAYOUT["hidden"]), "attention-input"),
            (LAYOUT["attention_head_dimension"] * BF16, "query-norm"),
            (LAYOUT["attention_head_dimension"] * BF16, "key-norm"),
            (q4_bytes(LAYOUT["hidden"], LAYOUT["attention_width"]), "attention-output"),
        ]
    else:
        mixer = [
            (q4_bytes(LAYOUT["packed_gdn"], LAYOUT["hidden"]), "gdn-input"),
            (LAYOUT["convolution"] * 4 * BF16, "gdn-convolution"),
            (LAYOUT["value_heads"] * 4, "gdn-decay"),
            (LAYOUT["value_heads"] * BF16, "gdn-time-bias"),
            (LAYOUT["head_dimension"] * BF16, "gdn-norm"),
            (q4_bytes(LAYOUT["hidden"], LAYOUT["attention_width"]), "gdn-output"),
        ]
    tail = [
        (q4_bytes(LAYOUT["intermediate"], LAYOUT["hidden"]), "mlp-gate"),
        (q4_bytes(LAYOUT["intermediate"], LAYOUT["hidden"]), "mlp-up"),
        (q4_bytes(LAYOUT["hidden"], LAYOUT["intermediate"]), "mlp-down"),
    ]
    entries = (
        [(hidden, "input-norm")] + mixer + [(hidden, "post-attention-norm")] + tail
    )
    return (
        (1 if full else 0),
        [size for size, _ in entries],
        [label for _, label in entries],
    )


def write_layer(source: Checkpoint, index: int, destination: Path) -> Path:
    prefix = f"language_model.model.layers.{index}."
    kind, _, _ = layer_sections(index)
    path = destination / f"layer-{index}.bin"
    packed = WeightFile(path, LAYER_MAGIC, index, kind)
    packed.section(source.bf16(prefix + "input_layernorm.weight"))

    if kind == 1:
        parts = [
            source.q4(prefix + "self_attn.q_proj"),
            source.q4(prefix + "self_attn.k_proj"),
            source.q4(prefix + "self_attn.v_proj"),
        ]
        packed.section(tile_q4(*concatenate(parts, LAYOUT["packed_full"])))
        packed.section(source.bf16(prefix + "self_attn.q_norm.weight"))
        packed.section(source.bf16(prefix + "self_attn.k_norm.weight"))
        packed.section(tile_q4(*source.q4(prefix + "self_attn.o_proj")))
    else:
        # The concatenation is qkv, z, b, a - b before a, which is not the
        # order the names suggest and was found by mapping rows back to their
        # source tensor rather than by reading them.
        parts = [
            source.q4(prefix + "linear_attn.in_proj_qkv"),
            source.q4(prefix + "linear_attn.in_proj_z"),
            source.q4(prefix + "linear_attn.in_proj_b"),
            source.q4(prefix + "linear_attn.in_proj_a"),
        ]
        packed.section(tile_q4(*concatenate(parts, LAYOUT["packed_gdn"])))
        packed.section(source.bf16(prefix + "linear_attn.conv1d.weight"))
        logarithm = widen_bf16(source.bf16(prefix + "linear_attn.A_log"))
        packed.section((-np.exp(logarithm)).astype("<f4").tobytes())
        packed.section(source.bf16(prefix + "linear_attn.dt_bias"))
        packed.section(source.bf16(prefix + "linear_attn.norm.weight"))
        packed.section(tile_q4(*source.q4(prefix + "linear_attn.out_proj")))

    packed.section(source.bf16(prefix + "post_attention_layernorm.weight"))
    for name in ("gate_proj", "up_proj", "down_proj"):
        packed.section(tile_q4(*source.q4(prefix + "mlp." + name)))
    packed.finish()
    return path


def write_head(source: Checkpoint, destination: Path) -> Path:
    path = destination / "head.bin"
    packed = WeightFile(path, HEAD_MAGIC, LAYOUT["layers"], 2)
    packed.section(source.bf16("language_model.model.norm.weight"))
    packed.section(tile_q4(*source.q4("language_model.lm_head")))
    packed.finish()
    return path


def write_embedding(source: Checkpoint, destination: Path) -> Path:
    path = destination / "embedding.bin"
    packed = WeightFile(path, EMBEDDING_MAGIC, LAYOUT["vocabulary"], LAYOUT["hidden"])
    for run in plain_q4(*source.q4("language_model.model.embed_tokens")):
        packed.section(run)
    packed.finish()
    return path


def concatenate(parts, rows):
    codes = np.vstack([part[0] for part in parts])
    scales = np.vstack([part[1] for part in parts])
    biases = np.vstack([part[2] for part in parts])
    return pad_rows(codes, scales, biases, rows)


def widen_bf16(payload: bytes):
    return (np.frombuffer(payload, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def compare(built: Path, reference: Path, sizes: list[int], labels: list[str]) -> bool:
    *_, ours = read_sections(built, sizes)
    *_, theirs = read_sections(reference, sizes)
    clean = True
    for label, mine, yours in zip(labels, ours, theirs):
        if mine == yours:
            continue
        clean = False
        if label == "gdn-decay":
            a = np.frombuffer(mine, dtype="<f4")
            b = np.frombuffer(yours, dtype="<f4")
            moved = int((a != b).sum())
            worst = float(np.abs(a - b).max() / np.abs(b).max())
            print(
                f"    {label}: {moved} of {a.size} values differ, "
                f"worst {worst:.1e} relative - the known exp difference"
            )
        else:
            first = next(i for i in range(len(mine)) if mine[i] != yours[i])
            print(f"    {label}: differs from byte {first} of {len(mine)}")
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--layers", default="all")
    arguments = parser.parse_args()

    destination = arguments.destination or Path("build/converted/target")
    destination.mkdir(parents=True, exist_ok=True)
    source = Checkpoint(arguments.source)

    if arguments.layers == "all":
        indices = list(range(LAYOUT["layers"]))
    else:
        indices = [int(value) for value in arguments.layers.split(",")]

    exact = 0
    for index in indices:
        built = write_layer(source, index, destination)
        kind, sizes, labels = layer_sections(index)
        print(f"  layer-{index}.bin ({'attention' if kind else 'gdn'})", end="")
        if arguments.verify:
            reference = arguments.verify / "target" / f"layer-{index}.bin"
            print()
            if compare(built, reference, sizes, labels):
                exact += 1
                print("    every section identical")
        else:
            print(" written")

    if arguments.verify:
        print(f"\n{exact}/{len(indices)} layers identical in every section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
