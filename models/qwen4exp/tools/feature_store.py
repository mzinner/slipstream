#!/usr/bin/env python3
"""Store the model's inner state compactly, for guesser training.

The engine's recording mode (SPLASH_CAPTURE_LAYERS + SPLASH_DUMP_PREFILL_FEATURES)
writes ~65 KB per prompt position. That is far too much for millions of
positions, so each chosen layer is compressed to its most important
directions (principal components, fitted once on a sample). The head input,
the guesser's main signal, is kept whole (float16).

  fit    raw recording -> compression.npz (mean and directions per part)
  store  reads the engine's records as they arrive (a named pipe) and appends
         the compressed parts plus the model's top picks to --out

Record format: see Runtime::dumpPrefillFeatures in runtime/model/Runtime.mm.

  .venv/bin/python models/qwen4exp/tools/feature_store.py fit raw/fit.bin \\
      --out compression.npz --tap-dims 512
  .venv/bin/python models/qwen4exp/tools/feature_store.py store features.pipe \\
      --compression compression.npz --out features/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x31464C53


def bf16(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        piece = stream.read(size - len(data))
        if not piece:
            if data:
                raise EOFError("record cut short")
            return None
        data += piece
    return bytes(data)


def records(stream):
    """Yields one dict per engine record."""
    while True:
        header = read_exact(stream, 24)
        if header is None:
            return
        magic, rows, taps, width, hidden, top = np.frombuffer(header, "<u4")
        if magic != MAGIC:
            raise ValueError("not a feature record")
        rows, taps, width, hidden, top = map(int, (rows, taps, width, hidden, top))
        layers = np.frombuffer(read_exact(stream, 4 * taps), "<u4").tolist()
        size = 2 * taps * rows * width + 2 * rows * hidden + 8 * rows * top
        body = read_exact(stream, size)
        offset = 0

        def take(dtype, count, shape):
            nonlocal offset
            array = np.frombuffer(body, dtype, count, offset).reshape(shape)
            offset += array.nbytes
            return array

        yield {
            "rows": rows,
            "layers": layers,
            "taps": take("<u2", taps * rows * width, (taps, rows, width)),
            "head": take("<u2", rows * hidden, (rows, hidden)),
            "top_ids": take("<u4", rows * top, (rows, top)),
            "top_logprobs": take("<f4", rows * top, (rows, top)),
        }


def fit_part(rows, dims):
    """Mean, the top `dims` directions, and the share of variance they keep."""
    x = rows.astype(np.float64)
    mean = x.mean(0)
    x -= mean
    covariance = x.T @ x / len(x)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    kept = {
        d: float(values[:d].sum() / values.sum()) for d in (128, 256, 512, 1024, 2048)
    }
    return mean.astype(np.float32), vectors[:, :dims].astype(np.float32), kept


def fit(args):
    parts = {}
    with open(args.raw, "rb") as stream:
        for record in records(stream):
            for index, layer in enumerate(record["layers"]):
                parts.setdefault(f"layer{layer}", []).append(
                    bf16(record["taps"][index])
                )
    saved = {}
    for name, chunks in parts.items():
        rows = np.concatenate(chunks)
        mean, directions, kept = fit_part(rows, args.tap_dims)
        saved[f"{name}_mean"], saved[f"{name}_directions"] = mean, directions
        share = "  ".join(f"{d}: {v:.3f}" for d, v in kept.items())
        print(
            f"{name:8s} {rows.shape[0]:,} rows x {rows.shape[1]}  variance kept  {share}"
        )
    np.savez(args.out, **saved)


def store(args):
    compression = np.load(args.compression)
    args.out.mkdir(parents=True, exist_ok=True)
    files, total, hidden = {}, 0, 0
    source = open(args.source, "rb")
    for record in records(source):
        parts = {
            f"layer{layer}": record["taps"][i]
            for i, layer in enumerate(record["layers"])
        }
        parts["head"] = record["head"]
        for name, raw in parts.items():
            if name == "head":
                packed = bf16(raw).astype(np.float16)
            else:
                mean = compression[f"{name}_mean"]
                packed = (
                    (bf16(raw) - mean) @ compression[f"{name}_directions"]
                ).astype(np.float16)
            files.setdefault(name, open(args.out / f"{name}.f16", "ab")).write(
                packed.tobytes()
            )
        files.setdefault("top_ids", open(args.out / "top_ids.u32", "ab")).write(
            record["top_ids"].tobytes()
        )
        files.setdefault(
            "top_logprobs", open(args.out / "top_logprobs.f32", "ab")
        ).write(record["top_logprobs"].tobytes())
        total += record["rows"]
        hidden = record["head"].shape[1]
        if total % 65536 < record["rows"]:
            print(f"{total:,} rows stored", file=sys.stderr, flush=True)
    for handle in files.values():
        handle.close()
    dims = {
        name: int(compression[f"{name}_directions"].shape[1])
        if name != "head"
        else hidden
        for name in files
        if name[0] != "t"
    }
    meta = args.out / "meta.json"
    before = json.loads(meta.read_text())["rows"] if meta.exists() else 0
    meta.write_text(
        json.dumps({"rows": before + total, "dims": dims, "top": 8}, indent=1)
    )
    print(f"{total:,} rows stored ({before + total:,} in total)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("raw", type=Path)
    fit_parser.add_argument("--out", type=Path, required=True)
    fit_parser.add_argument("--tap-dims", type=int, default=512)
    store_parser = commands.add_parser("store")
    store_parser.add_argument(
        "source", type=Path, help="engine output: a file or named pipe"
    )
    store_parser.add_argument("--compression", type=Path, required=True)
    store_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "store" and not args.source.exists():
        os.mkfifo(args.source)
    {"fit": fit, "store": store}[args.command](args)


if __name__ == "__main__":
    main()
