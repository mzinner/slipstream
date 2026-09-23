#!/usr/bin/env python3
"""Write the full-precision reference in llama-perplexity's KL-divergence format.

llama-perplexity --kl-divergence compares a model against a base file of log
probabilities. Writing that file from our own reference lets llama.cpp be
scored against the same ground truth as Splash, on the same token ids.

The format, from tools/perplexity/perplexity.cpp:
  "_logits_", uint32 n_ctx, int32 n_vocab, int32 n_chunk, int32 tokens[n_ctx]
  then, for each position n_ctx/2 .. n_ctx-2, a row of nv uint16 values:
  float scale, float min_log_prob, then each logit quantized to 16 bits
  above the row's floor (max - 16), with nv = 2*((n_vocab + 1)/2) + 4.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

V = 248320


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--passages", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--reference", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n-ctx", type=int)
    args = a.parse_args()
    ids = json.loads(Path(args.passages).read_text())[args.name]["ids"]
    n_ctx = args.n_ctx or (len(ids) // 2) * 2
    ref = np.fromfile(args.reference, dtype=np.uint16).reshape(len(ids), V)
    first = n_ctx // 2
    nv = 2 * ((V + 1) // 2) + 4
    with open(args.out, "wb") as out:
        out.write(b"_logits_")
        out.write(struct.pack("<Iii", n_ctx, V, 1))
        out.write(np.asarray(ids[:n_ctx], dtype=np.int32).tobytes())
        for j in range(first, n_ctx - 1):
            logits = (
                (ref[j].astype(np.uint32) << 16).view(np.float32).astype(np.float32)
            )
            mx = logits.max()
            mn = max(logits.min(), mx - 16)
            lse = np.float32(np.log(np.exp((logits - mx).astype(np.float64)).sum()))
            min_log_prob = np.float32(mn - mx - lse)
            scale = np.float32((mx - mn) / 65535.0)
            row = np.zeros(nv, dtype=np.uint16)
            row[:4] = np.frombuffer(
                struct.pack("<ff", scale, min_log_prob), dtype=np.uint16
            )
            q = np.where(logits > mn, np.rint((logits - mn) / scale), 0)
            row[4 : 4 + V] = np.clip(q, 0, 65535).astype(np.uint16)
            out.write(row.tobytes())
    print(
        f"n_ctx {n_ctx}: scored positions {first}..{n_ctx - 2} ({n_ctx - 1 - first} rows)"
    )


if __name__ == "__main__":
    main()
