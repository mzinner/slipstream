#!/usr/bin/env python3
"""Score an engine's logits against the full-precision reference.

For every position that has a next token, three numbers:

  same pick   the engine's most likely next token is the reference's
  KL          how far the engine's probabilities drift from the reference's
              (0 is identical; a well-quantized 4-bit model sits near 0.01-0.05)
  perplexity  how surprised each model is by the text that actually follows
              (lower is better; the reference's is the floor to compare with)

Positions can be restricted to a range, because at long context the
reference drops tokens through its sparse-attention indexer and the engine
does not, so the two diverge there for a known reason.
"""

import argparse
import json
from pathlib import Path

import numpy as np

V = 248320


def load(path, rows):
    raw = np.fromfile(path, dtype=np.uint16)
    assert raw.size == rows * V, f"{path}: {raw.size / V} rows, expected {rows}"
    return raw.reshape(rows, V)


def to_f32(block):
    return (block.astype(np.uint32) << 16).view(np.float32)


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def score(reference, engine, ids, begin, end, chunk=64):
    picks, kls, ref_nll, eng_nll = [], [], [], []
    for start in range(begin, end, chunk):
        stop = min(end, start + chunk)
        r = log_softmax(to_f32(reference[start:stop]))
        e = log_softmax(to_f32(engine[start:stop]))
        target = np.asarray(ids[start + 1 : stop + 1])
        picks.append(r.argmax(1) == e.argmax(1))
        kls.append((np.exp(r) * (r - e)).sum(1))
        rows = np.arange(stop - start)
        ref_nll.append(-r[rows, target])
        eng_nll.append(-e[rows, target])
    cat = np.concatenate
    return {
        "positions": end - begin,
        "same_pick": float(cat(picks).mean()),
        "kl": float(cat(kls).mean()),
        "ppl_reference": float(np.exp(cat(ref_nll).mean())),
        "ppl_engine": float(np.exp(cat(eng_nll).mean())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    ids = json.loads(Path(args.passages).read_text())[args.name]["ids"]
    T = len(ids)
    reference = load(args.reference, T)
    engine = load(args.engine, T)
    end = min(args.end or T - 1, T - 1)
    result = score(reference, engine, ids, args.begin, end)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
