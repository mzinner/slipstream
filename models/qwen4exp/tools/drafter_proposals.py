#!/usr/bin/env python3
"""A trained block guesser's proposals at the steps a real traced run visited.

The traced run (SPLASH_TRACE, greedy) fixes the text and the anchors; the
sequences it produced (prompt + output) were recorded with record_features.py
--sequences. For each traced step this writes the guesser's block for that
anchor, in trace order, for trace_report.py --proposals. Both guessers are then
scored on the same text with the same cost model.

  ~/venvs/drafter/bin/python models/qwen4exp/tools/drafter_proposals.py \\
      runs/v0 ~/models/qwen38-flash-next-drafter-data eval10 trace.jsonl out.jsonl \\
      prompts.txt > proposals.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_drafter as td  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="training output folder")
    parser.add_argument("data", type=Path)
    parser.add_argument("name", help="recorded --sequences folder name")
    parser.add_argument("trace")
    parser.add_argument("out")
    parser.add_argument("prompts")
    parser.add_argument("--window", type=int, default=2048)
    args = parser.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    split = td.Split(args.data, args.name, tuple(config["taps"]))
    embedding, head, draft_ids = td.load_frozen(args.data)
    model = td.Drafter(
        config["part_dims"], config["hidden"], d=config["d"], layers=config["layers"]
    )
    # Weights stay float32: bfloat16 keeps ~3 digits and swallows small updates.
    model.load_weights(str(args.run / "drafter.safetensors"))

    trace = [json.loads(line) for line in open(args.trace)]
    outs = [json.loads(line) for line in open(args.out) if line.startswith("{")]
    prompts = [p for p in open(args.prompts).read().strip().split(";")]
    step = 0
    for sample, (prompt, out) in enumerate(zip(prompts, outs)):
        row0 = split.samples[sample]["row"]
        full = list(map(int, prompt.split(","))) + out["generated_tokens"]
        if split.samples[sample]["length"] != len(full) or not np.array_equal(
            split.tokens[row0 : row0 + len(full)], full
        ):
            raise SystemExit(
                f"sample {sample}: recorded sequence differs from prompt + output"
            )
        for _ in out["step_ms"]:
            if step >= len(trace):
                break
            record = trace[step]
            step += 1
            anchor = row0 + record["anchor_pos"]
            start = max(row0, anchor - args.window + 1)
            guess = model(
                mx.array(split.features(start, anchor + 1)).astype(mx.float32),
                embedding[mx.array([int(split.tokens[anchor])])].astype(mx.float32),
                mx.array([anchor]),
                start,
            )
            logits = (guess[0] @ head.T.astype(guess.dtype)).astype(mx.float32)
            probabilities = mx.softmax(logits, -1)
            picks = np.array(mx.argmax(logits, -1))
            confidence = np.array(mx.max(probabilities, -1))
            print(
                json.dumps(
                    {
                        "anchor_pos": record["anchor_pos"],
                        "tokens": [int(draft_ids[p]) for p in picks],
                        "conf": [round(float(c), 5) for c in confidence],
                    }
                )
            )
    if step != len(trace):
        raise SystemExit(f"used {step} of {len(trace)} traced steps")


if __name__ == "__main__":
    main()
