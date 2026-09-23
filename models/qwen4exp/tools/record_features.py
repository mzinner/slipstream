#!/usr/bin/env python3
"""Record the model's inner state over the session corpus, for guesser training.

Picks sessions from the corpus (session_corpus.py) up to a token budget, runs
the engine's prompt-only recording mode over them through the memory guard,
and stores the compressed result (feature_store.py) as it arrives. Rows are
stored in sample order; samples.json says which corpus sample each run of rows
came from.

  .venv/bin/python models/qwen4exp/tools/record_features.py \\
      ~/models/qwen38-flash-next-drafter-data --split test
  .venv/bin/python models/qwen4exp/tools/record_features.py \\
      ~/models/qwen38-flash-next-drafter-data --split train --tokens 3000000

--sequences file (';'-separated token lists) --name NAME records given
sequences instead, e.g. prompts plus the model's own greedy continuations for
scoring a guesser against today's draft head.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_store  # noqa: E402

LAYERS = "11,23,35"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="session_corpus.py output folder")
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--sequences", type=Path, help="record these instead")
    parser.add_argument("--name", help="output folder name for --sequences")
    parser.add_argument(
        "--tokens", type=int, default=0, help="budget; 0 = every sample"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--package", type=Path, default=Path.home() / "models/qwen38-flash-next-splash"
    )
    parser.add_argument("--max-seconds", type=int, default=6 * 3600)
    args = parser.parse_args()

    if bool(args.split) == bool(args.sequences):
        raise SystemExit("give --split or --sequences")
    out = args.data / "features" / (args.split or args.name)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; move it aside first")
    out.mkdir(parents=True, exist_ok=True)
    ids = np.fromfile(args.data / "ids.u32", dtype="<u4")
    if args.sequences:
        texts = args.sequences.read_text().strip().split(";")
        ids = np.array([int(t) for text in texts for t in text.split(",")], dtype="<u4")
        lengths = [len(text.split(",")) for text in texts]
        starts = np.cumsum([0] + lengths[:-1])
        corpus = [
            {"offset": int(o), "length": n, "split": args.name}
            for o, n in zip(starts, lengths)
        ]
        np.save(out / "ids.npy", ids)
    else:
        corpus = json.loads((args.data / "samples.json").read_text())
    samples = [s for s in corpus if s["split"] == (args.split or args.name)]
    order = (
        np.arange(len(samples))
        if args.sequences
        else np.random.default_rng(args.seed).permutation(len(samples))
    )
    chosen, total = [], 0
    for index in order:
        if args.tokens and total + samples[index]["length"] > args.tokens:
            continue
        chosen.append(samples[index])
        total += samples[index]["length"]
    rows = 0
    for sample in chosen:
        sample["row"] = rows
        rows += sample["length"]
    (out / "samples.json").write_text(json.dumps(chosen, indent=1))
    prompts = out / "prompts.txt"
    prompts.write_text(
        ";".join(
            ",".join(map(str, ids[s["offset"] : s["offset"] + s["length"]]))
            for s in chosen
        )
    )
    print(
        f"{args.split or args.name}: {len(chosen)} samples, {total:,} tokens",
        flush=True,
    )

    pipe = out / "features.pipe"
    os.mkfifo(pipe)
    store_args = argparse.Namespace(
        source=pipe, compression=args.data / "compression.npz", out=out
    )
    store = threading.Thread(target=feature_store.store, args=(store_args,))
    store.start()
    environment = dict(
        os.environ,
        SPLASH_CAPTURE_LAYERS=LAYERS,
        SPLASH_DUMP_PREFILL_FEATURES=str(pipe),
        SPLASH_EXPERT_CACHE_GIB=os.environ.get("SPLASH_EXPERT_CACHE_GIB", "34"),
    )
    command = [
        str(ROOT / "dev/benchmarks/guarded.py"),
        "--max-seconds",
        str(args.max_seconds),
        "--",
        str(ROOT / "build/engine-tests/generate-sample"),
        str(ROOT / "build/splash.metallib"),
        str(args.package),
        "1",
        f"@{prompts}",
    ]
    with open(out / "engine.log", "w") as log:
        result = subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    # If the engine never opened the pipe, open it once so the reader stops.
    if store.is_alive():
        with open(pipe, "wb"):
            pass
    store.join()
    pipe.unlink()
    stored = (
        json.loads((out / "meta.json").read_text())["rows"]
        if (out / "meta.json").exists()
        else 0
    )
    print(f"engine exit {result.returncode}; {stored:,} of {rows:,} rows stored")
    if result.returncode or stored != rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
