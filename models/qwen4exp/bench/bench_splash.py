#!/usr/bin/env python3
"""Time Splash on the shared prompts: prefill speed and decode speed.

Decode speed is reported two ways because they answer different questions:
  mean     generated tokens / total decode time - what a user waits through
  median   the typical single step, which hides a slow start
Each prompt runs in its own process, as a request would on a fresh server;
the operating system's file cache stays warm between them.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def run(binary, metallib, model, ids, tokens):
    return run_many(binary, metallib, model, [ids], tokens)[0]


def run_many(binary, metallib, model, prompts, tokens):
    """Every prompt in one process, one after another, as a server sees them."""
    result = subprocess.run(
        [
            binary,
            metallib,
            model,
            str(tokens),
            ";".join(",".join(map(str, ids)) for ids in prompts),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return [summarize(json.loads(line)) for line in lines]


def summarize(out):
    steps = out["step_ms"]
    generated = len(out["generated_tokens"])
    decode_s = sum(steps) / 1000
    return {
        "prompt_tokens": out["prompt_tokens"],
        "prefill_tok_s": out["prompt_tokens"] / (out["prefill_ms"] / 1000),
        "generated": generated,
        "decode_mean_tok_s": (generated - 1) / decode_s if decode_s else 0.0,
        "decode_median_tok_s": 1000 / statistics.median(steps) if steps else 0.0,
        "tokens": out["generated_tokens"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(ROOT / "build/qwen4exp-prompts.json"))
    parser.add_argument(
        "--binary", default=str(ROOT / "build/engine-tests/generate-sample")
    )
    parser.add_argument("--metallib", default=str(ROOT / "build/splash.metallib"))
    parser.add_argument(
        "--model", default=str(Path.home() / "models/qwen38-flash-next-splash")
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--only", default="short,code,long")
    parser.add_argument("--out")
    parser.add_argument(
        "--one-process",
        action="store_true",
        help="run every prompt on one warm engine, as a server would",
    )
    args = parser.parse_args()
    prompts = json.loads(Path(args.prompts).read_text())
    if args.one_process:
        names = args.only.split(",")
        batch = [prompts["short"]["ids"]] + [prompts[n]["ids"] for n in names]
        results = dict(
            zip(
                ["warm-up"] + names,
                run_many(args.binary, args.metallib, args.model, batch, args.tokens),
            )
        )
        results.pop("warm-up")
        for name, r in results.items():
            print(
                f"{name:6s} prompt {r['prompt_tokens']:5d}  prefill {r['prefill_tok_s']:7.1f} tok/s"
                f"  decode {r['decode_mean_tok_s']:5.2f} tok/s (median step {r['decode_median_tok_s']:5.2f})"
                f"  generated {r['generated']}",
                flush=True,
            )
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=1))
        return
    # One throwaway run so the first measured prompt is not paying for cold files.
    run(args.binary, args.metallib, args.model, prompts["short"]["ids"], 8)
    results = {}
    for name in args.only.split(","):
        r = run(
            args.binary, args.metallib, args.model, prompts[name]["ids"], args.tokens
        )
        results[name] = r
        print(
            f"{name:6s} prompt {r['prompt_tokens']:5d}  prefill {r['prefill_tok_s']:7.1f} tok/s"
            f"  decode {r['decode_mean_tok_s']:5.2f} tok/s (median step {r['decode_median_tok_s']:5.2f})"
            f"  generated {r['generated']}",
            flush=True,
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
