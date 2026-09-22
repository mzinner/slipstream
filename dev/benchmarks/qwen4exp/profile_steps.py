#!/usr/bin/env python3
"""Where each decode step's time goes, and how many tokens it buys.

Runs generate-sample on one warm engine with SPLASH_STEP_TIMING=1 and joins,
per decode step: wall time, tokens produced, and the target's own breakdown
([Verify Timing]: expert staging and misses, GPU wall and pure GPU time,
MTP draft time). Prints the averages that set tok/s, the step-time spread,
and what the slowest tenth of steps has in common.

  tok/s = tokens per step / seconds per step, so both halves are shown.
"""
import argparse
import json
import os
import re
import statistics as st
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TIMING = re.compile(
    r"\[Verify Timing\] Resident 0\.\.\d+: ([\d.]+) ms \| Staging: ([\d.]+) ms "
    r"\(misses: (\d+)\) \| GPU Wall: ([\d.]+) ms \(pure GPU: ([\d.]+) ms\) \| MTP: ([\d.]+) ms")
TOKEN = re.compile(r"Token: \d+ \(step: (\d+),")


def pct(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="code")
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--temperature", default="")
    parser.add_argument("--warm", default="short",
                        help="prompt run first so the measured one is warm")
    args = parser.parse_args()
    prompts = json.loads((ROOT / "build/qwen4exp-prompts.json").read_text())
    ids = lambda name: ",".join(map(str, prompts[name]["ids"]))
    joined = ";".join(filter(None, [ids(args.warm) if args.warm else "", ids(args.prompt)]))
    env = dict(os.environ, SPLASH_STEP_TIMING="1")
    if args.temperature:
        env["SPLASH_TEMPERATURE"] = args.temperature
    run = subprocess.run(
        [str(ROOT / "build/engine-tests/generate-sample"), str(ROOT / "build/splash.metallib"),
         str(Path.home() / "models/qwen38-flash-next-splash"), str(args.tokens), joined],
        env=env, capture_output=True, text=True, check=True)
    result = json.loads(run.stdout.strip().splitlines()[-1])
    # stderr of the last prompt only: after its "Prefill completed".
    err = run.stderr.split("Prefill completed")[-1]
    timings = [tuple(map(float, m.groups())) for m in TIMING.finditer(err)]
    per_step = {}
    for m in TOKEN.finditer(err):
        per_step[int(m.group(1))] = per_step.get(int(m.group(1)), 0) + 1
    steps = result["step_ms"]
    tokens = [per_step.get(i + 1, 0) for i in range(len(steps))]
    n = min(len(steps), len(timings))
    rows = [(steps[i], tokens[i], *timings[i]) for i in range(n)]
    names = ["wall", "tokens", "resident", "staging", "misses", "gpu_wall", "pure_gpu", "mtp"]
    cols = {k: [r[j] for r in rows] for j, k in enumerate(names)}

    total_ms, total_tokens = sum(steps), sum(tokens)
    print(f"{args.prompt} T={args.temperature or 'greedy'}: {total_tokens} tokens in "
          f"{len(steps)} steps, {1000 * total_tokens / total_ms:.1f} tok/s")
    print(f"  tokens per step  mean {total_tokens / len(steps):.2f}   "
          + "  ".join(f"{k}:{tokens.count(k)}" for k in sorted(set(tokens))))
    print(f"  step ms          mean {st.mean(steps):.2f}  p50 {pct(steps, 50):.2f}  "
          f"p90 {pct(steps, 90):.2f}  p99 {pct(steps, 99):.2f}  max {max(steps):.1f}")
    print("  mean per step (ms):")
    target = [r[3] + r[5] + r[7] for r in rows]  # staging + gpu wall + mtp
    for k in ("resident", "staging", "gpu_wall", "pure_gpu", "mtp", "misses"):
        print(f"    {k:10s} {st.mean(cols[k]):7.2f}")
    other = [rows[i][0] - (rows[i][2] + rows[i][3] + rows[i][5] + rows[i][7]) for i in range(n)]
    print(f"    {'outside':10s} {st.mean(other):7.2f}   (wall minus resident, staging, GPU wall, MTP)")
    slow = sorted(range(n), key=lambda i: -rows[i][0])[: max(1, n // 10)]
    fast = [i for i in range(n) if i not in slow]
    print("  slowest 10% vs the rest (mean):")
    for j, k in enumerate(names):
        if k == "tokens":
            continue
        print(f"    {k:10s} {st.mean(rows[i][j] for i in slow):7.2f}  vs {st.mean(rows[i][j] for i in fast):7.2f}")


if __name__ == "__main__":
    main()
