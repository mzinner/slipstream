#!/usr/bin/env python3
"""Where decode time goes, how well guessing works, and what other guessing
strategies would have given - from one traced run.

Record a run (greedy, so the model's own output is fixed and known):

  SPLASH_TRACE=trace.jsonl build/engine-tests/generate-sample build/splash.metallib \\
      ~/models/qwen38-flash-next-splash 256 "$PROMPTS" > out.jsonl

then:

  models/qwen4exp/bench/trace_report.py trace.jsonl out.jsonl prompts.txt

prompts.txt holds the same ';'-separated token lists passed to generate-sample.

Sections:
  1. Time per step, split into parts (ms).
  2. Guessing: tokens per step, acceptance at each depth, confidence calibration.
  3. Cost model: step time against rows checked and draft depth (least squares).
  4. Strategies, each scored at the anchors the real run visited (same text):
     tokens per step and predicted tok/s from the cost model. "Upper bound"
     rows need a tree verifier (not built); the others run on today's chain.
"""

import argparse
import json
import statistics as st
from collections import Counter, defaultdict


def load(trace_path, out_path, prompts_path):
    trace = [json.loads(line) for line in open(trace_path)]
    outs = [json.loads(line) for line in open(out_path) if line.startswith("{")]
    prompts = [
        list(map(int, p.split(",")))
        for p in open(prompts_path).read().strip().split(";")
    ]
    steps = []  # (record, full sequence, step wall ms)
    i = 0
    for prompt, out in zip(prompts, outs):
        full = prompt + out["generated_tokens"]
        for ms in out["step_ms"]:
            if i < len(trace):
                steps.append((trace[i], full, ms))
            i += 1
    return steps, outs


def accepted_prefix(proposal, truth):
    n = 0
    for p, t in zip(proposal, truth):
        if p != t:
            break
        n += 1
    return n


def ngram_proposal(history, max_n, min_n, length):
    """Prompt lookup: continuation of the latest earlier match of the last
    n tokens (longest n first)."""
    for n in range(max_n, min_n - 1, -1):
        if len(history) <= n:
            continue
        key = history[-n:]
        for start in range(len(history) - n - 1, -1, -1):
            if history[start : start + n] == key:
                cont = history[start + n : start + n + length]
                if cont:
                    return cont, n
    return [], 0


def lstsq(xs, ys):
    """y ~ a + b1*x1 + b2*x2 ... by normal equations (tiny, no numpy)."""
    k = len(xs[0]) + 1
    rows = [[1.0] + list(x) for x in xs]
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    for c in range(k):  # Gauss-Jordan
        p = max(range(c, k), key=lambda r: abs(ata[r][c]))
        ata[c], ata[p], aty[c], aty[p] = ata[p], ata[c], aty[p], aty[c]
        for r in range(k):
            if r != c and ata[c][c]:
                f = ata[r][c] / ata[c][c]
                ata[r] = [a - f * b for a, b in zip(ata[r], ata[c])]
                aty[r] -= f * aty[c]
    return [aty[i] / ata[i][i] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("prompts")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=8,
        help="verify rows per step (anchor + guesses)",
    )
    a = ap.parse_args()
    steps, outs = load(a.trace, a.out, a.prompts)
    known = [(r, f, ms) for r, f, ms in steps if "accepted" in r]

    # 1. Time
    print(f"== 1. Time per step ({len(steps)} steps) ==")
    wall = st.mean(ms for _, _, ms in steps)
    parts = defaultdict(list)
    for r, _, _ in steps:
        for k, v in r["ms"].items():
            parts[k].append(v)
    mean = {k: st.mean(v) for k, v in parts.items()}
    rows = [
        ("whole step (wall)", wall),
        ("  draft head (MTP)", mean["mtp"]),
        ("    its GPU, first half", mean["mtp_gpu_a"]),
        ("    its expert staging", mean["mtp_stage"]),
        ("    its GPU, second half + pick", mean["mtp_gpu_b"]),
        ("  checking step, inside", mean["verify"] - mean["mtp"]),
        ("    GPU busy", mean["pure_gpu"]),
        ("    GPU wall (busy + waiting)", mean["gpu_wall"]),
        ("    waiting for SSD reads", mean["miss_reads"]),
        ("    waiting for pre-reads", mean["prefetch_wait"]),
        ("    host work between stages", mean["host_stage"]),
        ("  rest (last layer, head, sampling, bookkeeping)", wall - mean["verify"]),
    ]
    for name, v in rows:
        print(f"  {name:48s} {v:6.1f}")
    print(
        f"  expert reads from SSD per step: {st.mean(r['misses'] for r, _, _ in steps):.0f}"
    )

    # 2. Guessing
    print(f"\n== 2. Guessing ({len(known)} steps with a known outcome) ==")
    tps = st.mean(r["accepted"] + 1 for r, _, _ in known)
    print(
        f"  tokens per step {tps:.2f}   guesses made {st.mean(r['drafted'] for r, _, _ in known):.2f}"
        f"   kept {st.mean(r['accepted'] for r, _, _ in known):.2f}"
    )
    reach, keep = Counter(), Counter()
    for r, _, _ in known:
        for d in range(r["drafted"]):
            reach[d] += 1
            if r["accepted"] > d:
                keep[d] += 1
    print("  depth  reached  kept   rate")
    for d in sorted(reach):
        print(f"  {d + 1:5d}  {reach[d]:7d}  {keep[d]:5d}  {keep[d] / reach[d]:5.2f}")
    stop = Counter(
        "limit" if r["drafted"] == len(r["chosen"]) and r["drafted"] >= 5 else "unsure"
        for r, _, _ in known
    )
    print(f"  why guessing stopped: {dict(stop)}")
    print("  confidence -> how often right (all depths, given the depth was checked)")
    bins = defaultdict(lambda: [0, 0])
    for r, _, _ in known:
        for d in range(min(r["drafted"], r["accepted"] + 1)):
            b = min(9, int(r["conf"][d] * 10))
            bins[b][0] += 1
            bins[b][1] += r["accepted"] > d
    for b in sorted(bins):
        n, k = bins[b]
        print(
            f"    {b / 10:.1f}-{(b + 1) / 10:.1f}: {n:5d} guesses, right {k / n:5.2f}"
        )
    # Where the right token ranked among the head's candidates, at the first miss.
    rank = Counter()
    for r, f, _ in known:
        if r["accepted"] < r["drafted"]:
            d = r["accepted"]
            cands = r["cand"][d]
            rank[
                cands.index(r["next"]) + 1 if r["next"] in cands else "not in top 4"
            ] += 1
    print(
        f"  at the first wrong guess, the right token was the head's choice #: {dict(rank)}"
    )

    # 3. Cost model
    print("\n== 3. Cost model ==")
    xs = [(r["live"],) for r, _, _ in steps]
    ys = [ms - r["ms"]["mtp"] for r, _, ms in steps]
    c0, c1 = lstsq(xs, ys)
    depth_x = [(len(r["chosen"]),) for r, _, _ in steps]
    m0, m1 = lstsq(depth_x, [r["ms"]["mtp"] for r, _, _ in steps])
    print(f"  step without draft head ~= {c0:.1f} + {c1:.2f} x rows checked  (ms)")
    print(f"  draft head             ~= {m0:.1f} + {m1:.2f} x guesses made    (ms)")

    print("  what one more row costs, by part (ms):")
    for part in ("pure_gpu", "miss_reads", "host_stage", "prefetch_wait", "staging"):
        _, slope = lstsq(xs, [r["ms"][part] for r, _, _ in steps])
        print(f"    {part:14s} {slope:5.2f}")
    _, slope = lstsq(xs, [r["misses"] for r, _, _ in steps])
    print(f"    SSD expert reads per extra row: {slope:.1f}")

    def cost(rows_checked, mtp_depths):
        return c0 + c1 * rows_checked + (m0 + m1 * mtp_depths if mtp_depths else 0.0)

    # 4. Strategies
    print("\n== 4. Strategies at the same anchors (greedy) ==")
    print(f"  {'strategy':58s} tok/step  ms/step  tok/s")
    results = []

    def report(name, tok, ms):
        results.append((name, tok, ms))
        print(f"  {name:58s} {tok:8.2f} {ms:8.1f} {1000 * tok / ms:6.1f}")

    def run(name, policy):
        toks, mss = [], []
        for r, f, _ in known:
            truth = f[r["anchor_pos"] + 1 :]
            t, rows_checked, depths = policy(r, f, truth)
            toks.append(t)
            mss.append(cost(rows_checked, depths))
        report(name, st.mean(toks), st.mean(mss))

    run(
        "today: draft head chain",
        lambda r, f, t: (r["accepted"] + 1, 1 + r["drafted"], len(r["chosen"])),
    )
    run(
        "ceiling: no wasted rows (knows when to stop)",
        lambda r, f, t: (
            r["accepted"] + 1,
            1 + r["accepted"],
            min(len(r["chosen"]), r["accepted"] + 1),
        ),
    )

    def stop_on_chain(tau, min_single):
        """Keep guesses while the product of confidences stays >= tau (and
        each single one >= min_single). Can only shorten today's chain."""

        def policy(r, f, truth):
            keep, chain = 0, 1.0
            for d in range(r["drafted"]):
                chain *= r["conf"][d]
                if chain < tau or r["conf"][d] < min_single:
                    break
                keep = d + 1
            # The head still ran the step that produced the dropped guess.
            return (
                min(r["accepted"], keep) + 1,
                1 + keep,
                min(len(r["chosen"]), keep + 1),
            )

        return policy

    for tau in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        run(f"stop when chain confidence < {tau:.2f}", stop_on_chain(tau, 0.3))

    def chain_plus_alts(alts, min_conf):
        def policy(r, f, truth):
            k = r["accepted"]
            gained = 0
            extra = 0
            for d in range(r["drafted"]):
                if r["conf"][d] < min_conf:
                    extra += alts
            if (
                k < r["drafted"]
                and r["conf"][k] < min_conf
                and truth[k] in r["cand"][k][1 : 1 + alts]
            ):
                gained = 1
            return k + 1 + gained, 1 + r["drafted"] + extra, len(r["chosen"])

        return policy

    for alts, mc in ((1, 1.01), (1, 0.9), (2, 0.9), (3, 0.9)):
        run(
            f"upper bound: + runner-up leaves x{alts} where conf < {mc:.2f}",
            chain_plus_alts(alts, mc),
        )

    def lookup(max_n, min_n, length, mode):
        def policy(r, f, truth):
            history = f[: r["anchor_pos"] + 1]
            prop, n = ngram_proposal(history, max_n, min_n, length)
            ng = accepted_prefix(prop, truth)
            chain = r["accepted"]
            if mode == "lookup only":
                return ng + 1, 1 + len(prop), 0
            if mode == "lookup if found, else head":
                if prop:
                    # The head still reads the kept rows (its cache), one step.
                    return ng + 1, 1 + len(prop), 1
                return chain + 1, 1 + r["drafted"], len(r["chosen"])
            if mode == "both, keep longer (tree)":
                return (
                    max(ng, chain) + 1,
                    1 + r["drafted"] + len(prop),
                    len(r["chosen"]),
                )
            if mode == "head, then lookup extends":
                # Chain: head's guesses, then lookup continues from them.
                ext, _ = ngram_proposal(
                    history + r["chosen"][: r["drafted"]],
                    max_n,
                    min_n,
                    a.max_rows - 1 - r["drafted"],
                )
                full = r["chosen"][: r["drafted"]] + ext
                return accepted_prefix(full, truth) + 1, 1 + len(full), len(r["chosen"])

        return policy

    for mode in (
        "lookup only",
        "lookup if found, else head",
        "head, then lookup extends",
    ):
        for max_n, min_n in ((4, 2), (6, 3)):
            run(
                f"{mode} (match {min_n}-{max_n} tokens)",
                lookup(max_n, min_n, a.max_rows - 1, mode),
            )
    run(
        "upper bound: both, keep longer (tree, match 2-4)",
        lookup(4, 2, a.max_rows - 1, "both, keep longer (tree)"),
    )


if __name__ == "__main__":
    main()
