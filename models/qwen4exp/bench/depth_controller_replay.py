#!/usr/bin/env python3
"""Replay upstream PR incoai/splash#115's MTP depth controller on a traced run.

The controller (FlashMTPDepthController) picks one draft depth for a stretch of
text: it keeps an average of how often each guess position is right (alpha
0.08, faster after repeated first-guess misses), scores each depth as expected
tokens per step over its cost, switches only when 3% better, and every ~1 s
spends 4 steps probing another depth. Here the cost is our measured model and
the text is fixed by the greedy trace. Guesses deeper than today's engine
drafted are unknown and counted wrong, which leans slightly against the
controller.

  models/qwen4exp/bench/depth_controller_replay.py trace.jsonl out.jsonl prompts.txt

Result on 2026-09-23: 31-41 tok/s vs 40.5 / 43.8 for today's per-step
confidence stop (10-prompt suite / session cut points).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_report import accepted_prefix, load  # noqa: E402

C0, C1, M1 = 32.2, 8.94, 2.23  # step ~= C0 + C1 x rows + M1 x guesses (ms)


def cost(depth):
    return C0 + C1 * (1 + depth) + M1 * depth


def score(acceptance, depth):
    expected, prefix = 1.0, 1.0
    for position in range(depth):
        prefix *= acceptance[position]
        expected += prefix
    return expected / cost(depth)


def replay(known, max_depth, with_confidence_stop, alpha=0.08, repeat_alpha=0.5):
    acceptance = [0.6] * max_depth
    depth = min(3, max_depth)
    first_misses = 0
    tokens = milliseconds = since_probe = 0.0
    probe_left = probe_depth = 0
    for record, full in known:
        truth = full[record["anchor_pos"] + 1 :]
        drafted = record["chosen"][: record["drafted"]]
        if probe_left == 0 and since_probe >= 1000.0:
            probe_left, since_probe = 4, 0.0
            others = [d for d in range(max_depth + 1) if d != depth]
            probe_depth = (
                max_depth
                if depth == 0
                else max(others, key=lambda d: score(acceptance, d))
            )
        chosen = probe_depth if probe_left else depth
        probe_left = max(0, probe_left - 1)
        keep = min(chosen, len(drafted)) if with_confidence_stop else chosen
        proposal = drafted[:keep]
        matched = accepted_prefix(proposal, truth)
        step = cost(keep)
        tokens += matched + 1
        milliseconds += step
        since_probe += step
        for position in range(min(matched + 1, chosen, len(proposal))):
            hit = 1.0 if position < matched else 0.0
            rate = repeat_alpha if position == 0 and first_misses and not hit else alpha
            acceptance[position] = (1 - rate) * acceptance[position] + rate * hit
        first_misses = first_misses + 1 if matched == 0 and chosen else 0
        best = max(range(max_depth + 1), key=lambda d: score(acceptance, d))
        if best != depth and score(acceptance, best) >= score(acceptance, depth) * 1.03:
            depth = best
    return tokens / len(known), milliseconds / len(known)


def main():
    steps, _ = load(*sys.argv[1:4])
    known = [(record, full) for record, full, _ in steps if "accepted" in record]
    tokens = sum(r["accepted"] + 1 for r, _ in known) / len(known)
    milliseconds = sum(cost(r["drafted"]) for r, _ in known) / len(known)
    rows = [("today (per-step confidence stop)", tokens, milliseconds)]
    for max_depth in (3, 5):
        for stop in (False, True):
            label = f"PR controller, max {max_depth}" + (" + our stop" if stop else "")
            rows.append((label, *replay(known, max_depth, stop)))
    for label, tokens, milliseconds in rows:
        print(
            f"  {label:36s} {tokens:.2f} tok/step {milliseconds:5.1f} ms "
            f"{1000 * tokens / milliseconds:5.1f} tok/s"
        )


if __name__ == "__main__":
    main()
