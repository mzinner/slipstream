#!/usr/bin/env python3
"""How many experts each check step must read from the SSD, for a given expert
cache plan - replayed offline from a routing log, so every plan is compared on
the same text.

Record a log (SPLASH_ROUTE_LOG=path on generate-sample), then:

  models/qwen4exp/bench/cache_plan.py routes.log [--slots 272] [--budget 13056]

Reports, per eviction rule:
  - SSD reads per check step with the same number of slots in every layer;
  - the same after fitting the slots unevenly (more where a layer's reads fall
    fastest), under the same total;
  - that split itself (for SPLASH_EXPERT_SLOTS).

The slot split is fitted twice: to the mean reads a step (what a total does)
and to the 95th percentile (what a slow step does). Same slots either way;
only their distribution changes. A step is gated by its own experts, so the
tail is what shows up as a stalled decode - see docs/tokenspeed-optimizations.md.

Rules:
  engine   today's rule: evict the least used (use counts halve every 512
           steps), then the least recent; a prompt chunk loads its most-used
           experts
  engine:N the same, with use counts halving every N steps
  lru      evict the least recent
  optimal  evict the expert needed furthest ahead - the best any rule could do
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

HALF_LIFE = 512  # kFrequencyHalfLife in Qwen4ExpTarget.cpp
EXPERTS = 512


def load(path):
    """Per layer: [(phase, Counter of expert uses)], one entry per chunk/step."""
    events = defaultdict(list)
    last = None
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            phase, layer = parts[0], int(parts[1])
            experts = [int(e) for e in parts[2:]]
            if last == (phase, layer) and events[layer]:
                events[layer][-1][1].update(experts)
            else:
                events[layer].append((phase, Counter(experts)))
            last = (phase, layer)
    return events


def replay(sequence, capacity, rule):
    """SSD reads during each check step for one layer, in step order."""
    half_life = int(rule.split(":")[1]) if ":" in rule else HALF_LIFE
    rule = rule.split(":")[0]
    slot_of = np.full(EXPERTS, -1)
    expert_in = np.full(capacity, -1)
    recent = np.zeros(capacity, np.int64)
    frequency = np.zeros(EXPERTS, np.int64)
    clock, next_halving, per_step, reads = 0, half_life, [], 0
    if rule == "optimal":
        future = defaultdict(list)
        for t, (_, uses) in enumerate(sequence):
            for e in uses:
                future[e].append(t)
        pointer = defaultdict(int)

    def victim(protected, t):
        free = expert_in < 0
        if free.any():
            return int(np.flatnonzero(free)[0])
        candidates = np.flatnonzero(recent != clock)
        candidates = candidates[~np.isin(expert_in[candidates], protected)]
        if not len(candidates):
            return None
        if rule == "lru":
            return int(candidates[np.argmin(recent[candidates])])
        if rule == "engine":
            keys = (frequency[expert_in[candidates]] << 32) | recent[candidates]
            return int(candidates[np.argmin(keys)])
        ahead = [
            future[e][pointer[e]] if pointer[e] < len(future[e]) else 1 << 30
            for e in expert_in[candidates]
        ]
        return int(candidates[int(np.argmax(ahead))])

    for t, (phase, uses) in enumerate(sequence):
        clock += 1
        if clock >= next_halving:
            frequency >>= 1
            next_halving = clock + half_life
        for e in uses:
            frequency[e] += 16
            if rule == "optimal":
                pointer[e] += 1
        # A prompt chunk loads its most-used experts; a check step needs all.
        wanted = (
            [e for e, _ in uses.most_common(capacity)] if phase == "P" else list(uses)
        )
        protected = np.array(wanted)
        step_reads = 0
        for e in wanted:
            if slot_of[e] >= 0:
                recent[slot_of[e]] = clock
                continue
            if phase == "D":
                step_reads += 1
                reads += 1
            slot = victim(protected, t)
            if slot is None:
                continue
            if expert_in[slot] >= 0:
                slot_of[expert_in[slot]] = -1
            expert_in[slot], slot_of[e], recent[slot] = e, slot, clock
        if phase == "D":
            per_step.append(step_reads)
    return per_step


def p95(reads):
    """Nearest-rank 95th percentile: a real step, not an interpolation."""
    if not len(reads):
        return 0.0
    values = np.sort(np.asarray(reads))
    return float(values[min(len(values) - 1, int(np.ceil(0.95 * len(values))) - 1)])


def summary(reads):
    """Mean and tail reads a check step, for one layer's step list."""
    if not reads:
        return 0.0, 0.0, 0.0
    values = np.asarray(reads)
    return (float(values.mean()), p95(values), float(values.max()))


def summed(per_layer):
    """Total reads across layers for each step (all layers run every step)."""
    return np.sum(np.asarray(per_layer), axis=0).tolist()


def greedy_split(curve, statistic, grid, budget, step):
    """Assign slot chunks one at a time to the layer each chunk helps most."""
    split = {layer: grid[0] for layer in curve}
    spent = grid[0] * len(curve)
    while spent + step <= budget:
        best = None
        best_gain = 0.0
        for layer, values in curve.items():
            here, next_size = split[layer], split[layer] + step
            if next_size not in values:
                continue
            gain = statistic(values[here]) - statistic(values[next_size])
            if gain > best_gain:
                best, best_gain = layer, gain
        if best is None:
            break
        split[best] += step
        spent += step
    return split


def totals_split(curve, grid, budget, step):
    """Fit against the whole check step, not a layer: a step is gated by its
    total reads across layers, so each candidate chunk is scored by the p95
    (worst steps) of the summed step series, not by one layer's own tail."""
    split = {layer: grid[0] for layer in curve}
    spent = grid[0] * len(curve)
    base = {
        layer: {c: np.asarray(v, dtype=np.int64) for c, v in values.items()}
        for layer, values in curve.items()
    }
    current = summed([base[layer][split[layer]] for layer in curve])
    while spent + step <= budget:
        best = None
        best_tail = p95(current)
        for layer, values in base.items():
            here, next_size = split[layer], split[layer] + step
            if next_size not in values:
                continue
            candidate = current - values[here] + values[next_size]
            tail = p95(candidate)
            if best is None or tail < best_tail:
                best, best_tail = (layer, candidate), tail
        if best is None:
            break
        split[best[0]] += step
        current = best[1]
        spent += step
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log")
    parser.add_argument(
        "--slots", type=int, default=272, help="today's slots per layer"
    )
    parser.add_argument(
        "--budget", type=int, default=0, help="total slots; default slots x layers"
    )
    parser.add_argument("--rules", default="engine,lru,optimal")
    parser.add_argument("--step", type=int, default=16, help="split granularity")
    args = parser.parse_args()

    events = load(args.log)
    # Layer 48 is the draft head's own layer: it has its own cache and is not
    # part of the check step.
    layers = [layer for layer in sorted(events) if layer < 48]
    steps = sum(1 for p, _ in events[layers[0]] if p == "D")
    budget = args.budget or args.slots * len(layers)
    grid = list(
        range(args.slots - 8 * args.step, args.slots + 12 * args.step + 1, args.step)
    )
    print(
        f"{steps} check steps, {len(layers)} layers; budget {budget} slots; "
        f"mean and p95 SSD reads a step\n"
    )
    for rule in args.rules.split(","):
        # A layer's step list for every candidate size: the same trace, so
        # the three plans are compared on identical text and routing.
        curve = {
            layer: {c: replay(events[layer], c, rule) for c in grid} for layer in layers
        }
        plans = {"uniform": {layer: args.slots for layer in layers}}
        plans["mean"] = greedy_split(curve, np.mean, grid, budget, args.step)
        plans["p95 (per layer)"] = greedy_split(curve, p95, grid, budget, args.step)
        plans["p95 (a step)"] = totals_split(curve, grid, budget, args.step)
        for name, split in plans.items():
            per_layer = [curve[layer][split[layer]] for layer in layers]
            total = summed(per_layer)
            mean, tail, worst = summary(total)
            spread = max(split.values()) - min(split.values())
            print(
                f"{rule:8s} {name:7s} mean {mean:5.1f}  p95 {tail:5.1f}  "
                f"max {worst:5.1f}  slots {min(split.values())}..{max(split.values())}"
                f" (spread {spread})"
            )
        print(
            "         p95 split: "
            + ",".join(str(plans["p95 (a step)"][layer]) for layer in layers)
        )
        print(
            "         mean split: "
            + ",".join(str(plans["mean"][layer]) for layer in layers)
        )


if __name__ == "__main__":
    main()
