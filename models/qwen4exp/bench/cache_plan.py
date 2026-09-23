#!/usr/bin/env python3
"""How many experts each check step must read from the SSD, for a given expert
cache plan - replayed offline from a routing log, so every plan is compared on
the same text.

Record a log (SPLASH_ROUTE_LOG=path on generate-sample), then:

  models/qwen4exp/bench/cache_plan.py routes.log [--slots 272] [--budget 13056]

Reports, per eviction rule:
  - SSD reads per check step with the same number of slots in every layer;
  - the same with the slots split unevenly (more where a layer's reads fall
    fastest), under the same total;
  - the split itself (for SPLASH_EXPERT_SLOTS).

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
    """SSD reads during check steps for one layer."""
    half_life = int(rule.split(":")[1]) if ":" in rule else HALF_LIFE
    rule = rule.split(":")[0]
    slot_of = np.full(EXPERTS, -1)
    expert_in = np.full(capacity, -1)
    recent = np.zeros(capacity, np.int64)
    frequency = np.zeros(EXPERTS, np.int64)
    clock, next_halving, used, reads = 0, half_life, 0, 0
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
        for e in wanted:
            if slot_of[e] >= 0:
                recent[slot_of[e]] = clock
                continue
            if phase == "D":
                reads += 1
            slot = victim(protected, t)
            if slot is None:
                continue
            if expert_in[slot] >= 0:
                slot_of[expert_in[slot]] = -1
            expert_in[slot], slot_of[e], recent[slot] = e, slot, clock
        used += 1
    return reads


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
    print(f"{steps} check steps, {len(layers)} layers; budget {budget} slots\n")
    for rule in args.rules.split(","):
        curve = {
            layer: {c: replay(events[layer], c, rule) for c in grid} for layer in layers
        }
        uniform = sum(curve[layer][args.slots] for layer in layers)
        # Greedy split: start everyone at the grid's floor, then give each next
        # chunk of slots to the layer whose reads drop most.
        split = {layer: grid[0] for layer in layers}
        spent = grid[0] * len(layers)
        while spent + args.step <= budget:
            best = max(
                (layer for layer in layers if split[layer] + args.step in curve[layer]),
                key=lambda layer: (
                    curve[layer][split[layer]] - curve[layer][split[layer] + args.step]
                ),
                default=None,
            )
            if best is None:
                break
            split[best] += args.step
            spent += args.step
        uneven = sum(curve[layer][split[layer]] for layer in layers)
        print(
            f"{rule:8s} same slots everywhere: {uniform / steps:5.1f} reads a step   "
            f"uneven split: {uneven / steps:5.1f} ({100 * (uneven - uniform) / uniform:+.1f}%)"
        )
        print("         split: " + ",".join(str(split[layer]) for layer in layers))


if __name__ == "__main__":
    main()
