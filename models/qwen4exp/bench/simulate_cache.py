#!/usr/bin/env python3
"""Replay logged expert routing against expert-cache sizes and policies.

Input: SPLASH_ROUTE_LOG lines "P|D layer e0..e9" in execution order. Each
layer has its own cache. Reported: misses per generated token, which at the
measured ~15 GB/s SSD rate and 2.76 MB per expert is a disk-time floor.

Policies:
  lru          Splash today: prompt reading and generation share one LRU
  lru-bypass   prompt reading does not enter the cache; generation-only LRU
  lfu          keep the most-used experts (counts decay slowly)
  optimal      Belady: evict the expert needed furthest in the future; the
               best any policy could do with this cache size
"""

import sys
from collections import OrderedDict, defaultdict

EXPERT_MB, SSD_GBPS = 2.76, 15.0


def load(path):
    events = defaultdict(list)  # layer -> [(phase, set)]
    last = None
    for line in open(path):
        parts = line.split()
        phase, layer, experts = parts[0], int(parts[1]), set(map(int, parts[2:]))
        if phase == "P" and last == (phase, layer) and events[layer]:
            events[layer][-1][1].update(experts)  # one prompt chunk = one event
        else:
            events[layer].append((phase, experts))
        last = (phase, layer)
    return events


def simulate(events, capacity, policy):
    misses = 0
    for layer, seq in events.items():
        if policy == "optimal":
            future = defaultdict(list)
            for t, (phase, s) in enumerate(seq):
                for e in s:
                    future[e].append(t)
            pointer = defaultdict(int)
        cache, counts = OrderedDict(), defaultdict(float)
        for t, (phase, experts) in enumerate(seq):
            if policy == "lru-bypass" and phase == "P":
                continue
            for e in experts:
                counts[e] = counts[e] * 0.999 + 1
                if policy == "optimal":
                    pointer[e] += 1
            for e in experts:
                if e in cache:
                    cache.move_to_end(e)
                    continue
                if phase == "D":
                    misses += 1
                if len(cache) >= capacity:
                    # A prompt chunk can need more experts than fit; then
                    # nothing is protected and the oldest goes.
                    keep = experts if len(experts) < capacity else ()
                    if policy in ("lru", "lru-bypass"):
                        victim = next(x for x in cache if x not in keep)
                    elif policy == "lfu":
                        victim = min(
                            (x for x in cache if x not in keep), key=lambda x: counts[x]
                        )
                    else:

                        def next_use(x):
                            uses = future[x]
                            i = pointer[x]
                            return uses[i] if i < len(uses) else 1 << 30

                        victim = max((x for x in cache if x not in keep), key=next_use)
                    del cache[victim]
                cache[e] = True
    return misses


events = load(sys.argv[1])
tokens = sum(1 for p, _ in events[0] if p == "D")
print(f"{tokens} generated tokens replayed, 48 layers, 10 experts each\n")
print(
    f"{'slots/layer':>11} {'GiB':>5}  "
    + "  ".join(f"{p:>22}" for p in ("lru", "lru-bypass", "lfu", "optimal"))
)
for capacity in (128, 192, 256, 291, 384):
    gib = capacity * 48 * EXPERT_MB * 1e6 / 2**30
    cells = []
    for policy in ("lru", "lru-bypass", "lfu", "optimal"):
        m = simulate(events, capacity, policy) / tokens
        ms = m * EXPERT_MB / 1000 / SSD_GBPS * 1000
        cells.append(f"{m:6.1f} miss {ms:5.1f} ms")
    print(
        f"{capacity:>11} {gib:5.1f}  " + "  ".join(f"{c:>22}" for c in cells),
        flush=True,
    )
