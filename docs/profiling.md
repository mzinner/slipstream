# Measuring Slipstream

**Start every speed exercise with one traced run and its report.** It shows where a
step's time goes, how well guessing works, and what other guessing strategies would
have given, all from the same run. Then use the narrower tools below to dig into one
part.

```zsh
S=/tmp/run; P=$(python3 -c "import json;p=json.load(open('build/qwen4exp-speed-prompts.json'));print(';'.join(','.join(map(str,v)) for v in p.values()))")
echo "$P" > $S-prompts.txt
SPLASH_TRACE=$S-trace.jsonl SPLASH_EXPERT_CACHE_GIB=34 dev/benchmarks/guarded.py -- \
  build/engine-tests/generate-sample build/splash.metallib ~/models/qwen38-flash-next-splash 256 "$P" > $S-out.jsonl
models/qwen4exp/bench/trace_report.py $S-trace.jsonl $S-out.jsonl $S-prompts.txt
```

---

## The tools

| Tool | Answers | Cost |
|---|---|---|
| `models/qwen4exp/bench/speed_suite.py` | **Headline tok/s** over 10 varied prompts (greedy, or `--temperature 0.7`) | ~4 min |
| `SPLASH_TRACE=path` + `trace_report.py` | **Time per part, acceptance by depth, a cost model, and a replay of other guessing strategies** | one run |
| `SPLASH_STEP_TIMING=1` | The same timing split as one stderr line per step | free |
| `SPLASH_PROFILE_STEPS=N` | **GPU time per kernel** over N steps (each kernel in its own command, so slower than real) | one run |
| `SPLASH_HC_REPEAT=n` (`_PARTS`) | Cost of the hyper-connection kernels, by running them n times (answers unchanged) | one run |
| `SPLASH_ROUTE_LOG=path` + `cache_plan.py` | Which experts each layer picked; **replays eviction rules and an uneven split of the same memory**, and prints the split (`SPLASH_EXPERT_SLOTS`) | one run + ~1 min |
| `SPLASH_ROUTE_LOG=path` + `simulate_cache.py` | Older replay: whole-cache sizes and policies | one run |
| `[Verify Timing]` lines (`SPLASH_STEP_TIMING=1`) | **"predicted" is a running count of read-ahead reads**; with "misses" it gives total SSD reads a step | free |
| `profile_steps.py` | Slowest steps and what they have in common | one run |
| `compare_logits.py` | **Quality**: top-pick agreement and KL against the bf16 reference | ~5 min |
| `check_decode_consistency.py` | Does decoding agree with prompt processing on the same text | ~2 min |
| `bench_llama.py` / `bench_splash.py` | Same prompts, same token ids, on llama.cpp and here | ~10 min |
| `dev/benchmarks/guarded.py -- <cmd>` | Runs anything above safely (see README) | — |

## Reading the trace report

**1. Time per step.** Each line is milliseconds per step, averaged. "Waiting for SSD
reads" is time the GPU sat idle for missing experts; "GPU busy" is real work.

**2. Guessing.** "Rate" per depth is how often a guess at that depth was right, given
the guesses before it were. The confidence table checks whether the draft head's own
confidence means what it says (0.9 confident should be right ~90% of the time).

**3. Cost model.** Step time is fitted as `a + b × rows checked`. On qwen4exp, **each
extra row costs ~7.35 ms**, mostly ~16 more experts read from the SSD. This single
number decides which guessing ideas can pay off.

**4. Strategies.** Replays ideas at the same points of the same text. Greedy output
is fixed, so "what the model would have said" is known exactly. Measured accuracy of
this replay: it predicted 39.5 tok/s for the chain-confidence stop; the real run gave
39.1.

## SSD reads: the number to watch

A check step reads experts from the SSD in two ways:
- **read-ahead**: experts predicted for the next layer, read while the GPU works;
- **on demand**: experts the prediction missed; the GPU waits for these.

Both use the same SSD bandwidth. On 2026-09-23 the step read **87 ahead + 46 on
demand**; a third of the read-ahead was wasted (wrong guesses). Narrowing
read-ahead to each row's top 6 predicted experts and splitting cache slots
unevenly across layers brought it to **37 + 50**, +4% tok/s, same output.
Replays (`cache_plan.py`) count reads without read-ahead, so use them to rank
plans, then confirm with the engine.

## Pitfalls

- **Noise is ±2% run to run** (SSD timing). Do not believe a 1% gain from one run.
  For kernels, amplify: repeat the kernel so its cost dominates.
- **Tracing, logging and timing flags can slow a run.** Compare like with like.
- **The first steps after loading are cold.** The 10-prompt suite runs on one warm engine.
- **Compiling during a timed run skews it.** Do not build while measuring.
- **Background apps skew it too.** Google Drive syncing this folder took a full
  core and the SSD on 2026-09-23 (runs 3-4% slower). Alternate A and B runs.

## Where the time went on 2026-09-22 (qwen4exp, 10-prompt suite)

| Part | ms per step |
|---|---|
| GPU busy (checking step) | ~41 |
| Waiting for SSD reads | ~15 |
| Draft head | ~7 |
| Host work between stages, rest | ~12 |
| **Total** | **~75 (2.83 tokens a step → ~38–39 tok/s)** |

Ceiling with this draft head and perfect stopping: ~45 tok/s. Beyond that needs a
draft head that guesses better, or cheaper rows (fewer SSD reads per extra row).
