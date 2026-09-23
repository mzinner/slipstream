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
| `SPLASH_ROUTE_LOG=path` + `simulate_cache.py` | Which experts each layer picked; **replays cache sizes and policies offline** | one run |
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

## Pitfalls

- **Noise is ±2% run to run** (SSD timing). Do not believe a 1% gain from one run.
  For kernels, amplify: repeat the kernel so its cost dominates.
- **Tracing, logging and timing flags can slow a run.** Compare like with like.
- **The first steps after loading are cold.** The 10-prompt suite runs on one warm engine.
- **Compiling during a timed run skews it.** Do not build while measuring.

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
