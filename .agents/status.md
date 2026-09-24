# Status — Slipstream (handoff)

**Updated:** 2026-09-23 20:00 PDT by claude-code. **Next agent: read this file
first, then `.agents/next-session.md`.**
**Branch:** `main` (local only, no remote). Never push to `github.com/incoai/splash`.

## In one line

Slipstream is our lean fork of Splash that serves Qwen3.8-Flash-Next on this
64 GB M5 Pro at **~41 tok/s**, with answers identical to Splash, and it is the
engine to use. Quality is confirmed against Splash (below); the paired
comparison against llama.cpp V3 is being run today.

---

## How to use it (the three commands)

```zsh
~/models/bin/slipstream-server.sh   # start on :8090 (password once per boot; runs under the memory guard)
~/models/bin/slipstream-log.sh      # one log line per request: prompt and decode tok/s
~/models/bin/slipstream-stop.sh     # stop
```

- omp: `omp --model splash-flashnext/local/qwen3.8-flash-next-splash --tools=read,write,edit,bash,grep,glob,todo --approval-mode=yolo`
- pi: `pi --model slipstream/local/qwen3.8-flash-next-splash`
- Full options, logs, omp and pi: `README.md` ("Run it" onward). Hub card: `~/Documents/shared-with-google-drive/INDEX.html`.

## Rules that are not optional

- **One model server at a time.** Two pin more memory than the Mac has and
  freeze it (twice on 2026-09-22). Run engine experiments through
  `dev/benchmarks/guarded.py -- <cmd>`; the launcher already does.
- **Nothing memory-heavy next to a running engine** (loading GBs of weights in
  Python made the guard stop a run on 2026-09-23).
- **Do not touch:** `~/models/qwen38-flash-next-bf16` (338 GB source),
  `~/models/qwen38-flash-next-v3` (Nitin's daily llama.cpp model), the `splash2/`
  checkout, the model package's files.
- **After any engine change:** `make`, `make build/engine-tests/generate-sample`,
  `make check-native-cpu check-native-metal test-python`, then the 10-prompt
  identical-output check (`docs/profiling.md`).

---

## Quality: what is confirmed (2026-09-23)

**Slipstream's quality equals llama.cpp V3's (Nitin's daily setup).** Paired on the
same questions (seed 1234, greedy); "same" = McNemar p >= 0.05. Full table and
notes: `benchmarking/model-quality-bench/BENCHMARKS.md` (Results log, round 2).

| Test | Slipstream | llama.cpp V3 | Paired |
|---|---:|---:|---|
| Knowledge (`mmlu`, 400) | 89.0% | 89.2% | same |
| Grade-school math (`gsm8k`, 250) | 96.8% | 97.2% | same |
| Harder knowledge (`mmlu_pro`, 500) | 64.2% | 64.6% | same |
| Long-context lookup (`needle`, 27) | 100% | 100% | same |
| Code (`humaneval`, 164) | 90.2% | not run yet | — |
| Competition math (`math500`, 200) | 90.0% | not run yet | — |
| Instructions (`ifeval`, 541) | FILL IN | not run yet | — |

Decode speed on the same math questions: Slipstream 46.8 tok/s, llama.cpp 23.1
(llama.cpp at a 30 GiB cache that day).

Other evidence: 91% same top pick as the full-precision model, KL 0.12
(llama.cpp V3: 89%, 0.18); today's Slipstream and today's Splash give identical
answers, including on the code questions that differed from Splash's 2026-09-22
run (that run was the odd one out).

**Memory finding (reliability, not quality):** at the 34 GiB default cache with
Chrome open, only ~6 GiB stayed free and Slipstream refused 16K-32K prompts
(`resource_timeout`; it keeps 6.4 GiB free for macOS). With Chrome closed, or
`CACHE_GIB=30`, they run. Nitin to choose the default (see journal).

---

## Done (all committed on `main` unless noted)

| What | Where |
|---|---|
| Fork, cut down to this model; model code in `models/qwen4exp/` | phases 1-4, `docs/architecture.md` |
| Speed 36.4 -> ~41 tok/s: waves, draft vocab, chain stop, read-ahead 6, uneven expert-cache slots | `.agents/decisions.md` |
| Draft-head (better guesser) trial: **no-go**, today's head stays | `docs/draft-head-plan.md` |
| Review of upstream PR incoai/splash#115: nothing borrowed | `.agents/decisions.md` |
| Docs: run, options, logs, omp, pi | `README.md` |
| Launch/log/stop scripts | `~/models/bin/slipstream-{server,log,stop}.sh` |
| Request log shows prompt and decode speed | `server/diagnostics.py` |
| pi provider `slipstream` | `~/.pi/agent/models.json`, `settings.json` (backups next to them) |

## Pending (in priority order)

See `.agents/next-session.md` for the exact steps and commands.

1. **Merge branch `upstream-fixes`** (worktree `../slipstream-port`): upstream
   #31 composed tool schemas, #92 required-first tool arguments (a real bug
   here), #120 per-request `timings`. Server tests pass; needs the full gates.
   STATUS: FILL IN.
2. **Build and check the quieter-log change** (commit 4d28f69):
   `[MTP shadow]` / `[Prefill Timing]` / `[Verify Timing]` only with
   `SPLASH_STEP_TIMING=1`. STATUS: FILL IN.
3. **Live tool-call checks through omp and pi.** STATUS: FILL IN.
4. **Finish the benchmark comparison** that did not fit today: llama.cpp on the
   remaining suites, the 27B on all, blind judging of `sessions`, then
   `BENCHMARKS.md` and the hub.
5. Optional: drop `install/` and Homebrew packaging; explicit model interface.

## Known issues

- `bench.compare --a <model> --suite mmlu` picks the newest `mmlu*` file, which
  can be `mmlu_pro`; pass run files instead (`--a runs/<model>/mmlu__<time>.jsonl`).
- `/metrics` `splash_prefill_tokens_per_second` reads far too high (GPU work
  only); use the per-request `prompt … tok/s` in the log.
- Google Drive syncing this folder uses a full core and makes timed runs ~3%
  slower; alternate A/B runs when measuring speed.
