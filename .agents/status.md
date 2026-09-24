# Status — Slipstream (handoff)

**Updated:** 2026-09-23 20:00 PDT by claude-code. **Next agent: read this file
first, then `.agents/next-session.md`.**
**Branch:** `main` (local only, no remote). Never push to `github.com/incoai/splash`.

## In one line

Slipstream is our lean fork of Splash that serves Qwen3.8-Flash-Next on this
64 GB M5 Pro at **~41 tok/s** (about 2x llama.cpp V3), and it is the engine to
use. **Its quality equals llama.cpp V3's** on every paired test run (below).

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
| Instructions (`ifeval`, 541) | 87.8% | not run yet | — |

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

1. **Finish the benchmark comparison** that did not fit today: llama.cpp on
   `humaneval` (started 20:10 on 2026-09-23; may be partial, resume it),
   `ifeval` and `math500`; the 27B on all; `sessions` on both, then Nitin
   judges blind. Update `BENCHMARKS.md` and the hub after each suite.
2. **Decisions for Nitin** (see next-session.md, Task 2): default cache 34 vs
   30 GiB; full-precision reference via a hosted API; the 27B round.
3. Optional: drop `install/` and Homebrew packaging; explicit model interface.

**Done today (2026-09-23 evening), no longer pending:** upstream fixes merged
(2e4a4c8: #31 composed tool schemas, #92 required-first tool arguments, #120
per-request `timings`; all gates, 10/10 identical at 39.7 tok/s); quieter log
built and checked (0 developer lines); live tool calls through omp and pi both
worked on the merged build.

## Known issues

- **The model sometimes thinks until the length cap** (both engines; a model
  habit, worst at temperature 0): 1-8% of benchmark answers. In omp/pi it shows as
  a very long thinking phase with no answer; a lower thinking level or asking
  again gets past it. Details and counts: BENCHMARKS.md, round 2 notes.

- A second git worktree exists at `.kilo/worktrees/magical-antimony` (made by
  the Kilo Code VS Code extension, not by claude-code). Leave it unless Nitin says.
- `/metrics` `splash_prefill_tokens_per_second` reads far too high (GPU work
  only); use the per-request `prompt … tok/s` in the log.
- Google Drive syncing this folder uses a full core and makes timed runs ~3%
  slower; alternate A/B runs when measuring speed.
