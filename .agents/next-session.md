# Handoff to Gemini (or any coding agent) — Slipstream, 2026-09-23 night

Paste this into a new session started in
`~/Documents/shared-with-google-drive/model-serving/slipstream`.

---

You are continuing work on **Slipstream**, a lean fork of the Splash inference
engine that serves Qwen3.8-Flash-Next on this 64 GB Apple M5 Pro. Claude Code
did the work so far and is not available tomorrow.

Read, in order: `.agents/status.md` (current state, rules, done and pending),
the top 3 entries of `.agents/journal.md`, `.agents/memory.md`,
`.agents/decisions.md`, then `README.md`. The benchmark results are in
`~/Documents/shared-with-google-drive/benchmarking/model-quality-bench/BENCHMARKS.md`
("Results log", round 2).

**Write for Nitin in plain words** (`~/.agents/writing-style.md`): lead with the
point, short sentences, no jargon without a one-line explanation. When you need a
decision, give the situation, why it matters and the options as outcomes. Share
each result as soon as it lands; Nitin is curious about quality.

## Safety rules (non-negotiable)

- **Only one model server at a time.** Two freeze the Mac. Before starting one:
  `lsof -nP -iTCP -sTCP:LISTEN | grep -E ":(8000|8080|8090) "` must print nothing.
- Start servers only through the launchers below; they run under the memory
  guard (`dev/benchmarks/guarded.py`), which stops a server if free memory falls
  under 4 GiB. If the guard stops something, lower its cache; never lower the floor.
- Nothing memory-heavy (loading GBs in Python) next to a running server.
- Never touch `~/models/qwen38-flash-next-bf16`, `~/models/qwen38-flash-next-v3`,
  `../splash2/`. Never push anywhere.
- Chrome uses ~6 GB. With Chrome open, use `CACHE_GIB=30` for Slipstream and
  `CACHE=26` for llama.cpp.

## Servers

```zsh
# Slipstream (:8090)
CACHE_GIB=30 ~/models/bin/slipstream-server.sh      # foreground; add & / nohup to background
~/models/bin/slipstream-log.sh                      # per-request speeds
~/models/bin/slipstream-stop.sh

# llama.cpp V3 (:8080), keeping today's GPU limit (no password)
CTX=40960 CACHE=26 WIRED_MB=59392 nohup \
  ~/Documents/shared-with-google-drive/model-serving/slipstream/dev/benchmarks/guarded.py \
  --floor-gib 4 --max-seconds 64800 -- ~/models/bin/qwen-q40-server.sh > /tmp/llama.log 2>&1 &
kill $(lsof -nP -iTCP:8080 -sTCP:LISTEN -t)          # stop
```

## Task 1 — finish what is marked pending in `.agents/status.md`

Each item there says DONE or what is left. The likely leftovers:

- **Benchmark: llama.cpp on the suites it has not run** (code `humaneval` may be
  partial; `ifeval` and `math500` not run). In the benchmark folder:
  `./run_suites.sh v3-gguf-text humaneval` (then `ifeval`, `math500`).
  `--resume` is built in: finished answers are kept, failed ones retried.
  Each suite takes hours on llama.cpp (answers that loop to 8,192 tokens take
  ~6 min each). After each suite:
  `.venv/bin/python -m tools.round_summary --ref slipstream-flashnext --models v3-gguf-text`
  and update the round-2 table in `BENCHMARKS.md`.
- **Upstream fixes**, if `git branch --list upstream-fixes` still exists: see
  "Merging upstream-fixes" below.

## Task 2 — decisions to put to Nitin

1. **Default expert cache: 34 GiB (fastest) or 30 GiB?** At 34 with Chrome open,
   Slipstream refused 16K-32K-token prompts (it keeps 6.4 GiB free for macOS).
   At 30 they work. Measure the speed cost first:
   `models/qwen4exp/bench/identical/check.sh SPLASH_EXPERT_CACHE_GIB=30` vs `=34`
   (prints tok/s; alternate runs twice each), then ask with the numbers.
2. **Full-precision reference scores?** Nobody has run the unquantized model on
   these questions. Cheapest route: a hosted API serving Qwen3.8-Flash-Next,
   added to `models.yaml` (needs Nitin's API key and a few dollars).
3. **The 27B comparison** (`splash-27b-hq`, a different dense model): worth the
   ~6-8 hours of machine time?
4. **Blind judging of `sessions`** needs Nitin: both engines must first answer
   the suite (`./run_suites.sh <model> sessions`), then
   `.venv/bin/python -m bench.judge --a v3-gguf-text --b slipstream-flashnext`.

## Merging upstream-fixes (only if still pending)

```zsh
git merge --no-ff upstream-fixes          # on main; if .agents/status.md conflicts, keep main's
make -j10 && make build/engine-tests/generate-sample
dev/benchmarks/guarded.py --max-seconds 1500 -- make -j6 check-native-cpu check-native-metal
make check-python-engine check-source
models/qwen4exp/bench/identical/check.sh  # must print 10/10
git worktree remove ../slipstream-port && git branch -d upstream-fixes
```

Then a live tool-call check with the server running:

```zsh
omp --model splash-flashnext/local/qwen3.8-flash-next-splash -p "List the files in this folder using a tool, then reply DONE" < /dev/null
pi  --model slipstream/local/qwen3.8-flash-next-splash -p "List the files in this folder using a tool, then reply DONE" < /dev/null
```

## After any work

Append to `.agents/journal.md` (newest first, heading `## <date time> — gemini`),
overwrite `.agents/status.md`, update `.agents/tasks.md`, commit with a plain
message. Update the Slipstream card in
`~/Documents/shared-with-google-drive/INDEX.html` (back it up first; after
editing, check its script still parses: extract the `<script>` block and run
`node --check`).
