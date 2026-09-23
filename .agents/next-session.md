# Next session — Slipstream (for Gemini or any coding agent)

Paste this into a new session started in
`~/Documents/shared-with-google-drive/model-serving/slipstream`.

---

You are continuing work on **Slipstream**, a lean fork of the Splash inference
engine that serves Qwen3.8-Flash-Next on this 64 GB Apple M5 Pro. Read, in order:
`.agents/status.md` (current state, rules, what is done and pending), the top 3
entries of `.agents/journal.md`, `.agents/memory.md`, `.agents/decisions.md`,
then `README.md` and `docs/profiling.md`.

**Write for Nitin in plain words** (`~/.agents/writing-style.md`): lead with the
point, short sentences, no jargon without a one-line explanation. When you need a
decision, give the situation, why it matters and the options as outcomes.

**Safety first.** Only one model server may run at a time; two freeze the Mac.
Check before starting anything:
`lsof -nP -iTCP -sTCP:LISTEN | grep -E ":(8000|8080|8090) "`. Run engine
commands through `dev/benchmarks/guarded.py -- <cmd>`. Never touch
`~/models/qwen38-flash-next-bf16`, `~/models/qwen38-flash-next-v3`, `../splash2/`.
Never push anywhere (no remote here; `../splash` has the upstream remote — do not
push to it).

## Task 1 — check what claude-code finished on 2026-09-23

`.agents/status.md` lists items marked FILL IN. For each one still marked, do
the step below; for the rest, trust the status and the journal.

### 1a. Merge the upstream fixes (if `git branch --list upstream-fixes` still exists)

```zsh
git merge --no-ff upstream-fixes            # on main; if status.md conflicts, keep main's
make -j10 && make build/engine-tests/generate-sample
dev/benchmarks/guarded.py --max-seconds 1500 -- make -j6 check-native-cpu check-native-metal
make check-python-engine check-source       # 383 + 170 + 41 tests, lint, architecture
```

Then the identical-output check (10 prompts, greedy; must print 10/10):
`models/qwen4exp/bench/identical/check.sh` (reference output is in that folder).

Finally remove the worktree: `git worktree remove ../slipstream-port && git branch -d upstream-fixes`.

### 1b. Live tool-call checks (server must be running: `~/models/bin/slipstream-server.sh`)

```zsh
omp --model splash-flashnext/local/qwen3.8-flash-next-splash -p "List the files in this folder using a tool, then reply DONE" < /dev/null
pi  --model slipstream/local/qwen3.8-flash-next-splash -p "List the files in this folder using a tool, then reply DONE" < /dev/null
~/models/bin/slipstream-log.sh   # each request must show a Done line, no Error
```

## Task 2 — finish the benchmark comparison

Folder: `~/Documents/shared-with-google-drive/benchmarking/model-quality-bench`
(its `README.md` and `BENCHMARKS.md` explain the harness). Runs land in
`runs/<model>/<suite>__<time>.jsonl`. All use `--seed 1234`, greedy.

What exists after 2026-09-23 (check `ls runs/*/`):
- `slipstream-flashnext`: all 9 suites (agreement mmlu mmlu_pro gsm8k math500
  humaneval ifeval needle sessions), if the run finished.
- `v3-gguf-text` (llama.cpp V3): the suites claude-code had time for.
- `splash-flashnext`: 6 suites from 2026-09-22 (reference for "identical").
- `splash-27b-hq`: none.

To run the missing suites for one model (one server at a time):

```zsh
# llama.cpp V3 on :8080 (keeps today's GPU limit, so no password; guarded)
CACHE=34 WIRED_MB=59392 ~/Documents/shared-with-google-drive/model-serving/slipstream/dev/benchmarks/guarded.py \
  --floor-gib 4 --max-seconds 64800 -- ~/models/bin/qwen-q40-server.sh &
./run_suites.sh v3-gguf-text math500 needle sessions      # whatever is missing; --resume skips done items
kill $(lsof -nP -iTCP:8080 -sTCP:LISTEN -t)

# Qwen3.8-27B HQ on :8000 (a different, dense model): see run_round_2.sh step 3
```

Compare, always passing run FILES (name lookup can pick `mmlu_pro` for `mmlu`):

```zsh
.venv/bin/python -m bench.compare --a runs/v3-gguf-text/gsm8k__<time>.jsonl --b runs/slipstream-flashnext/gsm8k__<time>.jsonl --suite gsm8k
.venv/bin/python -m bench.judge --a v3-gguf-text --b slipstream-flashnext --suite sessions   # Nitin judges, blind
```

Write the results into `BENCHMARKS.md` (a results section: per suite, score for
each model, paired difference, "not distinguishable" where p > 0.05) and the
Slipstream card on `~/Documents/shared-with-google-drive/INDEX.html` (back it up
first: `cp INDEX.html INDEX.html.bak-$(date +%Y%m%d-%H%M%S)`; after editing,
check the page's script still parses).

## After any work

Append to `.agents/journal.md` (newest first, heading `## <date time> — gemini`),
overwrite `.agents/status.md`, update `.agents/tasks.md`, commit with a plain
message. Keep `.agents/` the source of truth.
