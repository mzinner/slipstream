# Slipstream

**A lean inference engine for running one big model very fast on one Mac, and a
playbook for doing it again with the next model.**

Slipstream serves Qwen3.8-Flash-Next (the `qwen4exp` architecture: 48 layers, 512
experts per layer, ~100 GB of weights) on a 64 GB Apple M5 Pro. The weights do not
fit in memory, so the experts stream from the SSD while a small draft head guesses
tokens ahead. That is the name: a slipstream is the low-drag wake a racing car rides
in ("drafting"); here the model drafts ahead and the weights stream behind.

It began as a copy of [Splash](../splash) (Apache-2.0, see LICENSE) and keeps only
what this model needs.

---

## Where it stands (2026-09-23)

| | |
|---|---|
| **Model** | Qwen3.8-Flash-Next, package `~/models/qwen38-flash-next-splash` (shared with Splash, unchanged) |
| **Speed** | ~41 tok/s greedy on the 10-prompt suite and on held-out coding sessions (Splash: 36.4 on 2026-09-22) |
| **Quality** | Greedy output identical to Splash token for token; 91% same top pick as the bf16 reference |
| **Removed from Splash** | two other models, the kernel tuner, the vision encoder, the unused DFlash draft (~19,300 lines, 1.45 GB less memory) |
| **Layout** | Everything specific to this model is in `models/qwen4exp/`; the rest is shared (`docs/architecture.md`) |
| **Now** | Benchmark round 2: Slipstream vs llama.cpp V3 vs Qwen3.8-27B (`.agents/status.md`) |

## Words we can't avoid

| Term | Meaning here |
|---|---|
| **expert** | One of 512 small feed-forward blocks per layer; each token uses 10 |
| **expert cache** | The experts kept in memory (272 per layer, 34 GiB); the rest are read from the SSD when needed |
| **draft head (MTP)** | A small extra layer in the model that guesses the next few tokens; the full model then checks them all in one step |
| **step** | One pass of the full model that checks the guesses and keeps the right ones |
| **package** | The converted model on disk: 4-bit experts, 8-bit everything else |

---

## Run it

**Three commands cover daily use: start, watch, stop.** The server speaks the
OpenAI and Anthropic APIs on `http://127.0.0.1:8090`, and omp and pi are already
set up for it.

```zsh
~/models/bin/slipstream-server.sh      # start (foreground; ~15 s to load)
~/models/bin/slipstream-log.sh         # in another tab: one line per request, with speeds
~/models/bin/slipstream-stop.sh        # stop, and see the memory it gave back
```

Starting asks for your password once per boot: it raises macOS's GPU memory limit
to 58 GiB. It refuses to start if another model server (llama.cpp, Splash, the 27B)
is running; two big models do not fit in 64 GB.

### Start options

Set these in front of the command, e.g. `CTX=65536 ~/models/bin/slipstream-server.sh`.

| Setting | Default | What it does | When to change it |
|---|---|---|---|
| `CTX` | `131072` | Longest conversation, in tokens | Lower (e.g. `65536`) to leave more memory for other apps |
| `CACHE_GIB` | `34` | Memory for experts kept in RAM; the rest are read from the SSD | `30` if you skip the password step (macOS's default GPU limit); a bigger cache is faster but risks freezing the Mac |
| `PORT` | `8090` | Port to serve on | Only if 8090 is taken; then omp needs the new port too |
| `WIRED_MB` | `59392` | macOS GPU memory limit it sets (needs `sudo` when it changes) | Leave it |
| `GUARD` | `1` | Runs under the memory guard: stops the server if free memory falls under 4 GiB instead of letting macOS freeze | `GUARD=0` only when debugging |
| `LOG` | `~/models/logs/slipstream-<date>.log` | Where the log goes | Rarely |
| `REPO` | this folder | Which engine checkout to run | To run Splash instead: `REPO=~/…/model-serving/splash` |

### Engine settings (speed only; answers never change)

These tune how the engine works. The model's output is the same whatever you set;
only speed changes. The defaults are the measured best. `SPLASH_` is the engine's
historical prefix.

| Setting | Default | What it does |
|---|---|---|
| `SPLASH_NO_MTP=1` | off | Turns off the draft head (the guesser): one token per step, much slower. For comparisons only |
| `SPLASH_MTP_CHAIN_MIN` | `0.35` | Stop guessing when the guesses' combined confidence falls below this |
| `SPLASH_LOOKAHEAD_EXPERTS` | `6` | Experts per row read ahead from the SSD, from a prediction. Tested 4–12; 5–8 tie |
| `SPLASH_EXPERT_SLOTS` | built-in profile | How expert memory is split across layers. `even` = the same for every layer (the old way) |
| `SPLASH_DRAFT_VOCAB` | `65536` | Tokens the draft head scores (the most common ones); `0` = all 248K |
| `SPLASH_DECODE_WAVES=0` | on | Turns off running cached experts while missing ones are read (slower) |

Measurement settings (per-step timing, traces, routing logs, profiling) are in
[docs/profiling.md](docs/profiling.md).

---

## Watch it: logs and speeds

**One line per request tells you the speed.** `slipstream-log.sh` follows the
newest log and shows just those lines, plus loading, memory and error events:

```
08:22:03 Done · input 333 · cached 64 · output 812 · think low · TTFT 1.3s · prompt 207 tok/s · decode 40.8 tok/s
```

| Field | Meaning |
|---|---|
| `input` | Prompt tokens in this request |
| `cached` | Of those, how many were reused from an earlier turn (not processed again) |
| `output` | Tokens written |
| `think` | Thinking level used (`default` = the chat template's, which is the highest) |
| `TTFT` | Time to first token, including any wait in the queue |
| `prompt … tok/s` | **Prompt speed**: new (uncached) prompt tokens per second |
| `decode … tok/s` | **Writing speed**: tokens per second after the first one |

Other lines you may see: `Loading`/`Ready` (startup), `Memory: growth paused` /
`growth available` (the server holding back context memory while it is tight;
normal), `Error · <code>` (a failed request), `guarded: killed …` (the memory guard
stopped the server).

**More detail when you need it:**

```zsh
~/models/bin/slipstream-log.sh all     # everything the server prints
curl -s localhost:8090/status | python3 -c "import json,sys; d=json.load(sys.stdin); print('ready', d['ready'], '| memory', d['memory_pressure'], '| waiting', d['admission']['waiting'])"
curl -s localhost:8090/metrics | grep -E "^splash_(decode|ttft|scheduler_(queued|decoding|prefilling))"
SPLASH_STEP_TIMING=1 ~/models/bin/slipstream-server.sh   # per-step timing lines (developer)
```

Averages over the whole run are in `/metrics` (`splash_decode_output_tokens_total`
over `splash_decode_wall_milliseconds_total` gives writing speed). Its
`splash_prefill_tokens_per_second` counts GPU work only and reads far too high;
use the per-request `prompt` figure instead.

---

## Use it from omp

**Nothing to set up: omp's `splash-flashnext` provider already points here** (port
8090, model id `local/qwen3.8-flash-next-splash`, defined in `~/.omp/agent/models.yml`).
Slipstream and Splash serve the same id, so whichever is running answers.

**The optimized command (the one to copy):**

```zsh
omp --model splash-flashnext/local/qwen3.8-flash-next-splash --tools=read,write,edit,bash,grep,glob,todo --approval-mode=yolo
```

| Flag | What it does |
|---|---|
| `--tools=read,write,edit,bash,grep,glob,todo` | Only these 7 tools. omp describes every enabled tool in its system prompt, so fewer tools means a shorter prompt and a faster first turn |
| `--approval-mode=yolo` | Runs tools without asking first. Use it only in folders where that is safe |
| `--thinking=low` (optional) | Less thinking, faster answers; also `off`, `medium` (default), `xhigh`. Note the `=` |

Other forms:

```zsh
omp --model splash-flashnext/local/qwen3.8-flash-next-splash --thinking=low --tools=read,write,edit,bash,grep,glob,todo --approval-mode=yolo
omp --model splash-flashnext/local/qwen3.8-flash-next-splash            # plain: all tools, asks before writes
omp --model splash-flashnext/local/qwen3.8-flash-next-splash -p "Reply with exactly: OK" < /dev/null   # quick check
```

- **Thinking levels:** `low`, `medium` (omp's default for this model), `xhigh`;
  `off` turns thinking off. The log's `think` field shows what the server got.
- **Context:** omp is told 126,976 tokens, 4K under the server's 131,072. If you
  start with a smaller `CTX`, lower `contextWindow` in `models.yml` to match.
- **Check it is connected:** run the quick check above, then look for its `Done`
  line in `slipstream-log.sh`.

---

## Use it from pi

**pi has a `slipstream` provider for this server** (added 2026-09-23 in
`~/.pi/agent/models.json`; model `local/qwen3.8-flash-next-splash`, port 8090). It
is also in pi's Ctrl+P list. pi's default model is still llama.cpp (`flashnext`).

```zsh
pi --model slipstream/local/qwen3.8-flash-next-splash                   # interactive
pi --model slipstream/local/qwen3.8-flash-next-splash --thinking low    # less thinking, faster answers
pi --model slipstream/local/qwen3.8-flash-next-splash -p "Reply with exactly: OK" < /dev/null   # quick check
```

- **Thinking levels:** `off`, `low`, `medium`, `xhigh` (the model's template has
  no others, so pi hides `minimal`, `high` and `max`). pi sends them as
  `reasoning_effort`; the log's `think` field shows what arrived.
- **Make it pi's default:** in `~/.pi/agent/settings.json` set `"defaultProvider":
  "slipstream"` and `"defaultModel": "slipstream/local/qwen3.8-flash-next-splash"`.
- **Context:** 126,976 tokens, as for omp. Lower `contextWindow` in `models.json` if
  you start the server with a smaller `CTX`.

---

## Other clients

Any OpenAI-compatible client works too. Chat responses (and the last chunk of a
stream) carry llama.cpp-style `timings`: `prompt_n`, `cache_n`, `prompt_per_second`
(new prompt tokens only), `predicted_n`, `predicted_per_second`.

```zsh
curl -s localhost:8090/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "local/qwen3.8-flash-next-splash",
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 200, "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

---

## Build, test, stay safe

- **Build:** `make` (engine), `make build/engine-tests/generate-sample` (test tool;
  `make` alone does not relink it).
- **Tests:** `make check-native-cpu check-native-metal test-python`, then the
  identical-output check in [docs/profiling.md](docs/profiling.md).
- **Memory:** without the password step (macOS's default GPU limit) start with
  `CACHE_GIB=30`; 34 does not leave room for context and the server refuses to start.
- **Safety rule:** run every engine experiment through
  `dev/benchmarks/guarded.py -- <cmd>` (the launcher does this for you). Two
  engines at once pin more memory than the Mac has and freeze it until its watchdog
  restarts it (this happened twice on 2026-09-22).

---

## Read next

| Document | For |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How the pieces fit: Metal backend, kernels, model, engine, server |
| [docs/new-model-playbook.md](docs/new-model-playbook.md) | Bringing up the next model, step by step, with the checks that catch mistakes |
| [docs/profiling.md](docs/profiling.md) | Every measurement tool: what it tells you and how to read it |
| [docs/draft-head-plan.md](docs/draft-head-plan.md) | Why a better guesser was tried and dropped (2026-09-23), and what it would take |
| [.agents/status.md](.agents/status.md) | Current state and next steps (shared with other coding agents) |
