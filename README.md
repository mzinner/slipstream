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

## Where it stands (2026-09-22)

| | |
|---|---|
| **Model** | Qwen3.8-Flash-Next, package `~/models/qwen38-flash-next-splash` (shared with Splash, unchanged) |
| **Speed** | ~39–40 tok/s greedy on the 10-prompt suite (Splash: 36.4 at the start of the day); 45 tok/s on short answers |
| **Quality** | Greedy output identical to Splash token for token; 91% same top pick as the bf16 reference |
| **Removed from Splash** | two other models, the kernel tuner, the vision encoder, the unused DFlash draft (~19,300 lines, 1.45 GB less memory) |
| **Layout** | Everything specific to this model is in `models/qwen4exp/`; the rest is shared (`docs/architecture.md`) |
| **Next** | a better draft head, to push past ~45 tok/s |

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

```zsh
# Server on :8090 (OpenAI and Anthropic APIs). Asks for your password once per boot
# to raise macOS's GPU memory limit to 58 GiB.
REPO=~/Documents/shared-with-google-drive/model-serving/slipstream \
  ~/models/bin/splash-flashnext-server.sh
```

Build: `make` (engine), `make build/engine-tests/generate-sample` (test tool).
Tests: `make check-native-cpu check-native-metal test-python`.

**Memory:** the launcher raises macOS's GPU memory limit to 58 GiB (`sudo`, once
per boot). Without that, use a smaller expert cache: `CACHE_GIB=30` still starts
with the full 128K context and costs a little speed. At the default limit, a 34 GiB
cache leaves too little room and the server refuses to start.

**Safety rule:** run every engine experiment through `dev/benchmarks/guarded.py -- <cmd>`.
Two engines at once pin more memory than the Mac has and freeze it until its
watchdog restarts it (this happened twice on 2026-09-22).

---

## Read next

| Document | For |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How the pieces fit: Metal backend, kernels, model, engine, server |
| [docs/new-model-playbook.md](docs/new-model-playbook.md) | Bringing up the next model, step by step, with the checks that catch mistakes |
| [docs/profiling.md](docs/profiling.md) | Every measurement tool: what it tells you and how to read it |
| [docs/draft-head-plan.md](docs/draft-head-plan.md) | The plan toward 50 tok/s: a block guesser trained on your own sessions |
| [.agents/status.md](.agents/status.md) | Current state and next steps (shared with other coding agents) |
