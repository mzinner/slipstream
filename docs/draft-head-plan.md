# Plan: a better guesser, toward 50 tok/s

> **Result (2026-09-23): step 3 said no-go. Today's draft head stays.** A block
> guesser trained on this Mac from 3M tokens of your sessions guesses far worse
> than the head the model shipped with. Scored at the same points of the same
> text (the model's own greedy output, replayed through the same cost model):
>
> | Text | Today's head | Trained block guesser |
> |---|---|---|
> | Held-out sessions (17 cut points) | **3.28 tokens a step, 38.9 tok/s** | 1.55, 25.2 tok/s |
> | 10-prompt suite | **2.82, 39.8 tok/s** | 1.16, 23.9 tok/s |
>
> First guess right: 84% for today's head, 56% for the trained guesser (still
> rising slowly). **Why:** the built-in head was trained by Qwen on vastly more
> text; 3M tokens is not enough to catch up from scratch. Two early design bugs
> were found and fixed on the way (a 1024-wide bottleneck, and no start from the
> model's own latest state), which took the guesser from 32% to 56%; they are not
> the reason for the gap. What was built stays useful: the engine's recording
> mode, the corpus, the compressed store, the trainer, and the report's
> proposal scoring. Details: "What we learned" at the end.

**The point:** 50 tok/s needs a guesser that is right about **9 times in 10 at each
of 5–7 guesses in a row**. Today's draft head is right 8 in 10 on the first guess,
falling to 6 in 10 by the fourth. That is already as good as the best published
retraining of this kind of head, so tuning it further buys little. The one kind
of guesser that reaches 9 in 10 guesses a whole block of tokens at once (DFlash).
It is trained on text written by the model itself. You have ~4.6M such tokens in
your own coding sessions. The plan: train our own block guesser on this Mac, with
a cheap go/no-go check before any engine work.

**Quality cannot drop.** The full model still checks every guess with the same
rule as today. A better guesser changes how many tokens each step keeps, never
which tokens come out. We still re-check that on every change (see "Quality").

## Words we can't avoid

| Term | Meaning here |
|---|---|
| **guess** (draft) | A token proposed ahead of time; the full model keeps it only if it agrees |
| **row** | One position the full model checks in a step: the last real token plus each guess |
| **draft head** | Today's guesser: one small extra layer, run once per guess |
| **block guesser** | Proposed: a few small layers that guess 8 tokens in one pass, reading the full model's inner state |
| **inner state** (features) | The full model's hidden vectors at a few layers; the guesser's main input |

---

## Where we are (10-prompt suite, greedy, 2026-09-22)

| | |
|---|---|
| Speed | 39.8 tok/s, 2.82 tokens per step, 70.8 ms per step |
| **Cost of one more row** | **~8.9 ms**, mostly ~20 more experts read from the SSD |
| Draft head, right given earlier guesses right | 0.82 · 0.72 · 0.67 · 0.60 · 0.61 (guesses 1–5) |
| Same, all guesses so far right | 0.82 · 0.59 · 0.40 (guesses 1–3) |
| FastMTP (best published retraining of a head like ours) | 0.80 · 0.56 · 0.36 |

**Why rows are so expensive here:** half the experts live on the SSD. Every extra
row wakes ~20 experts that must be read, at SSD speed. On a GPU server a row costs
almost nothing, which is why long guesses pay off there and not here.

## What it takes to reach 50 (our measured cost model)

| Guesser right each time (given earlier right) | Today's kind of head | Block guesser (one ~6 ms pass) |
|---|---|---|
| 8 in 10 | 39.5 tok/s | 40.6 |
| 8.5 in 10 | 43.2 | 45.2 |
| **9 in 10** | 48.3 | **51.9** |
| 9.5 in 10 | 56.7 | 62.3 |

Cheaper rows also help: today's head with rows at 6 ms instead of 8.9 would give
~44.6 tok/s. That is a separate track (below).

---

## What published work says

| Work | Idea | What it means for us |
|---|---|---|
| [Flash-dLLM](https://huggingface.co/papers/2609.26796) (the paper you sent) | Diffusion models guess and check their own tokens; a KV-cache kernel built to cut memory traffic | Our model writes one token at a time, so the method doesn't carry over. The lesson does: **design around memory traffic**, which for us is SSD reads |
| [DFlash](https://arxiv.org/abs/2602.06036) | Small block guesser reading 5 layers of the model's inner state; 16 tokens in one pass | **Right ~9 in 10** on math/code, ~8 in 10 on chat (Qwen3-8B). Trained on ~800K answers written by the target, on H200s. No guesser exists for our model |
| [FastMTP](https://arxiv.org/abs/2509.18362), [MTP-D](https://arxiv.org/abs/2603.23911) | Retrain the built-in head on the model's own outputs | Reaches about where our head already is |
| [Limits of Speculation in MoE](https://arxiv.org/html/2609.22156) | Measured cost model for checking guesses on mixture-of-experts models | Same shape as ours; **stop guessing when the next row's cost exceeds its expected gain.** Best average ~2.8 guesses |
| [EcoSpec](https://arxiv.org/abs/2607.12696) | Prefer guesses whose experts are already loaded | Up to 1.62× on big MoE models; an idea for later |
| [DraftExpert](https://arxiv.org/abs/2607.24434) | Experts on slow storage: small guesser + read the target's experts early | 86–88% of early reads were right; 1.45× on small MoE models |

---

## The plan

Each step ends with a check. The go/no-go check (step 3) comes before any engine
work, so a guesser that doesn't beat today's head costs us days, not weeks.

| # | Step | Done when | Machine time |
|---|---|---|---|
| 1 | **Collect text.** The model's own replies from `~/.omp/agent/sessions` (~4.6M tokens, with their conversation), as token ids through the real chat template; keep 5% of sessions aside for testing | Token files plus a count by source | minutes |
| 2 | **Record the inner state.** A prompt-only engine mode writes 3 layers of the full model's inner state, plus the model's own top pick at each position | Output matches a normal run token for token; disk use as planned (~25–35 GB, 71 GB free) | **~8–10 h, engine running** |
| 3 | **Train and judge offline.** An 8-token block guesser (3 small layers; the model's own embedding and 64K-token output layer, frozen), trained with MLX on the Mac | **Go/no-go:** on the held-out sessions, the cost model predicts ≥ 10% more tok/s than today's head. Otherwise stop and keep today's head | ~10–20 h |
| 4 | **Put it in the engine.** Guesser weights and kernels in `models/qwen4exp/`; check the first k guesses, where k is chosen by the guesser's confidence and the row cost | Identical greedy output on the 10-prompt suite; speed suite; trace report | — |
| 5 | **Quality round.** The paused benchmark round (Slipstream vs llama.cpp V3 vs 27B HQ) | Scores unchanged within noise | — |

**Separate track: cheaper rows.** It doesn't depend on the guesser, so it can run
alongside.
- Uneven expert-cache slots per layer: ~9% fewer SSD reads in replay.
- Fewer hand-offs between GPU and CPU: ~16 ms of host work per step today.
- Early reads of experts while the guesser runs, DraftExpert-style.

## Quality

- **Why output can't change:** the checking rule is untouched. Greedy keeps a guess
  only if it equals the model's own top pick. Sampling keeps it with the model's own
  probability and resamples otherwise. Either way the output follows the model's own
  choices exactly.
- **Checked anyway, after every engine change:**
  - identical greedy output on the 10-prompt suite;
  - top-pick agreement and KL against the bf16 reference (`compare_logits.py`);
  - the server smoke test;
  - finally, the benchmark round.

## Honest risks

- **Little data.** 4.6M tokens is ~1% of DFlash's training set. It is your real
  workload, which helps, but 9 in 10 is not assured. My educated guess, not a
  measurement: 42–47 tok/s from this path alone.
- **Text written by the 4-bit llama.cpp version, sometimes sampled**, not by this
  engine's greedy output. Step 2 records the model's own top pick at every position
  to correct for that.
- **The Mac is busy during step 2** (~40 GiB pinned for ~8–10 h). llama.cpp V3
  can't run at the same time; the memory guard refuses to start a second engine.
- **If step 3 fails:** renting GPUs to train on far more text is the next lever.
  It costs money and needs the 338 GB model uploaded, so it would be your call.

---

## What we learned (2026-09-23)

| Step | Outcome |
|---|---|
| 1. Collect text | 5.6M model-written tokens from 237 sessions; the model's own top pick matches the session text on 73% of those tokens (they were written by llama.cpp's 4-bit version, sometimes sampled) |
| 2. Record the inner state | 3.0M training positions + 0.3M held-out in ~2.7 h (~330 positions/s on 60K-token sessions), 23 GB. Recording is byte-identical run to run, and off by default |
| 3. Train and judge | No-go (table at the top). Losses kept falling slowly: this is a data-size limit, not a ceiling of the design |

**Mistakes worth not repeating:**
- **Nothing memory-heavy next to the engine.** Loading 2.5 GB of weights during a
  recording made the guard stop the engine (it did its job).
- **Test the training pipeline with a task whose answer is in the input** (copy
  the model's own last pick). It exposed the width bottleneck at once.
- **Guessers must work at the model's width and start from its latest state**:
  32% → 56% on the first guess.

**What would change the answer:** 20–100× more text written by this model
(DFlash used ~800K answers), which on this Mac means weeks of generation. Renting
GPUs could do it in days, at a cost; see status.md for the choice.
