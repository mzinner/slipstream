# Status — qwen4exp (Qwen3.8-Flash-Next) in Splash

**Updated:** 2026-09-22 16:50 PDT by claude-code (speed round after two crashes)
**Branch:** `qwen4exp-review` (not pushed; never push to upstream `incoai/splash`)

## In one line

Quality matches llama.cpp V3, decode is 35–50 tok/s (llama.cpp: 18, or 27
with its draft head), and prompts read at ~180–670 tok/s (llama.cpp ~110–367)
— same 36 GiB expert budget.

## Numbers (one warm engine, 384 tokens, guess cap 5)

| Prompt | Greedy | Sampled 0.7 | Prompt reading |
|---|---|---|---|
| Short, 65 tok | 50.0 | 46.1 | 178–217 tok/s |
| Code, 1,495 tok | 37.9 | 35.4 | ~583 tok/s |
| Long, 9,129 tok | 36.3 | 34.4 | ~600 tok/s (2,048-token chunks; 609–666 at 4,096) |

Server, agent-style (4.6K-token system prompt, 5 tools): first token 8.3 s
cold, ~1 s with the prompt cached; tool calls correct greedy and sampled.

Quality on the code prompt (bf16 reference): 91% same top pick, KL 0.12
(second half 90%, 0.16). llama.cpp V3: 89%, 0.18.

## Package format now (no backward compatibility)

- Routed experts 4-bit (unchanged). Mixers, output head, embedding: 8-bit.
  Hyper-connection mix weights: 8-bit in the matrix kernels' tiled layout
  (down projection padded 320 -> 512 outputs), so prompts run them as matrix
  products and decode reads them with its own kernels.
- 4-bit copy of the output head, used only by the MTP draft.
- Magics: layer MDFN0031, head MDFN0034, embedding MDFN0013, MTP combiner
  MDFN0035. Manifest `prefill_token_budget` 2048 (built maximum 2048). 4096
  was tried: ~10% faster long prompts, but its 2.2 GB prompt scratch (vs 1.2)
  left too little memory headroom under the server, which then stalled
  requests with resource_timeout.
- `convert_qwen4exp.py --requantize-mixers` rewrites the previous layer
  version in place in under a minute (experts copied); `--head-only` rewrites
  head.bin. A package at an older magic must be stepped through in order.

## Speed round, 2026-09-22 afternoon (commits c4eb2a7, b7cad30)

10-prompt suite (`dev/benchmarks/qwen4exp/speed_suite.py`, 256 tokens each,
34 GiB cache): **greedy 36.4 -> 37.6 tok/s, sampled 35.2 -> 37.5.** Greedy
outputs identical to before on every run. The 50 tok/s goal was not reached;
see "What's left" below for why.

- **Waves on by default** (`SPLASH_DECODE_WAVES=0` off): cached experts run
  on the GPU while the host reads missing ones.
- **Draft head scores the 64K most common tokens** when
  `target/draft-vocab.bin` exists (`dev/tools/draft_vocab.py`). Draft time
  12.3 -> 8.4 ms a step, same tokens per step. 32K was worse (head became
  overconfident, more rejected guesses).
- **Crash guard:** the loader refuses an expert cache that does not fit in
  free memory with 2 GiB to spare. Two engines at once froze the Mac twice
  today. Run experiments via `dev/benchmarks/guarded.py -- <cmd>`.

## Where decode time goes now (10-prompt suite, ~81 ms a step, 3.08 tokens a step)

| Part | ms/step | Notes |
|---|---|---|
| GPU busy | ~45 | ~1,200 small dependent kernels; hyper-connection mix ~5.6 of it |
| Waiting on SSD reads | ~17 | 3.6% of expert uses miss the cache |
| Draft head | ~8.5 | was 12.3 |
| Other host work | ~10 | hand-offs, staging |

## What's left (measured today, largest first)

1. **SSD misses (~17 ms).** Eviction is already frequency+recency; the
   prefetch is tuned. Splitting slots unevenly across layers (offline replay
   of real routing, `SPLASH_ROUTE_LOG`) would cut misses ~9% (~1.5 ms). Not done.
2. **Hyper-connection kernels (~5.6 ms)** run at ~40% of memory speed because
   each reads only 3.3 MB. Paired outputs, skipping dead rows and deduplicated
   scale loads were all no faster. The real fix, merging down + up into one
   launch with a grid-wide wait, risks a GPU deadlock; not attempted.
3. **More tokens per step** is the only lever big enough for 50 tok/s
   (needs ~4.1 tokens a step at today's step time). Confidence cutoff and
   guess limit are already at the balance point (0.2-0.4, 4-6 all within
   0.2 tok/s). Would need tree drafting or a better draft head.
4. Draft head still processes 8 rows when 1 is live (~0.5 ms).
5. Earlier measured and still true: 8-bit linear kernels at the bandwidth
   floor; per-layer GPU<->host hand-offs ~8 ms are structural
   (probe `dev/benchmarks/qwen4exp/probes/handoff_latency.mm`).

## Running it day to day

- Server: `~/models/bin/splash-flashnext-server.sh` (port 8090, context pinned
  at 131,072, ready in ~15 s, logs in `~/models/logs/`). Stop llama.cpp first.
  It now sets the GPU memory limit to 58 GiB (`sudo`, once per boot): macOS
  resets it on reboot, and at the default ~52 GiB the server refuses to start
  ("kv_pool_does_not_fit").
- omp: provider `splash-flashnext` in `~/.omp/agent/models.yml` (window 126,976):
  `omp --model splash-flashnext/local/qwen3.8-flash-next-splash`.
- Thinking: the model's template defaults to xhigh. omp sends a level only
  because the entry has `compat: {qwenTemplateReasoningEffort: true}` (without
  it omp never sent one, so everything ran at xhigh). omp default is medium;
  `--thinking=low|medium|xhigh|off` (with `=`) or Shift+Tab in a session. The
  server honours `reasoning_effort`, `enable_thinking` and
  `chat_template_kwargs`, and logs `think <level>` per request. Changing level
  mid-session changes the prompt, so the next turn re-reads it once.
- Reference card in Nitin's hub: `~/Documents/shared-with-google-drive/INDEX.html`.

## Handoff

Next-session prompt: `.agents/next-session.md`. Reference logits (bf16 ground
truth) and passages: `~/models/qwen38-flash-next-reference/` (README there).
Agent-style server check: `dev/benchmarks/qwen4exp/agent_turn.py`.

## Tools

- `dev/benchmarks/qwen4exp/profile_steps.py`: per-step decode profile
  (tokens/step, step-time spread, staging/GPU/MTP split).
- `reference_logits.py --quant group:bits,...`: price any quantization
  choice against the bf16 reference (the simulation matched the engine to
  0.1%).
- `SPLASH_STEP_TIMING=1`, `SPLASH_PROFILE_STEPS=N`, `SPLASH_LOG_ALLOCATIONS=1`,
  `SPLASH_MTP_DRAFTS=1..7`, `SPLASH_PREFILL_ROW_SLICES`, `SPLASH_PROMPT_KEEP`.
- `dev/benchmarks/guarded.py -- <cmd>`: run any engine experiment safely
  (refuses beside another engine, kills under 4 GiB free or after a time limit).
- `SPLASH_DRAFT_VOCAB=N` (0 = all tokens), `SPLASH_DECODE_WAVES=0`,
  `SPLASH_ROUTE_LOG=path` (routing for offline cache replay),
  `SPLASH_HC_REPEAT=n` / `_PARTS` (price the hyper-connection kernels).

## Do not touch

- `~/models/qwen38-flash-next-bf16` (338 GB source), `~/models/qwen38-flash-next-v3`
  (Nitin's daily llama.cpp model), the `splash2/` checkout.

Write-up: https://claude.ai/artifact/PB7F2nj2KRtST91NfMGoEQ
