# Status — qwen4exp (Qwen3.8-Flash-Next) in Splash

**Updated:** 2026-09-22 (review) by claude-code
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

## Where decode time goes (code prompt, ~95 ms a step, ~3.7 tokens a step at cap 5)

GPU ~46 ms (8-bit matrix kernels at the bandwidth floor; expert kernels ~70%);
SSD miss reads ~17 ms; per-layer GPU<->host hand-offs ~8 ms (structural: 164 us
each, probe `dev/benchmarks/qwen4exp/probes/handoff_latency.mm`); MTP draft
~13 ms (mostly real GPU work).

## Open work (measured gaps, largest first)

1. ~~8-bit linear decode kernels~~: measured at the bandwidth floor (3.4 GB
   of 8-bit weights a step in 10.8 ms); every tile/grid variant is slower.
2. **SSD miss reads** ~17 ms/step: lookahead foresees 66% of used experts
   (top-16 would 79%). A second prediction two layers ahead was tried and
   rejected (misses 59 -> 56 at best, tok/s down: its reads compete).
3. **GPU<->host hand-offs** ~8 ms/step (48 x ~164 us): structural. Routing
   can't move to the GPU without a mid-command-buffer host->GPU signal, and
   Metal doesn't make host writes visible there (probe: handoff_latency.mm).
4. **Drafting** ~13 ms/step: mostly real GPU work (~2 ms a guess, half of it
   the 318 MB draft output layer). Merging its two round trips per guess
   would save ~1 ms/step; shrinking the draft vocabulary would cost more in
   acceptance than it saves.
5. Expert quantization: group-32 and range-search simulated *worse* than the
   current format despite lower weight error - unexplained, worth a look.
6. 27B regression check needs a 4-bit 27B package.

## Running it day to day

- Server: `~/models/bin/splash-flashnext-server.sh` (port 8090, context pinned
  at 131,072, ready in ~15 s, logs in `~/models/logs/`). Stop llama.cpp first.
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

## Do not touch

- `~/models/qwen38-flash-next-bf16` (338 GB source), `~/models/qwen38-flash-next-v3`
  (Nitin's daily llama.cpp model), the `splash2/` checkout.

Write-up: https://claude.ai/artifact/PB7F2nj2KRtST91NfMGoEQ
