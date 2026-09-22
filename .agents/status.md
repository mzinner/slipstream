# Status — qwen4exp port

**Updated:** 2026-09-22 02:05 PDT by claude-code
**Branch:** `qwen4exp-review` (not pushed; never push to upstream `incoai/splash`)

## Where we are in one line

Flash-Next runs in Splash at **33–45 tok/s** (llama.cpp V3: 18, or 27 with its
draft head), same memory budget. The server works end to end, tool calls included.

## Numbers (one warm engine, 384 generated tokens, 36 GiB expert cache)

| Prompt | Greedy | Sampled 0.7 | Reading the prompt |
|---|---|---|---|
| Short, 65 tok | 45.1 tok/s | 41.5 | 87 tok/s |
| Code, 1,495 tok | 35.2 | 36.0 | 222 tok/s |
| Long, 9,129 tok | 37.9 | 33.3 | 211 tok/s |

Through the server: tool requests 33–52 tok/s; first token after a new
339-token tool prompt 3–4 s, 0.2 s when the prompt is cached.

## Quality

- Short text: perplexity 8.93 vs bf16 reference 8.93.
- Code prompt second half: 79% same top pick, KL 0.51 (llama.cpp 89%, 0.18).
  Drift grows with position; cause open. Sparse indexer unwired (>2K context).

## Open work, in order

1. **Code-prompt drift** (quality). Compare layer by layer with
   `reference_logits.py` at late positions.
2. **Short-prompt reading speed** 87 tok/s vs llama.cpp ~110.
3. **Sparse indexer** for contexts past 2,048 tokens.
4. 27B regression check needs a 4-bit 27B package: the installed 27B is
   `splash-packed-q8`, which only the `splash2` fork reads (main and this
   branch both reject it), so this branch cannot affect it today.

## Tools (all in repo)

- `generate-sample`: `SPLASH_PROFILE_STEPS=N`, `SPLASH_PROFILE_PREFILL=1`,
  `SPLASH_DUMP_PREFILL_LOGITS=path`, `SPLASH_TEMPERATURE`; several prompts
  separated by `;`. **Build it explicitly** (`make build/engine-tests/generate-sample`).
- `SPLASH_LOG_ALLOCATIONS=1`: every Metal allocation ≥ 1 MiB.
- `dev/benchmarks/qwen4exp/`: `bench_splash.py --one-process`, `bench_llama.py`,
  `reference_logits.py`, `compare_logits.py`, `simulate_cache.py`, `probes/`.
- Server: `.venv/bin/python -m server.server $M/target $M/draft --tokenizer $M/tokenizer
  --model local/qwen3.8-flash-next-splash --port 8090 --binary build/splash`
  with `M=~/models/qwen38-flash-next-splash`.

## Do not touch

- `~/models/qwen38-flash-next-bf16` — 338 GB source.
- `~/models/qwen38-flash-next-v3` — the model Nitin runs daily (llama.cpp, port 8080).
- `splash2/` checkout — serves the 27B.

Full write-up: https://claude.ai/artifact/PB7F2nj2KRtST91NfMGoEQ
