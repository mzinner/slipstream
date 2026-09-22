# Next session — Qwen3.8-Flash-Next in Splash

Paste the prompt below into a new Claude Code session started in this folder.

---

Continue the Qwen3.8-Flash-Next (qwen4exp) work in this Splash repo, on
branch `qwen4exp-review`.

First read, in order: `.agents/status.md` (current numbers, package format,
measured open gaps), the top 3 entries of `.agents/journal.md`, then
`.agents/memory.md` and `.agents/decisions.md`.

Goal: improve quality and speed as far as possible at llama.cpp's memory budget
(36 GiB expert cache, wired limit 58 GiB). Breaking compatibility or redesigning
is fine if it measures better. Profile before changing anything; after each
change, check that quality holds and output stays exact, then benchmark,
commit, and note what worked and what was rejected (with numbers) in `.agents/`.

Where things stand: quality 91% same top pick / KL 0.12 on the code prompt
(llama.cpp V3: 89% / 0.18); generating 34–50 tok/s; prompts 178–666 tok/s.

Open gaps, from status.md:
1. Finer expert rounding (group 32, error-minimizing ranges) scored WORSE on
   quality despite lower weight error. Find out why; it may point to a better
   format.
2. SSD miss reads ~17 ms/step (lookahead foresees 66% of used experts).
3. Drafting ~13 ms/step; merging the draft head's two GPU round trips per
   guess would save ~1 ms.
4. 27B regression check needs a 4-bit 27B package.

How to check work:
- Build: `make -j12 && make build/engine-tests/generate-sample` (the second
  is not rebuilt by the first).
- Tests: `make check-native-cpu` and `make check-native-metal` (both must pass).
- Speed: `.venv/bin/python dev/benchmarks/qwen4exp/bench_splash.py --tokens 384
  --one-process` (add `SPLASH_TEMPERATURE=0.7` for sampled);
  per-step breakdown: `dev/benchmarks/qwen4exp/profile_steps.py`.
- Quality: reference logits and passages live in
  `~/models/qwen38-flash-next-reference/` (see its README);
  `reference_logits.py --quant ...` simulates a quantization choice.
- Exactness: greedy output with drafting must equal `SPLASH_NO_MTP=1` output.
- Server + tools: start the server (command in status.md), then
  `.venv/bin/python dev/benchmarks/qwen4exp/agent_turn.py`.

Do not touch: `~/models/qwen38-flash-next-bf16` (338 GB source),
`~/models/qwen38-flash-next-v3` (my daily llama.cpp model), the `splash2/`
checkout. Do not push. Stop llama.cpp's server only if it's running and you
need the memory.

Write for me in plain words: lead with the result, short sentences, small
tables.
