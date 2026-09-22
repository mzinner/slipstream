# Memory — qwen4exp port

## Splash's geometry is compile-time

Model dimensions live in C++ layout structs, not a config file. A new
architecture therefore needs engine changes, not just a converter. This is the
fact that set the whole shape of this project.

## Where things are

| | |
|---|---|
| Source checkpoint | `~/models/qwen38-flash-next-bf16` — 338 GB, 131 shards, complete |
| Tensor name prefix | `model.language_model.` (the 27B uses `language_model.model.`) |
| Converter | `dev/tools/convert_qwen4exp.py` |
| 27B converter (format proof) | `dev/tools/convert_qwen38.py` |
| Verification scripts | `dev/tools/checks/` |
| Nitin's working V3 model | `~/models/qwen38-flash-next-v3` (GGUF, llama.cpp) |

## The file format is unforgiving, which is good

16-byte header, sections on 16384-byte boundaries, and the file must be consumed
**exactly** — `WeightFile::finish()` throws on a leftover byte. A missing or
mis-sized section fails loudly at load rather than quietly at inference.

Use `--dry-run` freely: it writes headers and leaves bodies as holes, so a
96.61 GiB package costs 920 KiB and the engine will still load, validate and
plan it. This caught a manifest constant guessed at 256 MiB where the engine
wanted 128 MiB, long before any real weights existed.

## Traps that cost real time

- **A synthetic test that agrees with itself proves nothing.** I reserved a
  section for `block_inject_weight` on the final hyper-connection mixer, which
  is built `use_combine=False` and has only three tensors. Every real package
  would have failed. My test wrote the same wrong layout and passed.
- **Confirm the build succeeded before interpreting a test result.** I read a
  stale binary's output twice and re-reasoned about a fix that was already
  correct.
- **N-gram table shape**: 320,001,536 x 160 (16 heads), not 20M x 2560. Same
  element count, so the 26.82 GiB total matched and hid the error.
- **`make` does not relink `generate-sample`.** Run
  `make build/engine-tests/generate-sample` after every engine change.
- **`/tmp` is not storage.** The bf16 reference logits took ~40 min to make;
  they now live in `~/models/qwen38-flash-next-reference/`.
- **Changing a shared constant breaks tests quietly in many places.** Raising
  the prompt chunk to 4096 needed ~12 test edits where 2048/2049 were
  hardcoded; tests should use the constants.
- **PLE gate offset**: normalized rows go at `row + (taps-1)*dilation`, because
  the convolution reads history first.

## About Nitin

- Runs this on an Apple-silicon laptop; disk is a recurring constraint.
- Prefers taking the smaller/simpler path first and deferring optimisation
  (chose a smaller model before expert streaming; deferred draft training).
- Wants plain language, conclusion first. See `/Users/nitin/.agents/writing-style.md`.
- Works across Claude Code, Codex and Gemini — keep this directory current.
