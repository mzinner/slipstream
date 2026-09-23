# Bringing up a new model

**The rule that matters most: prove each piece against the original model before
building on it.** Every serious bug in the qwen4exp port was a piece that looked
right, agreed with its own test, and was wrong. The steps below are ordered so
each one has a check that gives the same verdict every time before the next starts.

This playbook is distilled from porting Qwen3.8-Flash-Next (qwen4exp). File names
point at that port; copy its pattern.

**Where the code goes:** everything for the new model lives in `models/<name>/`
(layout, loader, forward pass, own kernels, converter, checks, benchmarks). It then
registers itself in four shared files; `docs/architecture.md` lists them.

---

## Words we can't avoid

| Term | Meaning |
|---|---|
| **reference** | The original model run in full precision (bf16) on the CPU: the ground truth |
| **top-pick agreement** | How often our engine's most likely next token equals the reference's |
| **KL** | How far our next-token probabilities are from the reference's (0 = identical) |
| **package** | The converted, quantized weights on disk that the engine maps |
| **layout** | A C++ struct of the model's fixed sizes, checked against the package |

---

## The steps

| # | Step | Done when | qwen4exp example |
|---|---|---|---|
| 1 | **Read the model** | You can list every tensor, its shape, and the order the forward pass uses them | HF config + reference code; tensor prefix `model.language_model.` |
| 2 | **Make the reference** | Full-precision next-token logits for a few fixed prompts, saved outside `/tmp` | `models/qwen4exp/bench/reference_logits.py` → `~/models/qwen38-flash-next-reference/` (~40 min) |
| 3 | **Write the layout** | A struct of constants with compile-time checks of derived sizes | `models/qwen4exp/Qwen4Exp.hpp`, test `qwen4exp_layout_test.cpp` |
| 4 | **Write the converter** | `--dry-run` writes a full-size package of headers only, and the engine loads and plans it | `models/qwen4exp/tools/convert_qwen4exp.py` (a 97 GiB package costs 920 KiB dry) |
| 5 | **New kernels** | Each new operation matches a numpy/torch version on random inputs | `hyper_connection_metal_test.mm`, `per_layer_embedding_metal_test.mm` |
| 6 | **One layer at a time** | Layer 0's output matches the reference layer 0 on real weights, then layer N | `models/qwen4exp/tools/checks/check_layer0_trace.py`, `check_layer_composition.py` |
| 7 | **Whole model** | Top-pick agreement and KL against the reference on real prompts | `compare_logits.py`: 91% same top pick, KL 0.12 (llama.cpp V3: 89%, 0.18) |
| 8 | **Memory plan** | Starts with the real package at the target context length, with margin | Experts streamed from the SSD; 34 GiB expert cache |
| 9 | **Decode speed** | Measured per part, with a draft method | See `docs/profiling.md`; MTP draft head |
| 10 | **Server** | Chat template, thinking levels and tool calls work through a real client | `agent_turn.py`; omp |
| 11 | **Quality benchmarks** | Paired against the best existing engine on the same prompts | `benchmarking/model-quality-bench` |

---

## Checks that catch the mistakes that actually happened

- **A test that agrees with itself proves nothing.** A synthetic package written by
  the same wrong assumption as the loader passed its test; every real package would
  have failed. Compare against the reference, or against the checkpoint's own tensors.
- **Totals can hide wrong shapes.** The n-gram table is 320,001,536 × 160, not
  20M × 2560; same element count, so the size check passed. Check shapes, not bytes.
- **Make the file format unforgiving.** Sections on fixed boundaries, and loading
  must consume the file exactly. A wrong section then fails at load, not as quietly
  wrong answers.
- **Keep greedy output as a regression check.** After any speed change, the greedy
  tokens on the 10-prompt suite must be identical. It is the cheapest check there is.
- **Confirm the build succeeded before reading a result.** A stale binary looks
  exactly like a working fix. `make` does not relink `generate-sample`; build it
  explicitly.
- **Tests should use named constants**, not the numbers they happen to have today.

## Machine safety (non-negotiable)

- Run every engine experiment through `dev/benchmarks/guarded.py -- <cmd>`.
  One engine pins ~40 GiB; two freeze a 64 GB Mac.
- Never start a background run until the previous one has provably exited.
- The model loader refuses an expert cache that does not fit in free memory.
- After a reboot, raise the GPU memory limit (`sudo sysctl iogpu.wired_limit_mb=59392`).

## Speed work, in the order that paid off for qwen4exp

1. **Get a baseline and a breakdown** before changing anything (`docs/profiling.md`).
2. **Draft method.** A draft head built into the model (MTP) beat a separately
   trained draft. Tune when to stop guessing with the trace simulator, not by feel.
3. **Hide waiting behind work.** Run cached experts while the missing ones load.
4. **Shrink what is read every step.** The draft head scores only the 64K most
   common tokens (`models/qwen4exp/tools/draft_vocab.py`).
5. **Measure small kernel changes by amplifying them** (repeat the kernel 8×);
   run-to-run noise is ±2%.

Things that did not pay off here are written down in `.agents/decisions.md`, so
the next model starts from them instead of repeating them.
