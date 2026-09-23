# Status — Slipstream

**Updated:** 2026-09-22 20:00 PDT by claude-code
**Branch:** `main` (local only; no remote. Never push to `incoai/splash`.)

## In one line

Slipstream is Splash cut down to Qwen3.8-Flash-Next (qwen4exp) and on its way to
being a framework for the next model. Greedy output is identical to Splash; speed
~38–39 tok/s on the 10-prompt suite; ~14,100 lines removed so far.

## Done

| Phase | What | Commit |
|---|---|---|
| 1 | Dropped Qwen3.8 dense, Qwen3.6 MoE, the kernel tuner | 7160200 |
| 2 | Text only: vision encoder, image protocol (v6), image cache keys, image server code; PDFs read as text | 114d979 |
| docs | README, docs/architecture.md, docs/new-model-playbook.md, docs/profiling.md | (this commit) |

Checks after each phase: native CPU and Metal gates, Python 383 + 168 + 41, 10/10
identical greedy outputs vs Splash on the real model, server smoke test.

## Next

1. **Phase 3 — remove the placeholder DFlash draft.** Every DFlash path is already
   guarded by `descriptor.draftPlaceholder`, so keeping only the placeholder branch
   preserves behaviour. It reaches: `Runtime.mm` (context prefill/commit, draft batch
   graph, draft rings), `DFlashDraft.*`, `DraftContextPlan.*`, `ops/DraftAttention.*`,
   `draft.metal`, `draft_context.metal`, state layout (draft rings per state cell),
   arenas (Draft* tensors, captured hidden rows), descriptor (draft layout,
   capabilities), memory plan (draft weights), engine (draft context stats,
   checkpoint spacing), tests (dflash/draft attention/selector/context plan).
   Keep the shared verify path: proposed tokens, acceptance, retained rows (MTP uses it).
   Expected gain: ~0.5–1 GB memory → a larger expert cache.
2. **Phase 4 — framework layout.** `core/` (metal, ops, engine, runtime),
   `models/qwen4exp/` (layout, loader, forward, kernels, converter, tools); an explicit
   model interface in place of `QwenTarget`; allow model folders to launch their own
   kernels in `check_architecture.py`; drop Homebrew/release packaging and installer;
   fix the ~100 lint errors in one-off scripts (or delete the scripts).
3. Launcher: `~/models/bin/splash-flashnext-server.sh` serves Slipstream with
   `REPO=<this folder>`; give Slipstream its own launcher once it replaces Splash.

## Speed toward 50 tok/s

Measured ceiling with this draft head and perfect stopping: ~45.6 tok/s. Each row
checked costs ~7.35 ms (~16 extra SSD expert reads). Trees, runner-up guesses and
prompt lookup all lose at that price. 50 needs a better-guessing draft head
(training) or cheaper rows. Details: Splash `.agents/decisions.md`, trace tools in
`docs/profiling.md`.

## Do not touch

`~/models/qwen38-flash-next-bf16` (338 GB source), `~/models/qwen38-flash-next-v3`
(Nitin's daily llama.cpp model), the `splash2/` checkout, the shared package's files.
