# How Slipstream fits together

**The short version:** a Python server turns chat requests into token ids and
talks to one native engine process over a pipe. The engine schedules requests,
and the model code turns each step into GPU work. Five layers, each allowed to
depend only on the ones below it (`dev/tools/check_architecture.py` enforces this):

```
 server/            Python: HTTP APIs, chat template, tools/grammar, PDF text
   │  binary protocol over a pipe (server/protocol.py <-> runtime/engine/Protocol.cpp)
 runtime/engine/    requests, scheduling, KV prefix cache, memory plan and governor
 runtime/model/     the model: package loading, forward pass, drafting, per-request state
 runtime/ops/       one C++ entry point per kernel family; picks the kernel variant
 runtime/metal/     GPU backend: buffers, command graphs, submission; kernels/*.metal
```

---

## What each layer owns

| Layer | Owns | Key files |
|---|---|---|
| **server** | OpenAI and Anthropic request shapes, chat template, thinking levels, tool-call grammar, streaming | `api_shapes.py`, `frontend.py`, `backend.py`, `runtime.py`, `protocol.py` |
| **engine** | Admitting requests within memory, batching, prompt chunks, reusing cached prompt prefixes, status | `Engine.cpp`, `Scheduler.cpp`, `Cache.cpp`, `KvCache.cpp`, `MemoryPlan.cpp`, `MemoryGovernor.cpp` |
| **model** | Reading the package, the forward pass, draft guesses, the expert cache, per-request recurrent state | `Qwen4Exp.*`, `Qwen4ExpTarget.*`, `Runtime.mm`, `QwenState.*`, `ModelDescriptor.mm` |
| **ops** | Launching kernels with the right sizes and variant | `Linear.*`, `MoE.*`, `PagedAttention.*`, `GDN.*`, `Sampling.*` |
| **metal** | Talking to the GPU: allocations, pipelines, events, pipelined submission | `MetalBackend.mm`, `CommandGraph.hpp`, `kernels/`, `abi/` (structs shared by C++ and Metal) |

**Startup** is assembled in one place: `runtime/engine/RuntimeResources.mm`
(load package → plan memory → allocate) and `Bootstrap.mm`.

---

## Shared versus model-specific

| Shared by any model in this family | Specific to qwen4exp |
|---|---|
| Metal backend, command graphs, pipelined submission | Layout and package format (`Qwen4Exp.hpp`, magic `MDFN0031`) |
| 4-bit and 8-bit matrix kernels, paged 8-bit KV attention | Hyper-connection kernels (4 residual streams) |
| Linear-attention (GDN) kernels and recurrent state | Per-layer n-gram embedding (PLE) kernels |
| MoE routing, grouped expert tiles | Expert streaming: per-layer cache, SSD reads, "waves" |
| Sampling and verification of guesses | MTP draft head, draft vocabulary, `mtp_pick` kernel |
| Engine, memory plan, prefix cache, server | Converter `dev/tools/convert_qwen4exp.py` |

## Where a model plugs in (today)

1. **Layout**: a struct of constants checked against the package manifest (`Qwen4ExpLayout`).
2. **Descriptor**: the package format name maps to a layout and a validator (`ModelDescriptor.mm`).
3. **Weights and loader**: `loadQwen4ExpWeights`, one alternative in `TargetWeights` (`ModelFactory.hpp`).
4. **Forward pass**: `Qwen4ExpTarget::addPrefill / addVerify / addHead / addEmbedding`,
   reached through the front class `QwenTarget`, which the runtime calls.
5. **Kernels**: Metal files plus parameter structs in `metal/abi/`.
6. **Converter**: checkpoint → package (`dev/tools/convert_qwen4exp.py`, `package_format.py`, `quantize.py`).

**Honest limit:** the runtime (`Runtime.mm`, state, arenas) assumes the Qwen hybrid
family: linear-attention and full-attention layers, an 8-bit paged KV cache, and a
recurrent state per request. A model from this family plugs in at the six points
above. A model outside it (for example pure attention with a different cache) also
needs the runtime's buffers and state generalized. That work is not done.

---

## One decode step, end to end

```
draft head guesses up to 5 tokens (stops early when unsure)       ~7 ms
  └ each guess: its own attention layer + experts + 64K-token output layer
full model checks anchor + guesses in one pass                    ~65 ms
  └ per layer: router → pick experts → cached ones run on the GPU
               while missing ones are read from the SSD ("waves")
keep the guesses the model agrees with, plus one token of its own
```

Tokens per step ≈ 2.8–3.1, so ~38–39 tok/s. See `docs/profiling.md` for how each
number is measured.

---

## Cleanup still to do

| Step | What | Why |
|---|---|---|
| Phase 3 | Remove the placeholder DFlash draft model (weights, draft context cache, hidden-state capture, draft kernels) | ~0.5–1 GB of memory held for nothing; a large part of `Runtime.mm` |
| Phase 4 | Move model code to `models/qwen4exp/` and shared code to `core/`; turn `QwenTarget` into an explicit model interface | New models then touch only their own folder |
| Phase 4 | Drop Homebrew/release packaging and the model installer | Not used here |
| Rule | Let a model folder launch its own kernels | `Qwen4ExpTarget.cpp` does today; the architecture check flags it |
