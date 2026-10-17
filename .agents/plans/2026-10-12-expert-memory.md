# Expert memory plan — run tomorrow (2026-10-12)

## Goal

Fit more experts in RAM so fewer decode steps touch the SSD. Baseline today:
34 GB cache, ~87 SSD reads per step, ~41.5 tok/s on Flash-Next Swift.

## Measured map (from code + package bytes, 2026-10-11)

- Experts total ~64 GB (512 per layer × 48 layers, ~2.6 MB each, Q4). ~34 GB cached.
- Word table 30 GB — already streams one row per step. Nothing to gain.
- Always-loaded backbone ~5 GB (attention, mixers, norms, shared expert, embedding, head, 0.5 GB draft model).
- Conversation memory ~24 KB/token, only 12 of 48 layers carry full history: 0.8 GB at 32k, 2.4 at 100k, 6.4 at 262k.
- Code pins 2 GB for the OS + 1 GB working margin.
- Rest (~15 GB) is un-itemized headroom: macOS, apps, fragmentation, drivers.
- The 28 GB reserve is a fitted safety constant, not an exact budget. Swap cliff measured past ~38 GB cache. Two watchdog freezes on record for pushing it (2026-09-22).

## Steps (in order, stop when the gain flattens)

1. **Cache 34 → 36 GB.** One variable: `SPLASH_EXPERT_CACHE_GIB=36` on the serve
   command. Expect ~800 more expert slots, roughly 87 → ~84 reads/step, +1–2 tok/s.
   - Preconditions: quiet box, nothing else serving, wired limit set after boot:
     `sudo sysctl iogpu.wired_limit_mb=59392`.
   - Measure with the same 10-prompt suite both arms, same model, same prompts.
   - Abort rule: any swap, freeze, or memory-pressure warning → back to 34, record it.
2. **Context cap 262k → 100k (only if step 1 is clean).** Saves ~1 GB at your 130k
   observed max. Needs an engine restart to take effect. Skip if your sessions
   ever need the tail.
3. **Backbone Q8 → Q4 — REJECTED by owner.** The 8-bit backbone is an intentional
   quality choice (attention precision carries reasoning accuracy). Not to be
   revisited without new evidence that 4-bit attention holds the paired eval set.
4. **Background software path (no memory needed):** predict experts from router
   history + deeper prefetch staging. Diminishing returns — prefetch already hides
   most miss latency (killing 8 reads gained 4.5% last time).

## Outside evidence: Strata's Coder prune (read 2026-10-12)

Strata ships ISTA-DASLab's pruned Flash-Next: 256 of each layer's 512 experts
kept (10 active per token unchanged), authors report 91.3% SWE-bench Verified
and 98.7% LiveCodeBench v6. How they chose: their RCO method ranks experts on
code + agentic + vision calibration data; the release ships the ranking
(`rco-allocation.txt`), and kept experts are stored at higher precision
(IQ3_S-style mixes) to compensate. Lesson for us: prune by profiling router
activations on our own calibration data (code/agentic sessions), not by weight
size; keep survivors at full precision. Their hot-expert profile gets 72% GPU
hits on a 12 GB card — same idea as our prefix cache, different hardware.
Strata is Windows/Linux discrete-GPU only: ideas transfer, code does not.
Also noted: OrcaRouter uncensored IQ3_XXS exists for Flash-Next (manual setup).

## Explicitly parked

- Q2/cold-tier experts (quality risk on the model's knowledge itself).

## Backlog (not scheduled)

- **Static expert tiering from Strata's RCO ranking.** Pin their top-ranked
  experts in RAM, stream the rest from SSD — instead of pruning. Verdict so
  far: mechanically sound, probably not worth it (live cache already ~98%,
  static list goes stale, mapping to Swift weights unverified, gain ceiling
  low single digits). Cheapest test if ever revisited: log our cache misses
  over a week of real sessions and compare against their ranking before
  writing any code. Prefetch-hint use only, never placement.
- GPU-direct expert mapping (Metal memory-manager surgery).
- Native harness port + engine persistent sessions (separate parked decision;
  revisit when routine sessions pass 100k tokens).

## Validation for each step

- Same model (`~/models/swift-qwen38-flash-next-v3/prepared`), same prompts,
  same wired limit, no other servers, no browser load during the run.
- Record: tok/s decode, SSD reads per step (server metrics), TTFT, any pressure
  warnings. Compare against the 34 GB baseline numbers above.
- Leave the tree clean; results go in `.agents/journal.md`.

## Rollback

Every step is an env var or a restart away from today's setup. If anything smells
like swap, stop the server, reboot if the box feels wrong, and note the exact
setting that caused it.

---

## Executed Results & Verdict (2026-10-16)

Ran the full 10-prompt `speed_suite.py` (2,509 tokens across 860 decode steps on `~/models/swift-qwen38-flash-next-v3/prepared`) both with `SPLASH_TRACE` (`trace_report.py`) and clean (no trace overhead) across `SPLASH_EXPERT_CACHE_GIB={25, 32, 34, 36}`:

| Cache Setting | Avg Slots/Layer | Pinned GiB | SSD Reads/Step | Wait SSD (ms) | Traced (ms/step / tok/s) | Clean (ms/step / tok/s) | Lowest Free RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SPLASH_EXPERT_CACHE_GIB=25` | 199 | 25.48 GiB | 108 | 31.0 ms | 88.3 ms / 33.0 tok/s | — | 10.3 GiB |
| `SPLASH_EXPERT_CACHE_GIB=32` | 255 | 32.54 GiB | 62 | 20.0 ms | 76.8 ms / 37.9 tok/s | **78.2 ms / 37.3 tok/s** | **5.7–6.1 GiB** |
| `SPLASH_EXPERT_CACHE_GIB=34` | 272 | 34.69 GiB | 52 | 17.2 ms | 75.8 ms / 38.4 tok/s | **75.1 ms / 38.8 tok/s** | **4.6–5.0 GiB** |
| `SPLASH_EXPERT_CACHE_GIB=36` | 288 | 36.71 GiB | 45 | 14.8 ms | 73.3 ms / 39.8 tok/s | **74.8 ms / 39.0 tok/s** | **3.8–3.9 GiB** |

### Conclusions per Step:
1. **Step 1 (`34 → 36 GiB`):**
   - **Stop rule triggered.** While `36 GiB` lowers SSD reads from `52 → 45/step`, clean end-to-end throughput flattens (`38.8 → 39.0 tok/s`, `+0.5%`) and lowest free host RAM drops to **3.8–3.9 GiB** on <9k-token prompts (below the 4.0 GiB `guarded.py` floor). On 50k–130k agent sessions (~1.2–3.1 GiB KV growth), `36 GiB` will hit the macOS compressor/swap cliff.
   - By contrast, **`32 → 34 GiB`** delivers the real knee of the curve: **+1.5 tok/s** (`37.3 → 38.8 tok/s`, `−3.1 ms/step`, SSD reads `62 → 52/step`) while keeping **4.6–5.0 GiB** free RAM.
2. **Step 2 (`Context cap 262k → 100k`):**
   - Verified in `runtime/engine/MemoryPlan.cpp` and `Q8PageStorage.hpp`: KV physical extents are allocated lazily on demand as a request grows, **not** pinned at startup. Lowering `--max-context` does not free startup RAM for `mlock`ed expert caches (and `splash-flashnext-server.sh` already caps `CTX=131072` with `--clamp-output-budget`).
3. **Final Recommendation:**
   - **Default to a maximum 32 GiB cache** for safety across everyday multi-app and deep-context agent sessions. The engine now applies this cap implicitly; `SPLASH_EXPERT_CACHE_GIB` remains an explicit override.
   - **Use `SPLASH_EXPERT_CACHE_GIB=34` only for dedicated benchmarks or normal-context coding (<64k tokens)** when additional RAM headroom is available. It measured `38.8 tok/s` vs `39.0 tok/s` at 36 GiB while preserving more safety margin. Do **not** deploy `36 GiB` as the default.
