# Status — qwen4exp port

**Updated:** 2026-09-20 10:45 PDT by claude-code
**Branch:** `qwen4exp-port` (22 commits, clean, `main` untouched)

## Where we are in one line

Porting **Qwen3.8-Flash-Next** (architecture `qwen4exp`) to the **Splash**
inference engine. All engine work is written and tested. The converter builds
and verifies both layer types against real weights, and a whole composed layer
matches the reference module exactly. **Nothing is blocked.**

## The next three steps, in order

1. **Write `write_head` and `write_embedding`** in
   `dev/tools/convert_qwen4exp.py`. `head_sections()` and
   `embedding_sections()` already exist and define the layout; only the writers
   are missing. `main()` prints "head and embedding are not wired up yet" at the
   end — that line goes away when they are done. Model them on
   `write_head`/`write_embedding` in `dev/tools/convert_qwen38.py`: same shape,
   different layout constants.

2. **Convert all 48 layers.**
   ```
   python3 dev/tools/convert_qwen4exp.py \
     --source ~/models/qwen38-flash-next-bf16 \
     --destination <out>
   ```
   Do **not** pass `--release-source` — there is room now, and the source is
   worth seven hours. Sanity-check first with `--dry-run --destination /tmp/x`,
   which costs 920 KiB and prints `96.61 GiB apparent`.

3. **Write the forward path** (`Qwen4ExpTarget`). `Runtime.mm` currently throws
   for `Qwen4ExpWeights` at target construction — that is the one place to
   start. The layer order is verified exactly; see `decisions.md`.

## Facts you can rely on

| | |
|---|---|
| Source checkpoint | complete — 48/48 layers, `lm_head`, `embed_tokens`, 128/128 n-gram shards |
| Disk free | 199 GiB, against 96.61 GiB needed |
| Composed layer vs. reference | max abs diff **0.000e+00** |
| Test suites | CPU and Metal both green at HEAD |

## Do not touch

- `~/models/qwen38-flash-next-bf16` — 338 GB source, seven hours to re-download.
- `~/models/qwen38-flash-next-v3` — the V3 GGUF Nitin actually runs.
- `splash2/install/models/incoai/Qwen3.8-27B-Splash-HQ` — a running server
  (PID was 7358, cwd `splash2`) has it open.
