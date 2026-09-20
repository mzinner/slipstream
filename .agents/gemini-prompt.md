# Prompt to hand Gemini

Copy everything below the line.

---

You are continuing a port that another agent (claude-code) started. The state is
written down — read it before doing anything.

**First, read these four things, in this order:**

1. `/Users/nitin/.agents/cross-agent-convention.md` — how we hand work between
   agents. You are `gemini` in the journal.
2. `/Users/nitin/.agents/writing-style.md` — how to write for me.
3. `.agents/status.md` in the repo — where things stand and the next steps.
4. `.agents/decisions.md` — the choices that are expensive to rediscover.

**The repo:** `/Users/nitin/Documents/shared-with-google-drive/model-serving/splash`

**Work on the branch `qwen4exp-gemini`** (already created, already at the right
commit). Do not commit to `qwen4exp-port` — that is the other agent's branch, and
keeping them separate is what lets us merge the two lines of work later.

**The job, in order:**

1. Write `write_head` and `write_embedding` in `dev/tools/convert_qwen4exp.py`.
   `head_sections()` and `embedding_sections()` already exist and define the
   layout; only the writers are missing. Model them on the functions of the same
   names in `dev/tools/convert_qwen38.py` — same shape, different constants.
   When they work, delete the line in `main()` that prints
   "head and embedding are not wired up yet".

2. Check the layout before converting anything:
   `python3 dev/tools/convert_qwen4exp.py --dry-run --destination /tmp/x`
   This writes headers only (about 920 KiB) and should print `96.61 GiB apparent`.

3. Convert all 48 layers:
   ```
   python3 dev/tools/convert_qwen4exp.py \
     --source ~/models/qwen38-flash-next-bf16 \
     --destination <somewhere with 100+ GiB free>
   ```

4. Load the resulting package in the engine and confirm it validates.

5. Write the forward path (`Qwen4ExpTarget`). `runtime/model/Runtime.mm`
   currently throws for `Qwen4ExpWeights` — that is where to start. The layer
   order is already verified exactly; `.agents/decisions.md` has it, and
   `dev/tools/checks/check_layer_composition.py` will re-check it for you.

**Rules, because some mistakes here are expensive:**

- **Never delete `~/models/qwen38-flash-next-bf16`.** It is 338 GB and takes
  seven hours to download. Do not pass `--release-source` to the converter —
  there is plenty of disk now, and that flag deletes the source as it runs.
- **Do not touch** `~/models/qwen38-flash-next-v3` (the model I actually run) or
  `model-serving/splash2/install/models/incoai/Qwen3.8-27B-Splash-HQ` (a running
  server has it open).
- **Run both test suites before each commit** and keep them green. The other
  agent did this for all 22 commits; don't break the streak.
- **Confirm a build actually succeeded before you interpret a test result.**
  Reading a stale binary's output has already cost time on this project once.
- **A synthetic test that agrees with itself proves nothing.** If you write a
  test and the code that generates its fixture, you have tested neither. Check
  against the real checkpoint or the reference implementation.

**When you finish meaningful work,** update `.agents/` as the convention says:
append to `journal.md`, overwrite `status.md`, add to `decisions.md`, tick
`tasks.md`.

**If something in `.agents/` contradicts the code, the code wins** — fix the note.

Ask me if you get stuck rather than guessing at the architecture.
