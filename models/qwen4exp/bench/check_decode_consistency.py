#!/usr/bin/env python3
"""Does decode agree with prefill?

Greedy generation runs through the decode path, where the per-layer
embedding history is committed only for the rows the verifier keeps. A
prefill over prompt + output recomputes every position from scratch. If the
decode history is right, the prefill's top pick at each generated position is
the token decode produced. Near-ties can flip between the two kernel paths,
so a handful of disagreements is expected; a history bug shows as many.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BIN, LIB = ROOT / "build/engine-tests/generate-sample", ROOT / "build/splash.metallib"
MODEL = Path.home() / "models/qwen38-flash-next-splash"
V = 248320


def run(ids, tokens, dump=None):
    env = dict(os.environ)
    if dump:
        env["SPLASH_DUMP_PREFILL_LOGITS"] = dump
    r = subprocess.run(
        [BIN, LIB, MODEL, str(tokens), ",".join(map(str, ids))],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    return json.loads(r.stdout)["generated_tokens"]


prompt = json.load(open(ROOT / "build/qwen4exp-prompts.json"))[
    sys.argv[1] if len(sys.argv) > 1 else "short"
]["ids"]
generated = run(prompt, 48)
full = prompt + generated
with tempfile.TemporaryDirectory() as tmp:
    dump = f"{tmp}/logits.bin"
    run(full, 1, dump)
    raw = np.fromfile(dump, dtype=np.uint16).reshape(len(full), V)
picks = raw[len(prompt) - 1 : len(full) - 1]
top = ((picks.astype(np.uint32) << 16).view(np.float32)).argmax(1)
agree = top == np.array(generated)
print(
    f"generated {len(generated)}; prefill picks the decoded token at {agree.sum()}/{len(agree)}"
)
if not agree.all():
    print("first disagreements at", np.nonzero(~agree)[0][:8].tolist())
