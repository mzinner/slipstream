#!/bin/zsh
# Identical-output check: 10 prompts, greedy, 256 tokens each, compared token
# for token with the reference (Splash's output on 2026-09-22; Slipstream has
# matched it after every change since). Also prints the speed.
#   models/qwen4exp/bench/identical/check.sh [ENV=VALUE ...]
# Needs: no other model server running; `make build/engine-tests/generate-sample`.
set -e
here=${0:A:h}; repo=${here:h:h:h:h}
cd $repo
out=$(mktemp -t identical).jsonl
env "$@" SPLASH_EXPERT_CACHE_GIB=34 dev/benchmarks/guarded.py --max-seconds 900 -- \
  build/engine-tests/generate-sample build/splash.metallib ~/models/qwen38-flash-next-splash \
  256 "@$here/prompts.txt" > $out
python3 - $here/reference.jsonl $out <<'PY'
import json, sys
ref = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
new = [json.loads(l) for l in open(sys.argv[2]) if l.startswith("{")]
same = sum(a["generated_tokens"] == b["generated_tokens"] for a, b in zip(ref, new))
tokens = sum(len(b["generated_tokens"]) for b in new)
ms = sum(sum(b["step_ms"]) for b in new)
print(f"same output as reference: {same}/{len(ref)}   decode {1000 * tokens / ms:.1f} tok/s")
sys.exit(0 if same == len(ref) == len(new) else 1)
PY
