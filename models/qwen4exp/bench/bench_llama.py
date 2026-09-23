#!/usr/bin/env python3
"""Time a running llama-server on the same prompts, fed as the same token ids.

Uses the server's own timings, so network and JSON overhead are excluded.
Greedy (temperature 0) to match the Splash runs. The prompt cache is
disabled per request so every prompt is prefilled from scratch.
"""

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def complete(url, ids, tokens):
    body = json.dumps(
        {
            "prompt": ids,
            "n_predict": tokens,
            "temperature": 0,
            "top_k": 1,
            "cache_prompt": False,
            "ignore_eos": False,
            "return_tokens": True,
        }
    ).encode()
    request = urllib.request.Request(
        url + "/completion", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--prompts", default=str(ROOT / "build/qwen4exp-prompts.json"))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--only", default="short,code,long")
    parser.add_argument("--out")
    args = parser.parse_args()
    prompts = json.loads(Path(args.prompts).read_text())
    complete(args.url, prompts["short"]["ids"], 8)  # warm-up
    results = {}
    for name in args.only.split(","):
        out = complete(args.url, prompts[name]["ids"], args.tokens)
        t = out["timings"]
        r = {
            "prompt_tokens": t["prompt_n"],
            "prefill_tok_s": t["prompt_per_second"],
            "generated": t["predicted_n"],
            "decode_mean_tok_s": t["predicted_per_second"],
            "draft_n": t.get("draft_n"),
            "draft_accepted": t.get("draft_n_accepted"),
            "tokens": out.get("tokens", []),
        }
        results[name] = r
        acc = (
            f"  drafts accepted {r['draft_accepted']}/{r['draft_n']}"
            if r["draft_n"]
            else ""
        )
        print(
            f"{name:6s} prompt {r['prompt_tokens']:5d}  prefill {r['prefill_tok_s']:7.1f} tok/s"
            f"  decode {r['decode_mean_tok_s']:5.2f} tok/s  generated {r['generated']}{acc}",
            flush=True,
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
