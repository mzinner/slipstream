#!/usr/bin/env python3
"""CLI to run prompt generation through Splash generate-sample and decode tokens."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate tokens using Splash engine")
    parser.add_argument(
        "--model-root", default="/Users/nitin/models/qwen38-flash-next-splash"
    )
    parser.add_argument("--metallib", default="build/splash.metallib")
    parser.add_argument("--binary", default="build/engine-tests/generate-sample")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--raw", action="store_true", help="Do not format with chat template"
    )
    args = parser.parse_args()

    model_root = Path(args.model_root)
    tokenizer_dir = model_root / "tokenizer"
    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    if args.raw:
        prompt_text = args.prompt
    else:
        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    print("\n--- PROMPT ---")
    print(prompt_text)
    print("----------------\n")

    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Encoded into {len(input_ids)} tokens.")
    tokens_str = ",".join(str(tid) for tid in input_ids)

    cmd = [
        args.binary,
        args.metallib,
        str(model_root),
        str(args.max_tokens),
        tokens_str,
    ]

    print(f"Running generation via {args.binary}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()

    if result.returncode != 0:
        print(f"Generation failed with return code {result.returncode}!")
        print("STDERR:")
        print(result.stderr)
        print("STDOUT:")
        print(result.stdout)
        sys.exit(result.returncode)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    try:
        output_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Failed to parse JSON output:")
        print(result.stdout)
        sys.exit(1)

    generated_ids = output_data.get("generated_tokens", [])
    prefill_ms = output_data.get("prefill_ms", 0.0)
    step_ms = output_data.get("step_ms", [])
    num_generated = len(generated_ids)

    print(f"Generated token IDs: {generated_ids}")
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print(f"\n--- GENERATED OUTPUT ({num_generated} tokens) ---")
    print(decoded_text)
    print("------------------------------------------------\n")

    total_decode_ms = sum(step_ms) if step_ms else 0.0
    avg_step_ms = total_decode_ms / num_generated if num_generated else 0.0
    decode_tok_per_sec = (
        (num_generated / (total_decode_ms / 1000.0)) if total_decode_ms > 0 else 0.0
    )

    print("--- TIMING & PERFORMANCE ---")
    print(f"Prompt length:      {len(input_ids)} tokens")
    print(
        f"Prefill time:       {prefill_ms:.2f} ms ({len(input_ids) / (prefill_ms / 1000.0) if prefill_ms > 0 else 0:.1f} tok/s)"
    )
    print(f"Generated length:   {num_generated} tokens")
    print(f"Total decode time:  {total_decode_ms:.2f} ms")
    print(f"Average step time:  {avg_step_ms:.2f} ms/tok")
    print(f"Decode speed:       {decode_tok_per_sec:.2f} tok/s")
    print(f"Wall total elapsed: {(t1 - t0):.2f} s")


if __name__ == "__main__":
    main()
