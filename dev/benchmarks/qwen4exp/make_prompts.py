#!/usr/bin/env python3
"""Build the fixed benchmark prompts, as token ids, shared by every engine.

Three sizes, because speed and quality both change with context length:

  short   a one-line question                        ~20 tokens
  code    a real source file plus a request          ~1,500 tokens
  long    several source files plus a request        ~6,000 tokens

The long prompt is past 2,048 tokens on purpose: that is where qwen4exp's
sparse-attention indexer starts to drop tokens, so dense attention and the
reference stop agreeing there.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
TOKENIZER = Path.home() / "models/qwen38-flash-next-splash/tokenizer"


def source(*names):
    return "\n\n".join(
        f"// ===== {name} =====\n" + (ROOT / name).read_text() for name in names
    )


def main(out: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    ask = "\n\nExplain what this code does, then list any bugs you see.\n"
    texts = {
        "short": "Write a Python function that checks whether a string is a palindrome.",
        "code": source("runtime/ops/GDN.cpp") + ask,
        "long": source(
            "runtime/ops/GDN.cpp",
            "runtime/ops/MoE.cpp",
            "runtime/model/StateLayout.hpp",
            "runtime/engine/MemoryGovernor.cpp",
        )
        + ask,
    }
    prompts = {}
    for name, text in texts.items():
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompts[name] = {"text": rendered, "ids": ids}
        print(f"{name:6s} {len(ids):6d} tokens")
    out.write_text(json.dumps(prompts))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build/qwen4exp-prompts.json")
