#!/usr/bin/env python3
"""Publish the staged Qwen3.8-Flash-Next-Splash package to the Hub.

Run dev/tools/make_flash_next_splash_upload.py first, then:
    python3 dev/tools/upload_flash_next_splash.py [--repo REPO_ID] [--dry-run]

Uploads every staged file with full resume (Xet). README.md is written here
so the tracked tracking-tag front matter stays versioned with the repo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

STAGE = Path.home() / "models/upload-flash-next-splash/Qwen3.8-Flash-Next-Splash"

README = """---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- apple-silicon
- metal
- splash
- splash-packed-q4
- qwen4exp
- flash-next
- moe
- mtp
- speculative-decoding
- ngram-embedding
- 4bit
---

# Qwen3.8-Flash-Next-Splash (Swift V3, Splash Q4 package)

> [!IMPORTANT]
> ### ⚠️ Runtime Package Compatibility Notice
> This model is packed in the **`splash-packed-q4-qwen4exp` (schema_version 5)** format:
> tiled 4-bit experts + 8-bit resident tensors for the Slipstream/Splash Metal runtime
> on Apple Silicon. It is **not** a GGUF. llama.cpp users want the sibling GGUF repo:
> 👉 **[nitinpanj/qwen38-flash-next-v3](https://huggingface.co/nitinpanj/qwen38-flash-next-v3)**
>
> To run this package, use the Slipstream runtime (Qwen4Exp Metal kernels):
> 👉 **[https://github.com/npanj/slipstream](https://github.com/npanj/slipstream)**

## Overview

Swift V3 splice of **Qwen3.8-Flash-Next** (48-layer hybrid GDN + full-attention MoE,
512 routed experts / 10 active, hyper-connections, n-gram per-layer embedding) packed
for native Metal execution:

- Routed MoE experts and the n-gram per-layer embedding table: tiled **Q4_0-style Q4** (streamed from SSD)
- Attention projections, hyper-connection mixers, linear-attention out, shared experts,
  token embedding and output head: **Q8_0** (resident)
- Built-in **MTP** heads + **n-gram Prompt-Lookup drafting** for speculative decoding
  (7 proposal tokens, adaptive depth controller)
- 100.3 GiB total, ~44 tok/s sustained decode on a 64 GiB M-series Mac

| Component | Files | Size |
| :--- | :--- | ---: |
| Target (48 layers + embedding/head/ngram/MTP) | `target/*.bin` | ~99.6 GiB |
| Draft head + 5 draft layers | `draft/*.bin` | ~0.5 GiB |
| Tokenizer (chat template inline) | `tokenizer/` | ~20 MiB |
| Verified manifest (sizes + sha256 per file) | `manifest.json` | — |

## Step-by-Step Setup

### 1. Clone and build the runtime
```bash
git clone https://github.com/npanj/slipstream.git
cd slipstream
make -j4
```

### 2. Serve the model
```bash
./splash serve --model nitinpanj/Qwen3.8-Flash-Next-Splash --port 8090
```
*(First launch verifies `manifest.json` against the Hub, downloads only the listed
artifacts with per-file sha256 verification into `install/models/`, then starts the
OpenAI-compatible API on `http://127.0.0.1:8090`.)*

### 3. Connect coding agents
```bash
omp --model splash/nitinpanj/Qwen3.8-Flash-Next-Splash
# or any OpenAI-compatible client at http://127.0.0.1:8090/v1
```

Requirements: Apple Silicon Mac, ≥64 GiB unified memory recommended, ~110 GiB free
disk for the download. One model server at a time.

## Benchmarks (M-series, 64 GiB, temp 0.0, 145-item suite)

| Benchmark | Accuracy | Decode |
| :--- | ---: | ---: |
| AIME 2025 | 45.0% | 44.3 tok/s |
| MATH-500 (L4-5) | 62.9% | 44.8 tok/s |
| GPQA Diamond | 54.3% | 44.8 tok/s |
| GSM8K | 96.0% | 45.6 tok/s |
| HumanEval | 92.0% | 40.6 tok/s |
| **Overall** | **70.3%** | **43.9 tok/s** |

Against a dense 27B Q8 Splash model: +2.7% overall accuracy, parity speed,
+8.6% on GPQA Diamond.

## Integrity

`manifest.json` lists every artifact with byte size and sha256; the runtime
installer refuses to start on any mismatch. Package built by
`dev/tools/convert_qwen4exp_gguf.py` from the Swift V3 GGUF shards
(`nitinpanj/qwen38-flash-next-v3`).
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="nitinpanj/Qwen3.8-Flash-Next-Splash")
    parser.add_argument(
        "--dry-run", action="store_true", help="list staged files, upload nothing"
    )
    args = parser.parse_args()

    manifest = STAGE / "manifest.json"
    if not manifest.exists():
        print("run make_flash_next_splash_upload.py first", file=sys.stderr)
        return 1
    (STAGE / "README.md").write_text(README)

    files = sorted(
        path.relative_to(STAGE).as_posix()
        for path in STAGE.rglob("*")
        if path.is_file()
    )
    total = sum((STAGE / name).stat().st_size for name in files)
    print(f"{args.repo}: {len(files)} files, {total / 2**30:.2f} GiB")
    for name in files:
        print(f"  {name}")
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(STAGE),
        repo_id=args.repo,
        repo_type="model",
        token=token,
        commit_message="Qwen3.8-Flash-Next-Splash Swift V3 package",
        ignore_patterns=["*.lock", ".DS_Store"],
    )
    print(f"uploaded to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
