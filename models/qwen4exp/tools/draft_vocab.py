#!/usr/bin/env python3
"""Rank the vocabulary by how often tokens appear in real text.

The MTP draft head can score only the most common tokens instead of all
~248K (SPLASH_DRAFT_VOCAB=N). Its guesses are still checked by the full
model, so this changes speed, never answers. This tool writes the ranking
the engine reads: <package>/target/draft-vocab.bin, little-endian uint32
token ids, most frequent first, every id exactly once.

Counts come from any .jsonl / .json / source files under the given paths
(every string value in JSON is counted). Ties go to the lower id, and ids
never seen follow in id order.

  .venv/bin/python models/qwen4exp/tools/draft_vocab.py ~/models/qwen38-flash-next-splash \
      ~/.omp/agent/sessions ../../benchmarking/model-quality-bench/data .
"""

import argparse
import collections
import json
import struct
import sys
from pathlib import Path

from tokenizers import Tokenizer

SOURCE = {
    ".py",
    ".cpp",
    ".hpp",
    ".mm",
    ".h",
    ".metal",
    ".md",
    ".sh",
    ".ts",
    ".js",
    ".rs",
    ".go",
    ".txt",
}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def texts(root: Path, cap_bytes: int):
    files = (
        [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    )
    for path in files:
        if any(
            part in {".venv", "node_modules", ".git", "build"} for part in path.parts
        ):
            continue
        if path.stat().st_size > cap_bytes:
            continue
        if path.suffix in {".jsonl", ".json"}:
            try:
                raw = path.read_text(errors="ignore")
            except OSError:
                continue
            lines = raw.splitlines() if path.suffix == ".jsonl" else [raw]
            for line in lines:
                try:
                    yield from strings(json.loads(line))
                except ValueError:
                    continue
        elif path.suffix in SOURCE:
            try:
                yield path.read_text(errors="ignore")
            except OSError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--max-file-mb", type=float, default=64)
    a = ap.parse_args()
    package = Path(a.package).expanduser()
    tok = Tokenizer.from_file(str(package / "tokenizer" / "tokenizer.json"))
    vocab = json.loads((package / "manifest.json").read_text())["target"][
        "vocabulary_size"
    ]
    counts = collections.Counter()
    batch, total = [], 0
    for root in a.corpus:
        for text in texts(Path(root).expanduser(), int(a.max_file_mb * 2**20)):
            batch.append(text)
            if len(batch) == 512:
                for enc in tok.encode_batch(batch, add_special_tokens=False):
                    counts.update(enc.ids)
                    total += len(enc.ids)
                batch = []
    for enc in tok.encode_batch(batch, add_special_tokens=False):
        counts.update(enc.ids)
        total += len(enc.ids)
    ranked = sorted((i for i in counts if i < vocab), key=lambda i: (-counts[i], i))
    seen = set(ranked)
    ranked += [i for i in range(vocab) if i not in seen]
    out = package / "target" / "draft-vocab.bin"
    out.write_bytes(struct.pack(f"<{len(ranked)}I", *ranked))

    def covered(n):
        return sum(counts[i] for i in ranked[:n]) / total

    print(
        f"{total:,} tokens counted, {len(seen):,} distinct; wrote {out} ({len(ranked):,} ids)"
    )
    for n in (8192, 16384, 32768, 65536):
        print(f"  top {n:>6,}: {100 * covered(n):.2f}% of counted tokens")


if __name__ == "__main__":
    sys.exit(main())
