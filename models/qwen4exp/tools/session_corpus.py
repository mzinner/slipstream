#!/usr/bin/env python3
"""Turn omp sessions written by this model into token ids for guesser training.

Each session becomes one or more samples: the conversation rendered through the
model's own chat template (earlier reasoning kept, as the server does with
preserve_thinking), plus a mask marking the tokens this model wrote. Turns by
other models stay as context but are not marked: they would teach the guesser
the wrong habits. A session longer than --max-tokens continues in a new sample
that restarts from its first user message (omp compacts long sessions too).

Output (in --out):
  ids.u32      every sample's token ids, back to back (little-endian uint32)
  written.u8   1 where this model wrote the token, else 0 (same length)
  samples.json one entry per sample: offset, length, written count, split, source
A fixed 5% of sessions (by a hash of the file name) is kept aside as "test".

  .venv/bin/python models/qwen4exp/tools/session_corpus.py \
      ~/.omp/agent/sessions --out ~/models/qwen38-flash-next-drafter-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

MODEL_MARK = "flash-next"
# The template accepts low, medium and xhigh; the server maps the rest the same way.
EFFORTS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
}


def load_session(path):
    """Messages in file order, the model names seen, and the thinking level."""
    messages, models, level = [], set(), None
    for line in open(path, errors="replace"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "thinking_level_change" and level is None:
            level = record.get("thinkingLevel")
        message = record.get("message")
        if record.get("type") != "message" or not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        parts = (
            content
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
        )
        text = "".join(
            p.get("text") or ""
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        )
        if role == "user":
            messages.append({"role": "user", "content": text})
        elif role == "toolResult":
            messages.append({"role": "tool", "content": text})
        elif role == "assistant":
            model = str(message.get("model") or record.get("model") or "?")
            models.add(model)
            thinking = "".join(
                p.get("thinking") or ""
                for p in parts
                if isinstance(p, dict) and p.get("type") == "thinking"
            )
            calls = [
                {
                    "type": "function",
                    "function": {
                        "name": p.get("name"),
                        "arguments": p.get("arguments") or {},
                    },
                }
                for p in parts
                if isinstance(p, dict) and p.get("type") == "toolCall"
            ]
            entry = {"role": "assistant", "content": text, "mine": MODEL_MARK in model}
            if thinking:
                entry["reasoning_content"] = thinking
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
    return messages, models, level


def render(tokenizer, messages, level):
    options = {"tokenize": False, "preserve_thinking": True}
    if level == "off":
        options["enable_thinking"] = False
    elif level in EFFORTS:
        options["reasoning_effort"] = EFFORTS[level]
    messages = [{k: v for k, v in m.items() if k != "mine"} for m in messages]
    return tokenizer.apply_chat_template(messages, **options)


def encode(tokenizer, messages, level, max_tokens):
    """Samples of (token ids, written-mask) covering the session."""
    head = next((i + 1 for i, m in enumerate(messages) if m["role"] == "user"), 0)
    samples, start = [], None  # start: first message after the head in a later window
    while True:
        window = messages if start is None else messages[:head] + messages[start:]
        ids, written, used = encode_window(tokenizer, window, level, max_tokens)
        consumed = used if start is None else used - head
        if consumed <= 0:
            break
        samples.append((ids, written))
        start = consumed if start is None else start + consumed
        if start >= len(messages):
            break
    return samples


def encode_window(tokenizer, messages, level, max_tokens):
    """Token ids, written-mask and messages used; stops at max_tokens or where
    rendering stops being a clean continuation of the text before it."""
    ids, written = [], []
    done, used = "", 0
    # Cut only around assistant turns: consecutive tool results render as one
    # block, so a cut between them would not be a clean continuation.
    cuts = [
        count
        for count in range(1, len(messages) + 1)
        if count == len(messages)
        or messages[count - 1]["role"] == "assistant"
        or messages[count]["role"] == "assistant"
    ]
    for count in cuts:
        text = render(tokenizer, messages[:count], level)
        if not text.startswith(done):
            break
        piece = tokenizer(text[len(done) :], add_special_tokens=False)["input_ids"]
        if len(ids) + len(piece) > max_tokens:
            break
        ids += piece
        written += [bool(messages[count - 1].get("mine"))] * len(piece)
        done, used = text, count
    return ids, written, used


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sessions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path.home() / "models/qwen38-flash-next-splash/tokenizer",
    )
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--test-percent", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    args.out.mkdir(parents=True, exist_ok=True)
    all_ids, all_written, samples = [], [], []
    skipped = {"other model": 0, "no reply": 0}
    offset = 0
    for path in sorted(args.sessions.rglob("*.jsonl")):
        messages, models, level = load_session(path)
        if not models:
            skipped["no reply"] += 1
            continue
        if not any(MODEL_MARK in m for m in models):
            skipped["other model"] += 1
            continue
        bucket = int(hashlib.sha256(path.name.encode()).hexdigest(), 16) % 100
        for ids, written in encode(tokenizer, messages, level, args.max_tokens):
            if not any(written):
                continue
            samples.append(
                {
                    "offset": offset,
                    "length": len(ids),
                    "written": int(sum(written)),
                    "split": "test" if bucket < args.test_percent else "train",
                    "source": str(path.relative_to(args.sessions)),
                }
            )
            all_ids.append(np.asarray(ids, dtype="<u4"))
            all_written.append(np.asarray(written, dtype=np.uint8))
            offset += len(ids)
    np.concatenate(all_ids).tofile(args.out / "ids.u32")
    np.concatenate(all_written).tofile(args.out / "written.u8")
    (args.out / "samples.json").write_text(json.dumps(samples, indent=1))
    for split in ("train", "test"):
        part = [s for s in samples if s["split"] == split]
        print(
            f"{split}: {len(part)} sessions, {sum(s['length'] for s in part):,} tokens, "
            f"{sum(s['written'] for s in part):,} written by the model"
        )
    print(f"skipped: {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
