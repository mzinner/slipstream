#!/usr/bin/env python3
"""Train a block guesser (DFlash-style) on recorded inner state, with MLX.

The guesser sees the model's inner state at every earlier position (the head
input plus three compressed middle layers, recorded by record_features.py) and
the anchor token the model just picked. In one pass it guesses the next
BLOCK - 1 tokens: the block is the anchor followed by blank slots, which attend
to each other and to the recorded state. Its last layer maps into the model's
own head-input space and reuses the model's output layer (frozen, limited to
the draft vocabulary), as the draft head does.

Labels are the model's own top picks, not the session text: a guess is kept
only if it equals the model's pick. A label counts only while the session text
so far agrees with the model's picks (after the first disagreement, the text
no longer shows what the model would have said).

Runs in its own environment: ~/venvs/drafter (mlx, numpy).

  ~/venvs/drafter/bin/python models/qwen4exp/tools/train_drafter.py \\
      ~/models/qwen38-flash-next-drafter-data --out ~/models/qwen38-flash-next-drafter-data/runs/v0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

BLOCK = 8
TAPS = ("layer11", "layer23", "layer35")


# ----------------------------------------------------------------------------
# Data


class Split:
    """One recorded split: features (memory-mapped), tokens, and anchors."""

    def __init__(self, data: Path, name: str, taps=TAPS):
        folder = data / "features" / name
        meta = json.loads((folder / "meta.json").read_text())
        self.rows = meta["rows"]
        dims = meta["dims"]
        self.head = np.memmap(
            folder / "head.f16", np.float16, "r", shape=(self.rows, dims["head"])
        )
        self.taps = [
            np.memmap(folder / f"{t}.f16", np.float16, "r", shape=(self.rows, dims[t]))
            for t in taps
        ]
        self.top = np.fromfile(folder / "top_ids.u32", "<u4").reshape(self.rows, -1)[
            :, 0
        ]
        samples = json.loads((folder / "samples.json").read_text())
        if (folder / "ids.npy").exists():
            corpus = np.load(folder / "ids.npy")
            written = np.ones(len(corpus), np.uint8)
        else:
            corpus = np.fromfile(data / "ids.u32", "<u4")
            written = np.fromfile(data / "written.u8", np.uint8)
        self.tokens = np.zeros(self.rows, np.int64)
        self.written = np.zeros(self.rows, bool)
        self.sample_start = np.zeros(self.rows, np.int64)
        for s in samples:
            r, o, n = s["row"], s["offset"], s["length"]
            self.tokens[r : r + n] = corpus[o : o + n]
            self.written[r : r + n] = written[o : o + n].astype(bool)
            self.sample_start[r : r + n] = r
        self.samples = samples
        # The model's pick at row r is the token it would put at r + 1.
        follows = np.zeros(self.rows, bool)
        follows[1:] = self.top[:-1] == self.tokens[1:]
        follows[self.sample_start == np.arange(self.rows)] = False
        self.follows = follows
        # Anchors: written by the model, and the model itself would have picked
        # the anchor (at inference the anchor is always the model's pick).
        end = np.zeros(self.rows, np.int64)
        for s in samples:
            end[s["row"] : s["row"] + s["length"]] = s["row"] + s["length"]
        position = np.arange(self.rows)
        self.anchors = np.flatnonzero(
            self.written
            & follows
            & (position + BLOCK <= end)
            & (position - self.sample_start >= 16)
        )

    def labels(self, anchors):
        """[n, BLOCK-1] model picks for the slots, and a mask of valid ones."""
        steps = np.arange(BLOCK - 1)
        rows = anchors[:, None] + steps[None, :]  # the pick at row a+j is token a+j+1
        labels = self.top[rows]
        agree = self.follows[anchors[:, None] + 1 + steps[None, :]]
        # Slot j is valid if the text agreed with the model at slots before it.
        valid = np.concatenate(
            [
                np.ones((len(anchors), 1), bool),
                np.cumprod(agree[:, :-1], 1).astype(bool),
            ],
            1,
        )
        return labels, valid

    def features(self, begin, end):
        parts = [self.head[begin:end]] + [t[begin:end] for t in self.taps]
        return np.concatenate(parts, 1)


def window_batches(split: Split, window: int, anchors_per_window: int, rng):
    """Endless (window features, anchors relative to the window) pairs."""
    while True:
        a = int(split.anchors[rng.integers(len(split.anchors))])
        start = max(int(split.sample_start[a]), a - window // 2)
        end = min(start + window, split.rows)
        inside = split.anchors[(split.anchors >= start + 1) & (split.anchors < end)]
        inside = inside[split.sample_start[inside] == split.sample_start[a]]
        if len(inside) > anchors_per_window:
            inside = np.sort(rng.choice(inside, anchors_per_window, replace=False))
        yield start, end, inside


# ----------------------------------------------------------------------------
# Model


class Layer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.norm_q = nn.RMSNorm(d)
        self.norm_ctx = nn.RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm_ffn = nn.RMSNorm(d)
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)
        self.rope = nn.RoPE(self.head_dim, base=10000)

    def __call__(self, x, context, x_pos, c_pos, mask):
        """x: [Q, d] slot states, context: [C, d] recorded state; mask [Q, C+Q]."""
        h = self.norm_q(x)
        c = self.norm_ctx(context)
        keys_in = mx.concatenate([c, h], 0)
        positions = mx.concatenate([c_pos, x_pos], 0)

        def split(t, pos):
            t = t.reshape(-1, self.heads, self.head_dim).transpose(1, 0, 2)
            return self.rope(t, offset=0) if pos is None else rope_at(self.rope, t, pos)

        q = split(self.q(h), x_pos)
        k = split(self.k(keys_in), positions)
        v = self.v(keys_in).reshape(-1, self.heads, self.head_dim).transpose(1, 0, 2)
        out = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )[0]
        x = x + self.o(out.transpose(1, 0, 2).reshape(x.shape[0], -1))
        h = self.norm_ffn(x)
        return x + self.down(nn.silu(self.gate(h)) * self.up(h))


def rope_at(rope, t, positions):
    """RoPE at explicit positions: t [heads, n, dim], positions [n]."""
    dim = t.shape[-1]
    inv = 1.0 / (rope.base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    angles = positions.astype(mx.float32)[:, None] * inv[None, :]
    cos, sin = mx.cos(angles), mx.sin(angles)
    a, b = t[..., : dim // 2], t[..., dim // 2 :]
    return mx.concatenate([a * cos - b * sin, a * sin + b * cos], -1).astype(t.dtype)


class Drafter(nn.Module):
    def __init__(self, part_dims, hidden, d=1024, layers=3, heads=8, ffn=2816):
        super().__init__()
        # Each recorded part has its own scale; normalize them before mixing.
        self.part_dims = list(part_dims)
        self.part_norms = [nn.RMSNorm(n) for n in self.part_dims]
        self.feature_in = nn.Linear(sum(self.part_dims), d, bias=False)
        self.last_in = nn.Linear(sum(self.part_dims), d, bias=False)
        self.token_in = nn.Linear(hidden, d, bias=False)
        self.blank = mx.zeros((hidden,))
        self.layers = [Layer(d, heads, ffn) for _ in range(layers)]
        self.norm_out = nn.RMSNorm(d)
        self.out = nn.Linear(d, hidden, bias=False)

    def __call__(self, features, anchor_embeddings, anchors, window_start):
        """features [C, F] for window rows; anchors [A] absolute rows.
        Returns head-input guesses [A, BLOCK-1, hidden]."""
        count = anchors.shape[0]
        bounds = np.cumsum([0] + self.part_dims)
        features = mx.concatenate(
            [
                norm(features[:, a:b])
                for norm, a, b in zip(self.part_norms, bounds[:-1], bounds[1:])
            ],
            1,
        )
        context = self.feature_in(features)
        rel = anchors - window_start  # anchor's row within the window
        # Like the draft head, every slot starts from the latest recorded state
        # (the row before the anchor), not only what attention finds.
        last = self.last_in(features[rel - 1])
        slots = mx.concatenate(
            [
                anchor_embeddings[:, None, :],
                mx.broadcast_to(self.blank, (count, BLOCK - 1, self.blank.shape[0])),
            ],
            1,
        )
        x = (self.token_in(slots) + last[:, None, :]).reshape(count * BLOCK, -1)
        x_pos = (rel[:, None] + mx.arange(BLOCK)[None, :]).reshape(-1)
        c_pos = mx.arange(features.shape[0])
        # Slots see recorded state strictly before their anchor, and their own block.
        see_context = c_pos[None, :] < mx.repeat(rel, BLOCK)[:, None]
        block_id = mx.repeat(mx.arange(count), BLOCK)
        see_block = block_id[:, None] == block_id[None, :]
        allowed = mx.concatenate([see_context, see_block], 1)
        mask = mx.where(allowed, 0.0, -1e9).astype(x.dtype)
        for layer in self.layers:
            x = layer(x, context, x_pos, c_pos, mask)
        x = self.out(self.norm_out(x)).reshape(count, BLOCK, -1)
        return x[:, 1:, :]


# ----------------------------------------------------------------------------
# Frozen pieces of the model


def load_frozen(data: Path):
    frozen = np.load(data / "frozen.npz")
    return (
        mx.array(frozen["embedding"]),  # [vocab, hidden] float16
        mx.array(frozen["head"]),  # [draft vocab, hidden] float16
        frozen["draft_ids"].astype(np.int64),  # draft slot -> token id
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--d", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--no-taps", action="store_true", help="head input only")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--train-split", default="train", help="for smoke tests")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    taps = () if args.no_taps else TAPS
    train, test = (
        Split(args.data, args.train_split, taps),
        Split(args.data, "test", taps),
    )
    embedding, head, draft_ids = load_frozen(args.data)
    slot_of = np.full(embedding.shape[0], -1, np.int64)
    slot_of[draft_ids] = np.arange(len(draft_ids))
    part_dims = [train.head.shape[1]] + [t.shape[1] for t in train.taps]
    model = Drafter(part_dims, embedding.shape[1], d=args.d, layers=args.layers)
    model.set_dtype(mx.bfloat16)
    optimizer = optim.AdamW(
        learning_rate=optim.join_schedules(
            [
                optim.linear_schedule(0, args.lr, 100),
                optim.cosine_decay(args.lr, args.steps),
            ],
            [100],
        ),
        weight_decay=0.01,
    )
    weights = mx.array(np.exp(-np.arange(BLOCK - 1) / 5.0).astype(np.float32))
    print(
        f"train anchors {len(train.anchors):,}  test anchors {len(test.anchors):,}  "
        f"parameters {sum(v.size for _, v in nn.utils.tree_flatten(model.parameters())) / 1e6:.1f}M"
    )

    def batch(split, start, end, anchors):
        labels, valid = split.labels(anchors)
        slots = slot_of[labels]
        valid &= slots >= 0
        return (
            mx.array(split.features(start, end)).astype(mx.bfloat16),
            embedding[mx.array(split.tokens[anchors])].astype(mx.bfloat16),
            mx.array(anchors),
            start,
            mx.array(np.maximum(slots, 0)),
            mx.array(valid),
        )

    def loss_fn(model, features, anchor_emb, anchors, start, slots, valid):
        guess = model(features, anchor_emb, anchors, start)
        logits = (guess @ head.T.astype(guess.dtype)).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, slots, reduction="none")
        w = valid.astype(mx.float32) * weights[None, :]
        return (ce * w).sum() / mx.maximum(w.sum(), 1.0)

    (args.out / "config.json").write_text(
        json.dumps(
            {
                "d": args.d,
                "layers": args.layers,
                "taps": list(taps),
                "part_dims": part_dims,
                "hidden": int(embedding.shape[1]),
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=1,
        )
    )
    grad_fn = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(0)
    batches = window_batches(train, args.window, args.anchors, rng)
    log = open(args.out / "log.jsonl", "a")
    clock = time.time()
    for step in range(1, args.steps + 1):
        start, end, anchors = next(batches)
        if not len(anchors):
            continue
        loss, grads = grad_fn(model, *batch(train, start, end, anchors))
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step % 50 == 0:
            print(
                f"step {step}  loss {loss.item():.3f}  {time.time() - clock:.0f}s",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, test, batch, head, args.window)
            result.update(step=step, loss=float(loss.item()))
            print(json.dumps(result), flush=True)
            log.write(json.dumps(result) + "\n")
            log.flush()
            model.save_weights(str(args.out / "drafter.safetensors"))


def evaluate(model, split, batch, head, window, limit=4000):
    """Given the text agreed so far: how often each slot's top guess equals the
    model's pick, and the chain length kept (counted only on valid labels)."""
    rng = np.random.default_rng(1)
    anchors = np.sort(
        rng.choice(split.anchors, min(limit, len(split.anchors)), replace=False)
    )
    hits = np.zeros(BLOCK - 1)
    seen = np.zeros(BLOCK - 1)
    for sample_start in np.unique(split.sample_start[anchors]):
        group = anchors[split.sample_start[anchors] == sample_start]
        for i in range(0, len(group), 64):
            part = group[i : i + 64]
            start = max(int(sample_start), int(part[-1]) - window + 1)
            part = part[part > start]
            if not len(part):
                continue
            features, emb, a, s, slots, valid = batch(
                split, start, int(part[-1]) + 1, part
            )
            guess = model(features, emb, a, s)
            picks = np.array(mx.argmax(guess @ head.T.astype(guess.dtype), -1))
            slots, valid = np.array(slots), np.array(valid)
            right = (picks == slots) & valid
            chain = np.cumprod(right, 1).astype(bool)
            prior = np.concatenate([np.ones((len(part), 1), bool), chain[:, :-1]], 1)
            usable = prior & valid
            hits += (chain & usable).sum(0)
            seen += usable.sum(0)
    rates = hits / np.maximum(seen, 1)
    return {
        "rate": [round(float(r), 3) for r in rates],
        "counted": seen.astype(int).tolist(),
    }


if __name__ == "__main__":
    main()
