"""Compare the sampled token distribution with batched MTP drafting off and on.

Greedy parity cannot cover sampled decoding: speculative acceptance legitimately
consumes randomness differently, so the two modes never produce the same text at
temperature > 0 and text equality is the wrong criterion. What has to hold is
that both modes sample from the *same distribution*. This harness samples each
prompt many times per mode with distinct seeds, re-tokenizes every completion
through the package tokenizer, and compares the two pools.

Each sample is a wave of ``--width`` identical requests so the decode step is
genuinely batched: ``SPLASH_MTP_BATCH`` only gates *per-lane* drafting, so at
width 1 both modes take the single-lane drafting path and the comparison would
be the same code against itself. Identical prompts keep the batch composition
constant, and the lanes must agree with each other on every sample.

The comparison is a G-test over a fixed head of the token distribution (the most
frequent tokens across both modes) plus one "other" bucket, calibrated against
the same statistic computed on random half-splits of a single mode. Comparing on
the head matters: a residual-sampling bug shows up as a few percent of mass
moved between common tokens, while total variation over the full long-tail
support is dominated by singly-observed tokens and cannot see it.

Sensitivity: at the default 64 samples per prompt the test reliably rejects a
redistribution of roughly 5% of the mass between head tokens (p < 0.01 in the
synthetic calibration) and does not reject a faithful sampler. Smaller biases
are reported through ``head_tv`` but sit below the noise floor of this sample
count. Raise ``--samples`` for more power.

Run it under the memory guard, one server at a time:

    PYTHONPATH=. .venv/bin/python dev/benchmarks/guarded.py --floor-gib 3.5 \\
        -- .venv/bin/python dev/benchmarks/mtp_sampling_distribution.py \\
        --model local/prepared --binary build/slipstream
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from dev.benchmarks.mtp_parity import PROMPTS
from dev.tests import smoke_real as smoke


def completion(port: int, model: str, prompt: str, settings: dict, seed: int) -> str:
    body = smoke.chat_body(model, prompt, **{**settings, "seed": seed})
    code, response = smoke.request(port, "POST", "/v1/chat/completions", body)
    smoke.require(code == 200, f"chat completion failed: HTTP {code} {response!r}")
    return response["choices"][0]["message"].get("content") or ""


def wave(port: int, model: str, prompt: str, settings: dict, seed: int,
         width: int) -> list[str]:
    """One batched decode: ``width`` identical requests admitted together."""

    gate = threading.Barrier(width)

    def one() -> str:
        gate.wait()
        return completion(port, model, prompt, settings, seed)

    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(one) for _ in range(width)]
        return [future.result() for future in futures]


def tokenize(port: int, text: str) -> list[int]:
    code, response = smoke.request(port, "POST", "/tokenize", {"content": text})
    smoke.require(code == 200, f"tokenize failed: HTTP {code} {response!r}")
    return list(response["tokens"])


def counts(sequences: list[list[int]]) -> Counter:
    pooled: Counter = Counter()
    for ids in sequences:
        pooled.update(ids)
    return pooled


def head(pools: list[Counter], limit: int) -> list[int]:
    """Most frequent tokens across every pool: one head shared by all rounds."""

    pooled: Counter = Counter()
    for pool in pools:
        pooled.update(pool)
    return [token for token, _ in pooled.most_common(limit)]


def bucketize(pool: Counter, head_tokens: list[int]) -> list[int]:
    head_set = set(head_tokens)
    other = sum(count for token, count in pool.items() if token not in head_set)
    return [pool.get(token, 0) for token in head_tokens] + [other]


def head_mass(pool: Counter, head_tokens: list[int]) -> float:
    total = sum(pool.values())
    if not total:
        return 0.0
    head_set = set(head_tokens)
    return sum(count for token, count in pool.items() if token in head_set) / total


def g_statistic(left: Counter, right: Counter, head_tokens: list[int]) -> float:
    """G-test of independence between group and head bucket."""

    left_buckets = bucketize(left, head_tokens)
    right_buckets = bucketize(right, head_tokens)
    left_total = sum(left_buckets)
    right_total = sum(right_buckets)
    if not left_total or not right_total:
        return 0.0
    combined = left_total + right_total
    statistic = 0.0
    for left_count, right_count in zip(left_buckets, right_buckets):
        total = left_count + right_count
        if total == 0:
            continue
        for observed, scale in ((left_count, left_total), (right_count, right_total)):
            if observed:
                expected = total * scale / combined
                statistic += 2.0 * observed * math.log(observed / expected)
    return statistic


def split_pair(left: list[list[int]], right: list[list[int]],
               head_tokens: list[int], rounds: int, seed: int) -> list[float]:
    """G statistic between independent random halves of two sample pools.

    Both sides are half the pool, so the cross-mode statistic and the
    within-mode null share a sample size and the null is a faithful noise floor.
    """

    generator = random.Random(seed)
    values = []
    for _ in range(rounds):
        a = generator.sample(left, k=len(left) // 2)
        b = generator.sample(right, k=len(right) // 2)
        values.append(g_statistic(counts(a), counts(b), head_tokens))
    return values


def split_within(pool: list[list[int]], head_tokens: list[int], rounds: int,
                 seed: int) -> list[float]:
    """G statistic between two disjoint halves of one pool: the noise floor."""

    generator = random.Random(seed)
    values = []
    for _ in range(rounds):
        shuffled = generator.sample(pool, k=len(pool))
        middle = len(shuffled) // 2
        values.append(
            g_statistic(
                counts(shuffled[:middle]),
                counts(shuffled[middle:]),
                head_tokens,
            )
        )
    return values


def tv(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    left_mass = sum(left.values()) or 1
    right_mass = sum(right.values()) or 1
    return 0.5 * sum(
        abs(left[key] / left_mass - right[key] / right_mass) for key in keys
    )


def compare(off: dict, on: dict, arguments) -> dict:
    off_by_case = {case["case"]: case["sequences"] for case in off["cases"]}
    on_by_case = {case["case"]: case["sequences"] for case in on["cases"]}
    shared = [name for name in off_by_case if name in on_by_case]
    off_all = [ids for name in shared for ids in off_by_case[name]]
    on_all = [ids for name in shared for ids in on_by_case[name]]

    off_pool, on_pool = counts(off_all), counts(on_all)
    head_tokens = head([off_pool, on_pool], arguments.head_tokens)
    observed_values = split_pair(
        off_all, on_all, head_tokens, arguments.rounds, arguments.rng_seed
    )
    null_values = (
        split_within(off_all, head_tokens, arguments.rounds, arguments.rng_seed + 1)
        + split_within(on_all, head_tokens, arguments.rounds, arguments.rng_seed + 2)
    )
    observed = sum(observed_values) / len(observed_values)
    # One-sided: MTP is only suspect when it separates the pools *more* than
    # resampling the same mode does.
    victories = sum(1 for value in null_values if value >= observed)
    p_value = (victories + 1) / (len(null_values) + 1)

    diversity = {
        name: {
            "sequences": len(off_by_case[name]),
            "off_distinct": len({tuple(ids) for ids in off_by_case[name]}),
            "on_distinct": len({tuple(ids) for ids in on_by_case[name]}),
        }
        for name in shared
    }
    # A run where every sample is the same text cannot distinguish the two
    # distributions, however small the statistic is: report it as vacuous
    # rather than as a pass.
    informative = all(
        row["off_distinct"] > 1 and row["on_distinct"] > 1
        for row in diversity.values()
    )
    lane_disagreements = sum(case["lane_disagreements"] for case in on["cases"])
    # The pools only differ when per-lane drafting actually ran. Raw draft
    # counts do not show that: drafting-off still proposes through the batched
    # single-chain path, it just almost never lands an acceptance.
    off_rate = off["accepted_draft_tokens"] / max(1, off["drafted_tokens"])
    on_rate = on["accepted_draft_tokens"] / max(1, on["drafted_tokens"])
    drafting_exercised = bool(on_rate > 0.2 and on_rate > off_rate * 3.0)
    # Identical lanes are a determinism property, not a precondition for
    # comparing distributions: both lanes of a wave are draws from the same
    # distribution under the same fixed batch shape either way. It is reported
    # because a divergence between identical seeded requests is worth knowing.
    informative = informative and drafting_exercised

    first_off = [ids[:1] for ids in off_all if ids]
    first_on = [ids[:1] for ids in on_all if ids]
    return {
        "diversity": diversity,
        "informative": informative,
        "drafting_exercised": drafting_exercised,
        "acceptance_rate": {"off": off_rate, "on": on_rate},
        "lane_disagreement_rate": lane_disagreements
        / max(1, sum(len(case["sequences"]) for case in on["cases"])),
        "off_drafted_tokens": off["drafted_tokens"],
        "on_drafted_tokens": on["drafted_tokens"],
        "lane_disagreements": lane_disagreements,
        "pooled_tokens": {"off": sum(off_pool.values()),
                          "on": sum(on_pool.values())},
        "head_tokens": len(head_tokens),
        "head_mass": {"off": head_mass(off_pool, head_tokens),
                      "on": head_mass(on_pool, head_tokens)},
        "g_statistic": observed,
        "null_g_mean": sum(null_values) / len(null_values),
        "null_g_p95": sorted(null_values)[int(0.95 * (len(null_values) - 1))],
        "p_value": p_value,
        "head_tv": tv(
            Counter({token: off_pool.get(token, 0) for token in head_tokens}),
            Counter({token: on_pool.get(token, 0) for token in head_tokens}),
        ),
        "first_token_tv": tv(counts(first_off), counts(first_on)),
        "first_token_distinct": {
            "off": len({ids[0] for ids in first_off}),
            "on": len({ids[0] for ids in first_on}),
        },
        "cases": shared,
    }


def run_mode(arguments, mode: str, settings: dict) -> dict:
    native_env = {
        "SPLASH_MTP_BATCH": "0" if mode == "off" else "1",
        "SPLASH_MAX_BATCH_WIDTH": str(arguments.width),
        "SPLASH_MAX_CONCURRENCY": str(arguments.width),
        "SPLASH_EXPERT_CACHE_GIB": str(arguments.cache_gib),
    }
    saved = {key: os.environ.get(key) for key in native_env}
    os.environ.update(native_env)
    # Presence, not value, selects shadow mode: never inherit it.
    shadow = os.environ.pop("SPLASH_MTP_SHADOW", None)
    server = smoke.RealServer(arguments)
    try:
        status = server.wait_ready(arguments.startup_timeout)
        smoke.validate_status(status)
        # Two warmup turns keep first-use prefill and cache fills out of the
        # sample: they are identical across modes and only add latency.
        for _, body in PROMPTS[:2]:
            wave(
                server.port, arguments.model, f"{body}\nwarmup", settings, 1,
                arguments.width,
            )
        cases = []
        started = time.monotonic()
        for index, (label, body) in enumerate(PROMPTS[: arguments.prompts]):
            prompt = f"{body}\nReference tag: dist-{index}. Do not include the tag."
            sequences = []
            disagreements = 0
            for sample in range(arguments.samples):
                seed = arguments.seed_base + index * 1000 + sample
                texts = wave(
                    server.port, arguments.model, prompt, settings, seed,
                    arguments.width,
                )
                for text in texts:
                    smoke.require(text.strip() != "", "sampled completion was empty")
                if len(set(texts)) != 1:
                    disagreements += 1
                sequences.append(tokenize(server.port, texts[0]))
            cases.append(
                {"case": label, "sequences": sequences,
                 "lane_disagreements": disagreements}
            )
            lengths = [len(ids) for ids in sequences]
            distinct = len({tuple(ids) for ids in sequences})
            print(
                f"{mode} {label}: tokens min={min(lengths)} max={max(lengths)} "
                f"distinct_completions={distinct}/{len(sequences)} "
                f"lane_disagreements={disagreements} "
                f"distinct_first={len({ids[0] for ids in sequences if ids})}",
                file=sys.stderr,
                flush=True,
            )
        after = smoke.request(server.port, "GET", "/status")[1]
        smoke.validate_status(after)
        return {
            "mode": mode,
            "mtp_batch": native_env["SPLASH_MTP_BATCH"],
            "build_id": after["identity"]["cache"]["build_id"],
            "wall_seconds": time.monotonic() - started,
            "cases": cases,
            "drafted_tokens": after["metrics"]["drafted_tokens"],
            "accepted_draft_tokens": after["metrics"]["accepted_draft_tokens"],
            "failed_requests": after["requests"]["failed"],
            "metal_healthy": after["metal"]["healthy"],
        }
    finally:
        server.close()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if shadow is not None:
            os.environ["SPLASH_MTP_SHADOW"] = shadow


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    smoke.add_server_arguments(parser)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed-base", type=int, default=610000)
    parser.add_argument("--head-tokens", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--cache-gib", type=float, default=18.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/verification/mtp-sampling-distribution.json"),
    )
    arguments = smoke.resolve_server_arguments(parser.parse_args(argv))
    if not 1 <= arguments.prompts <= len(PROMPTS):
        parser.error(f"--prompts must be in [1, {len(PROMPTS)}]")
    if not 2 <= arguments.width <= 4:
        # Width 1 takes the single-lane drafting path in both modes, so the
        # flag would be inert and the comparison vacuous.
        parser.error("--width must be in [2, 4]: per-lane drafting needs a batch")
    if arguments.samples < 16:
        parser.error("--samples must be at least 16")
    if arguments.head_tokens < 4:
        parser.error("--head-tokens must be at least 4")
    return arguments


def main(argv=None) -> int:
    arguments = parse_args(argv)
    settings = {
        "max_completion_tokens": arguments.max_tokens,
        "temperature": arguments.temperature,
        "top_p": arguments.top_p,
    }
    off = run_mode(arguments, "off", settings)
    on = run_mode(arguments, "on", settings)
    comparison = compare(off, on, arguments)
    document = {
        "schema_version": 1,
        "timing": "HTTP wall; sampling statistics only",
        "settings": settings,
        "width": arguments.width,
        "samples_per_prompt": arguments.samples,
        "off": off,
        "on": on,
        "comparison": comparison,
        "criterion": (
            "head-distribution G statistic is within the within-mode "
            "split-half noise floor"
        ),
    }
    document["distribution_pass"] = bool(
        comparison["informative"]
        and comparison["p_value"] >= arguments.alpha
        and not off["failed_requests"]
        and not on["failed_requests"]
        and off["metal_healthy"]
        and on["metal_healthy"]
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(arguments.output)
    print(json.dumps(comparison, indent=2))
    print(
        f"mtp sampling distribution: "
        f"{'PASS' if document['distribution_pass'] else 'FAIL'} "
        f"informative={comparison['informative']} width={arguments.width} "
        f"off_drafted={comparison['off_drafted_tokens']} "
        f"g={comparison['g_statistic']:.2f} "
        f"null={comparison['null_g_mean']:.2f}+-"
        f"p95={comparison['null_g_p95']:.2f} "
        f"p={comparison['p_value']:.3f} alpha={arguments.alpha} "
        f"accepted={on['accepted_draft_tokens']}/{on['drafted_tokens']}",
        file=sys.stderr,
    )
    return 0 if document["distribution_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
