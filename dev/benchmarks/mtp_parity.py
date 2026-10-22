"""Compare generated text with batched MTP drafting off and on.

Every prompt is sent ``--width`` times into one decode batch (twice, run
sequentially, at width 1), once per mode. Greedy runs must agree between the two
modes on every byte; sampled runs only have to reproduce themselves, since
speculative acceptance consumes randomness differently. A wave of identical
prompts keeps the batch composition fixed, which is the only condition under
which greedy text is comparable: the batched kernels already depend on batch
shape, so a mixed wave would report composition drift as if it were an MTP
difference.

Run it under the memory guard, one server at a time:

    PYTHONPATH=. .venv/bin/python dev/benchmarks/guarded.py --floor-gib 3.5 \\
        -- .venv/bin/python dev/benchmarks/mtp_parity.py \\
        --model local/prepared --binary build/slipstream --width 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from dev.tests import smoke_real as smoke

PROMPTS = [
    ("arith", "A notebook costs $7 and a pen costs $2. Maya buys 5 items and "
              "spends $49. How many pens did she buy? Show the equations."),
    ("series", "Derive a closed form for 1 + 4 + 9 + ... + n^2, then use it to "
               "compute the sum for n = 40."),
    ("python", "Write a Python function that merges overlapping integer "
               "intervals, with a docstring and one example."),
    ("rust", "Write a Rust single-producer single-consumer ring buffer with "
             "atomic head and tail indices. Explain the memory ordering."),
    ("sql", "Write a SQL query returning monthly retention cohorts from a table "
            "events(user_id, ts). Explain the CTE steps."),
    ("systems", "Compare Raft and Paxos for replicated logs. State one advantage "
                "of each and one operational cost."),
    ("prose", "Explain in two short paragraphs why cache invalidation is "
              "difficult in distributed systems."),
    ("json", "Return a JSON object describing three hiking trails with name, "
             "distance_km and difficulty fields. No commentary."),
]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ask(port: int, model: str, prompt: str, settings: dict) -> dict:
    body = smoke.chat_body(model, prompt, **settings)
    code, response = smoke.request(port, "POST", "/v1/chat/completions", body)
    smoke.require(code == 200, f"chat completion failed: HTTP {code} {response!r}")
    choice = response["choices"][0]
    text = choice["message"].get("content") or ""
    return {
        "sha256": digest(text),
        "completion_tokens": response["usage"]["completion_tokens"],
        "finish_reason": choice.get("finish_reason"),
    }


def wave(port: int, model: str, prompt: str, settings: dict, width: int) -> list:
    if width == 1:
        return [ask(port, model, prompt, settings) for _ in range(2)]
    gate = threading.Barrier(width + 1)

    def one():
        gate.wait()
        return ask(port, model, prompt, settings)

    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(one) for _ in range(width)]
        gate.wait()
        return [future.result() for future in futures]


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
        cases = []
        for index, (label, body) in enumerate(PROMPTS[: arguments.prompts]):
            prompt = f"{body}\nReference tag: parity-{index}. Do not include the tag."
            started = time.monotonic()
            results = wave(server.port, arguments.model, prompt, settings, arguments.width)
            cases.append(
                {
                    "case": label,
                    "prompt_sha256": digest(prompt),
                    "wall_seconds": time.monotonic() - started,
                    "results": results,
                    "repeatable": len({item["sha256"] for item in results}) == 1,
                }
            )
            print(
                f"{mode} {label}: repeatable={cases[-1]['repeatable']} "
                f"sha={results[0]['sha256'][:12]} "
                f"tokens={[item['completion_tokens'] for item in results]}",
                file=sys.stderr,
                flush=True,
            )
        after = smoke.request(server.port, "GET", "/status")[1]
        smoke.validate_status(after)
        metrics = after["metrics"]
        return {
            "mode": mode,
            "mtp_batch": native_env["SPLASH_MTP_BATCH"],
            "build_id": after["identity"]["cache"]["build_id"],
            "cases": cases,
            "decode_batches_by_width": after["scheduler"]["decode_batches_by_width"],
            "decode_output_tokens": metrics["decode_output_tokens"],
            "decode_wall_ms": metrics["decode_wall_ms"],
            "decode_tokens_per_second": metrics["decode_tokens_per_second"],
            "drafted_tokens": metrics["drafted_tokens"],
            "accepted_draft_tokens": metrics["accepted_draft_tokens"],
            "failed_requests": after["requests"]["failed"],
            "memory_pressure": after["memory_pressure"],
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


def compare(off: dict, on: dict) -> dict:
    off_by_case = {case["case"]: case for case in off["cases"]}
    rows = {}
    for case in on["cases"]:
        left = off_by_case[case["case"]]
        rows[case["case"]] = {
            "off_repeatable": left["repeatable"],
            "on_repeatable": case["repeatable"],
            "on_matches_off": [item["sha256"] == left["results"][0]["sha256"]
                               for item in case["results"]],
        }
    return {
        "cases": rows,
        "all_off_repeatable": all(row["off_repeatable"] for row in rows.values()),
        "all_on_repeatable": all(row["on_repeatable"] for row in rows.values()),
        "all_on_match_off": all(all(row["on_matches_off"]) for row in rows.values()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    smoke.add_server_arguments(parser)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cache-gib", type=float, default=18.0)
    parser.add_argument(
        "--output", type=Path, default=Path("build/verification/mtp-parity.json")
    )
    arguments = smoke.resolve_server_arguments(parser.parse_args(argv))
    if not 1 <= arguments.width <= 4:
        parser.error("--width must be in [1, 4]")
    if not 1 <= arguments.prompts <= len(PROMPTS):
        parser.error(f"--prompts must be in [1, {len(PROMPTS)}]")
    if arguments.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return arguments


def main(argv=None) -> int:
    arguments = parse_args(argv)
    settings = {"max_completion_tokens": arguments.max_tokens}
    if arguments.temperature > 0.0:
        settings["temperature"] = arguments.temperature
        settings["top_p"] = arguments.top_p
        settings["seed"] = arguments.seed if arguments.seed is not None else 20261022
    off = run_mode(arguments, "off", settings)
    on = run_mode(arguments, "on", settings)
    document = {
        "schema_version": 1,
        "timing": "HTTP/native wall; not GPU time",
        "width": arguments.width,
        "prompts": arguments.prompts,
        "settings": settings,
        "off": off,
        "on": on,
        "comparison": compare(off, on),
    }
    comparison = document["comparison"]
    # Greedy decoding is a function of the batch, so the two modes must agree
    # byte for byte. Sampled decoding is not: speculative acceptance consumes
    # randomness differently, so only each mode's own seeded repeatability is
    # required, and the cross-mode column is informational.
    greedy = arguments.temperature == 0.0
    document["criterion"] = (
        "greedy byte parity" if greedy else "seeded repeatability per mode"
    )
    document["correctness_pass"] = bool(
        comparison["all_off_repeatable"]
        and comparison["all_on_repeatable"]
        and (comparison["all_on_match_off"] or not greedy)
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
        f"mtp parity: {'PASS' if document['correctness_pass'] else 'FAIL'} "
        f"({document['criterion']}) width={arguments.width} "
        f"steps off={sum(off['decode_batches_by_width'].values())} "
        f"on={sum(on['decode_batches_by_width'].values())} "
        f"accepted={on['accepted_draft_tokens']}/{on['drafted_tokens']}",
        file=sys.stderr,
    )
    return 0 if document["correctness_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
