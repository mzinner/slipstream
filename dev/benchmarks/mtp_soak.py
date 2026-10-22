"""Sustained-load soak: tail latency, memory and drift with batched MTP.

Aggregate throughput can improve while a lane's tail latency degrades: when a
request waits on a co-tenant's verify step the mean barely moves, but the 99th
percentile inter-token gap does. This harness holds a fixed batch width for a
fixed wall-clock window per mode, streams every completion so token arrival
times are client-observed, and records p50/p95/p99 inter-token latency, time to
first token, throughput, memory growth and draft-acceptance drift between the
first and last windows.

Both modes run the same schedule, so the comparison isolates drafting rather
than load shape.

Run it under the memory guard, one server at a time:

    PYTHONPATH=. .venv/bin/python dev/benchmarks/guarded.py --floor-gib 3.5 \\
        -- .venv/bin/python dev/benchmarks/mtp_soak.py \\
        --model local/prepared --binary build/slipstream --duration 600
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

from dev.benchmarks.mtp_parity import PROMPTS
from dev.tests import smoke_real as smoke


def stream_completion(port: int, body: dict) -> dict:
    """POST a streaming chat completion and time every content delta."""

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    started = time.monotonic()
    arrivals: list[float] = []
    completion_tokens = 0
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        smoke.require(response.status == 200, f"stream failed: HTTP {response.status}")
        while True:
            line = response.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage:
                completion_tokens = int(usage.get("completion_tokens") or 0)
            for choice in chunk.get("choices") or ():
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    arrivals.append(time.monotonic())
    finally:
        connection.close()
    finished = time.monotonic()
    gaps = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
    return {
        "ttft_ms": (arrivals[0] - started) * 1000.0 if arrivals else None,
        "wall_ms": (finished - started) * 1000.0,
        "tokens": completion_tokens or len(arrivals),
        "gaps_ms": [gap * 1000.0 for gap in gaps],
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def summarise(mode: str, requests: list[dict], samples: list[dict]) -> dict:
    # Per-token cost per request, not inter-chunk gaps: an SSE event carries
    # however many tokens the native step produced, and a speculative step
    # produces more of them than a single-lane one, so chunk gaps measure the
    # transport's batching and would compare the two modes on different scales.
    per_token = [
        (item["wall_ms"] - item["ttft_ms"]) / max(1, item["tokens"] - 1)
        for item in requests
        if item["ttft_ms"] is not None and item["tokens"] > 1
    ]
    gaps = [gap for item in requests for gap in item["gaps_ms"]]
    ttfts = [item["ttft_ms"] for item in requests if item["ttft_ms"] is not None]
    tokens = sum(item["tokens"] for item in requests)
    half = len(requests) // 2
    early = [
        (item["wall_ms"] - item["ttft_ms"]) / max(1, item["tokens"] - 1)
        for item in requests[:half]
        if item["ttft_ms"] is not None and item["tokens"] > 1
    ]
    late = [
        (item["wall_ms"] - item["ttft_ms"]) / max(1, item["tokens"] - 1)
        for item in requests[half:]
        if item["ttft_ms"] is not None and item["tokens"] > 1
    ]
    accepted = []
    for earlier, later in zip(samples, samples[1:]):
        drafted = later["drafted"] - earlier["drafted"]
        wins = later["accepted"] - earlier["accepted"]
        accepted.append(wins / drafted if drafted else None)
    windows = [value for value in accepted if value is not None]
    first_sample, last_sample = samples[0], samples[-1]
    # Skip the first window: the KV cache is still filling, so its growth is
    # the working set, not a leak.
    steady = samples[1] if len(samples) > 2 else samples[0]
    return {
        "mode": mode,
        "requests": len(requests),
        "tokens": tokens,
        "per_token_ms": {
            "p50": percentile(per_token, 0.50),
            "p95": percentile(per_token, 0.95),
            "p99": percentile(per_token, 0.99),
            "max": max(per_token) if per_token else None,
            "mean": statistics.fmean(per_token) if per_token else None,
            "samples": len(per_token),
        },
        "sse_event_gap_ms": {
            "p50": percentile(gaps, 0.50),
            "p99": percentile(gaps, 0.99),
            "note": "transport batching; not comparable across modes",
        },
        "ttft_ms": {
            "p50": percentile(ttfts, 0.50),
            "p99": percentile(ttfts, 0.99),
        },
        "per_token_drift": {
            "early_p99": percentile(early, 0.99),
            "late_p99": percentile(late, 0.99),
        },
        "throughput_tokens_per_second": tokens
        / max(1e-9, sum(item["wall_ms"] for item in requests) / 1000.0),
        "acceptance": {
            "first_window": windows[0] if windows else None,
            "last_window": windows[-1] if windows else None,
            "min_window": min(windows) if windows else None,
            "windows": len(windows),
        },
        "memory": {
            "current_bytes_first": first_sample["current_bytes"],
            "current_bytes_last": last_sample["current_bytes"],
            "current_bytes_steady": steady["current_bytes"],
            "steady_growth_bytes": last_sample["current_bytes"]
            - steady["current_bytes"],
            "observed_resident_first": first_sample["observed_resident"],
            "observed_resident_last": last_sample["observed_resident"],
            "denied_reservations": last_sample["denied"]
            - first_sample["denied"],
        },
        "pressure_levels": sorted({sample["pressure"] for sample in samples}),
        "failed_requests": last_sample["failed"],
        "metal_healthy": last_sample["metal"],
        "restarts": last_sample["restarts"],
    }


def snapshot(port: int) -> dict:
    status = smoke.request(port, "GET", "/status")[1]
    metrics = status["metrics"]
    return {
        "current_bytes": status["memory_actual"]["current_bytes"],
        "observed_resident": status["memory_governor"]["observed_resident_bytes"],
        "denied": status["memory_governor"]["denied_reservations"],
        "pressure": status["memory_pressure"],
        "failed": status["requests"]["failed"],
        "metal": status["metal"]["healthy"],
        "restarts": status["transport"]["restarts"],
        "drafted": metrics["drafted_tokens"],
        "accepted": metrics["accepted_draft_tokens"],
        "decode_tokens": metrics["decode_output_tokens"],
        "decode_wall_ms": metrics["decode_wall_ms"],
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
        samples = [snapshot(server.port)]
        requests: list[dict] = []
        lock = threading.Lock()
        stop = time.monotonic() + arguments.duration
        wave_id = 0
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.width) as pool:
            while time.monotonic() < stop:
                # Rotate prompts so the schedule does not degenerate into cache
                # hits, and tag each wave so KV pages cannot all be reused.
                label, body = PROMPTS[wave_id % arguments.prompts]
                prompt = f"{body}\nSoak tag: {wave_id}. Do not include the tag."
                body_json = smoke.chat_body(
                    arguments.model,
                    prompt,
                    **{
                        **settings,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                )

                def one() -> dict:
                    result = stream_completion(server.port, body_json)
                    result["case"] = label
                    return result

                futures = [pool.submit(one) for _ in range(arguments.width)]
                for future in concurrent.futures.as_completed(futures):
                    with lock:
                        requests.append(future.result())
                wave_id += 1
                if time.monotonic() - started >= arguments.sample_seconds * len(samples):
                    samples.append(snapshot(server.port))
                if wave_id % 10 == 0:
                    print(
                        f"{mode} waves={wave_id} requests={len(requests)} "
                        f"elapsed={time.monotonic() - started:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
        samples.append(snapshot(server.port))
        summary = summarise(mode, requests, samples)
        summary["mtp_batch"] = native_env["SPLASH_MTP_BATCH"]
        summary["wall_seconds"] = time.monotonic() - started
        summary["build_id"] = smoke.request(server.port, "GET", "/status")[1][
            "identity"
        ]["cache"]["build_id"]
        return summary
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
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--sample-seconds", type=float, default=10.0)
    parser.add_argument("--cache-gib", type=float, default=18.0)
    # Every wave uses a fresh prompt tag, so the KV cache legitimately grows
    # with the working set; this bound catches runaway growth, not cache fill.
    # Cross-mode growth is checked separately in gate().
    parser.add_argument("--max-growth-gib", type=float, default=4.0)
    parser.add_argument("--max-itl-regression", type=float, default=0.25)
    parser.add_argument(
        "--output", type=Path, default=Path("build/verification/mtp-soak.json")
    )
    arguments = smoke.resolve_server_arguments(parser.parse_args(argv))
    if not 1 <= arguments.width <= 4:
        parser.error("--width must be in [1, 4]")
    if arguments.duration < arguments.sample_seconds * 2:
        parser.error("--duration must cover at least two status windows")
    return arguments


def gate(off: dict, on: dict, arguments) -> tuple[bool, list[str]]:
    """Tail latency may not regress beyond the allowance; nothing may break."""

    reasons = []
    for summary in (off, on):
        if summary["failed_requests"]:
            reasons.append(f"{summary['mode']}: {summary['failed_requests']} failed")
        if not summary["metal_healthy"]:
            reasons.append(f"{summary['mode']}: Metal unhealthy")
        if summary["restarts"]:
            reasons.append(f"{summary['mode']}: runtime restarted")
        if "critical" in summary["pressure_levels"]:
            reasons.append(f"{summary['mode']}: critical memory pressure")
        growth = summary["memory"]["steady_growth_bytes"] / float(1 << 30)
        if growth > arguments.max_growth_gib:
            reasons.append(
                f"{summary['mode']}: steady-state memory grew {growth:.2f} GiB"
            )
        if summary["memory"]["denied_reservations"] > 0:
            reasons.append(
                f"{summary['mode']}: "
                f"{summary['memory']['denied_reservations']} denied reservations"
            )
    on_growth = on["memory"]["steady_growth_bytes"]
    off_growth = off["memory"]["steady_growth_bytes"]
    # Drafting must not add steady-state growth over the same workload.
    if on_growth > off_growth * 1.5 + (256 << 20):
        reasons.append(
            f"on-mode grew {(on_growth - off_growth) / (1 << 30):.2f} GiB more "
            "than drafting-off"
        )
    off_p99 = off["per_token_ms"]["p99"]
    on_p99 = on["per_token_ms"]["p99"]
    if off_p99 and on_p99 and on_p99 > off_p99 * (1.0 + arguments.max_itl_regression):
        reasons.append(
            f"p99 per-token latency regressed {off_p99:.1f} -> {on_p99:.1f} ms "
            f"(> {arguments.max_itl_regression:.0%})"
        )
    return (not reasons), reasons


def main(argv=None) -> int:
    arguments = parse_args(argv)
    settings = {
        "max_completion_tokens": arguments.max_tokens,
        "temperature": arguments.temperature,
        "top_p": arguments.top_p,
        "seed": arguments.seed,
    }
    off = run_mode(arguments, "off", settings)
    on = run_mode(arguments, "on", settings)
    passed, reasons = gate(off, on, arguments)
    document = {
        "schema_version": 1,
        "timing": "client-observed SSE arrival times; includes HTTP transport",
        "width": arguments.width,
        "duration_seconds": arguments.duration,
        "settings": settings,
        "off": off,
        "on": on,
        "gate": {"pass": passed, "reasons": reasons},
        "criterion": (
            "no failures, no restart, bounded steady-state memory, "
            f"p99 per-token latency within {arguments.max_itl_regression:.0%} "
            "of drafting-off"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "off_per_token_ms": off["per_token_ms"],
                "on_per_token_ms": on["per_token_ms"],
                "off_throughput": off["throughput_tokens_per_second"],
                "on_throughput": on["throughput_tokens_per_second"],
                "off_ttft": off["ttft_ms"],
                "on_ttft": on["ttft_ms"],
                "off_acceptance": off["acceptance"],
                "on_acceptance": on["acceptance"],
                "off_memory": off["memory"],
                "on_memory": on["memory"],
                "gate": document["gate"],
            },
            indent=2,
        )
    )
    def label(summary: dict) -> str:
        value = summary["per_token_ms"]["p99"]
        return "n/a" if value is None else f"{value:.1f}ms"

    print(
        f"mtp soak: {'PASS' if passed else 'FAIL'} width={arguments.width} "
        f"p99 per-token off={label(off)} on={label(on)} "
        f"tokens off={off['tokens']} on={on['tokens']}"
        + ("" if passed else f" reasons={reasons}"),
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
