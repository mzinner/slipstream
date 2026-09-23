"""Concise console diagnostics without request-body logging."""

import sys
import time


def log_unexpected(error):
    try:
        print_status(
            f"Error · internal_server_error · {type(error).__name__}", error=True
        )
    except Exception:
        pass


def print_status(message, *, error=False):
    print(
        f"{time.strftime('%H:%M:%S')} {message}",
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def print_request(record):
    outcome = record["outcome"]
    if outcome == "error":
        print_status(f"Error · {record.get('error_code', 'runtime_error')}", error=True)
        return
    metrics = record.get("metrics", {})
    latency = metrics.get("request_latency", {})
    parts = [
        "Cancelled" if outcome == "cancelled" else "Done",
        f"input {record['prompt_tokens']:,}",
        f"cached {metrics.get('cache', {}).get('matched_tokens', 0):,}",
        f"output {record.get('completion_tokens', 0):,}",
    ]
    if record.get("thinking"):
        parts.append(f"think {record['thinking']}")
    tools = record.get("tools")
    if isinstance(tools, dict) and tools.get("count"):
        parts.append(f"tools {tools['count']}·{tools.get('signature', '')}")
    ttft = latency.get("ttft_ms")
    speed = latency.get("stream_tokens_per_second")
    if ttft is not None:
        parts.append(f"TTFT {ttft / 1000:.1f}s")
    # Prompt speed: the new (uncached) prompt tokens over the time from the
    # engine starting the request to its first token (queueing excluded).
    new_tokens = metrics.get("prefill", {}).get("tokens", 0)
    to_first = latency.get("start_to_first_token_ms")
    if new_tokens and to_first:
        parts.append(f"prompt {new_tokens * 1000 / to_first:,.0f} tok/s")
    if speed is not None:
        parts.append(f"decode {speed:.1f} tok/s")
    print_status(" · ".join(parts))
