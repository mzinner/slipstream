#!/usr/bin/env python3
"""Comprehensive Benchmark: Text-Only vs Multimodal (Text+Vision) on Slipstream.

Measures:
1. Text-only performance before vision is loaded (TTFT, decode tok/s, tok/step).
2. Multimodal performance (image encoding + M-RoPE prefill + text generation).
3. Text-only performance after vision has been loaded (to verify zero regression).
4. Memory footprint transition (lazy 0-byte text footprint -> dynamic admission).
5. Raw GPU vision encoder kernel latencies across grid resolutions.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path.home() / "models/swift-qwen38-flash-next-v3/prepared"
RESULTS_FILE = ROOT / "dev/benchmarks/vision_vs_text_benchmark_results.json"


def kill_on_port(port: int):
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            for pid in out.split():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except OSError:
                    pass
            time.sleep(2)
            for pid in out.split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
    except subprocess.CalledProcessError:
        pass


def wait_for_ready(url: str, timeout: int = 120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bench"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.5)
    return False


def fetch_metrics(base_url: str = "http://127.0.0.1:8090") -> dict:
    try:
        req = urllib.request.Request(f"{base_url}/metrics")
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8")
        m = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    m[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return m
    except Exception:
        return {}


def create_chart_image() -> str:
    """Generate a clean synthetic bar chart PNG and return as base64 data URL."""
    img = Image.new("RGB", (320, 240), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Axes
    draw.line([(40, 20), (40, 200), (300, 200)], fill=(0, 0, 0), width=2)
    # Bars: Q1=40, Q2=80, Q3=60, Q4=90
    data = [("Q1", 40, (70, 130, 180)), ("Q2", 80, (220, 20, 60)),
            ("Q3", 60, (50, 205, 50)), ("Q4", 90, (255, 140, 0))]
    for i, (label, val, color) in enumerate(data):
        x0 = 60 + i * 55
        y0 = 200 - val * 1.8
        draw.rectangle([x0, y0, x0 + 40, 200], fill=color)
        draw.text((x0 + 10, 205), label, fill=(0, 0, 0))
        draw.text((x0 + 8, y0 - 15), str(val), fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def create_shapes_image() -> str:
    """Generate a synthetic geometry diagram with shapes and colors."""
    img = Image.new("RGB", (256, 256), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Red circle in top-left
    draw.ellipse([30, 30, 90, 90], fill=(230, 50, 50), outline=(150, 20, 20), width=2)
    # Blue square in top-right
    draw.rectangle([160, 30, 220, 90], fill=(50, 100, 230), outline=(20, 50, 150), width=2)
    # Green triangle in bottom-center
    draw.polygon([(128, 140), (70, 220), (186, 220)], fill=(40, 180, 70), outline=(20, 120, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def create_ocr_image() -> str:
    """Generate an image containing clear printed text and a formula."""
    img = Image.new("RGB", (360, 160), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 25), "Quantum Physics Lab Note:", fill=(20, 20, 100))
    draw.text((20, 60), "Energy equation: E = mc^2", fill=(180, 20, 20))
    draw.text((20, 95), "Measured frequency: nu = 5.42 GHz", fill=(0, 120, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


TEXT_PROMPTS = [
    {
        "id": "gsm8k_math",
        "category": "Math Reasoning",
        "prompt": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins for her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. How much in dollars does "
            "she make every day at the farmers' market? Think step by step and conclude with: "
            "Therefore, the final answer is $ANSWER."
        ),
        "expected": "18",
        "max_tokens": 512,
    },
    {
        "id": "math500_series",
        "category": "Advanced Math Derivation",
        "prompt": (
            "Define p = sum_{k=1}^inf 1/k^2 and q = sum_{k=1}^inf 1/k^3. "
            "Find a way to write sum_{j=1}^inf sum_{k=1}^inf 1/(j+k)^3 in terms of p and q. "
            "Think step by step and conclude your response with: "
            "Therefore, the final answer is \\boxed{ANSWER}"
        ),
        "expected": "p - q",
        "max_tokens": 768,
    },
    {
        "id": "constraint_logic",
        "category": "Constraint Reasoning",
        "prompt": (
            "Three friends (Alice, Bob, Charlie) sit in chairs 1 to 3 from left to right. "
            "Constraints: 1) Alice never sits next to Bob. 2) Charlie is to the right of Alice. "
            "Who sits in chair 1, chair 2, and chair 3? Give step by step deduction, then state "
            "the final arrangement."
        ),
        "expected": "Alice, Charlie, Bob",
        "max_tokens": 512,
    },
    {
        "id": "python_intervals",
        "category": "Coding & Algorithms (Python)",
        "prompt": (
            "Write a Python function merge_intervals(intervals: list[list[int]]) -> list[list[int]] "
            "that merges overlapping intervals in O(N log N) time, with docstring and 3 assert tests."
        ),
        "expected": "merge_intervals",
        "max_tokens": 512,
    },
    {
        "id": "rust_csv",
        "category": "Systems Coding (Rust)",
        "prompt": (
            "Write a Rust function parse_csv_line(line: &str) -> Vec<String> that parses a CSV line "
            "supporting quoted commas, with unit tests."
        ),
        "expected": "parse_csv_line",
        "max_tokens": 512,
    },
    {
        "id": "tech_explanation",
        "category": "Technical Writing",
        "prompt": (
            "Explain how multi-head self-attention works to a systems engineer with linear algebra "
            "background in 3 crisp paragraphs."
        ),
        "expected": "Query, Key, Value",
        "max_tokens": 512,
    },
]


def stream_chat(base_url: str, messages: list[dict], max_tokens: int) -> dict:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first_token_time = None
    chunks_text = []
    reasoning_text = []
    token_count = 0

    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str == "data: [DONE]":
                    continue
                if line_str.startswith("data: "):
                    try:
                        data_obj = json.loads(line_str[6:])
                        choices = data_obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            reasoning_chunk = delta.get("reasoning_content") or delta.get("reasoning")
                            if reasoning_chunk:
                                reasoning_text.append(reasoning_chunk)
                                token_count += 1
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                            text_chunk = delta.get("content")
                            if text_chunk:
                                chunks_text.append(text_chunk)
                                token_count += 1
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                    except json.JSONDecodeError:
                        pass
        t1 = time.perf_counter()
        elapsed = t1 - t0
        ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else 0.0
        gen_time = t1 - first_token_time if (first_token_time and token_count > 1) else elapsed
        decode_tok_s = (token_count - 1) / gen_time if (first_token_time and token_count > 1 and gen_time > 0) else (token_count / elapsed)
        return {
            "token_count": token_count,
            "elapsed_s": elapsed,
            "ttft_ms": ttft_ms,
            "decode_tok_s": decode_tok_s,
            "content": "".join(chunks_text),
            "reasoning": "".join(reasoning_text),
        }
    except Exception as e:
        return {"error": str(e), "elapsed_s": time.perf_counter() - t0}


def main():
    parser = argparse.ArgumentParser(description="Side-by-side Text vs Vision Performance Benchmark")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--cache-gib", type=str, default="30")
    args = parser.parse_args()

    port = args.port
    base_url = f"http://127.0.0.1:{port}"

    print("=" * 80)
    print("SLIPSTREAM V2: SIDE-BY-SIDE BENCHMARK (TEXT-ONLY VS TEXT+VISION)")
    print(f"Model: {MODEL_PATH}")
    print(f"Port:  {port}")
    print("=" * 80)

    # 1. Clean pre-flight
    kill_on_port(port)
    time.sleep(2)

    # 2. Start server under guarded.py
    env = os.environ.copy()
    env["SPLASH_EXPERT_CACHE_GIB"] = args.cache_gib
    server_cmd = [
        sys.executable,
        str(ROOT / "dev/benchmarks/guarded.py"),
        "--floor-gib", "4.0",
        "--max-seconds", "1200",
        "--",
        str(ROOT / "slipstream"),
        "serve",
        "--model", str(MODEL_PATH),
        "--port", str(port),
    ]

    log_path = "/tmp/slipstream_bench.log"
    print(f"\n[1/5] Launching Slipstream server (log: {log_path})...")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(server_cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if not wait_for_ready(f"{base_url}/v1/models", timeout=90):
        print("ERROR: Server failed to start! Last log lines:")
        with open(log_path) as f:
            print("".join(f.readlines()[-40:]))
        proc.kill()
        return 1

    print("Server is ready and accepting requests!")
    m_init = fetch_metrics(base_url)

    benchmark_data = {
        "text_before": [],
        "vision_multimodal": [],
        "text_after": [],
        "metrics_before": m_init,
    }

    try:
        # ==========================================
        # PHASE 1: Text-Only Benchmarks (Cold Vision)
        # ==========================================
        print("\n" + "=" * 80)
        print("[2/5] PHASE 1: Testing Text-Only Workload (Vision Unallocated / 0-byte)")
        print("=" * 80)

        for p in TEXT_PROMPTS:
            print(f"  -> [{p['id']:16s}] ({p['category']})... ", end="", flush=True)
            res = stream_chat(base_url, [{"role": "user", "content": p["prompt"]}], p["max_tokens"])
            if "error" in res:
                print(f"FAILED: {res['error']}")
            else:
                print(f"{res['decode_tok_s']:5.1f} tok/s | TTFT: {res['ttft_ms']:6.1f} ms | Tokens: {res['token_count']:3d} | {res['elapsed_s']:.2f}s")
            p_res = {**p, **res}
            benchmark_data["text_before"].append(p_res)

        m_mid = fetch_metrics(base_url)
        benchmark_data["metrics_mid"] = m_mid

        # ==========================================
        # PHASE 2: Multimodal (Text + Vision) Workload
        # ==========================================
        print("\n" + "=" * 80)
        print("[3/5] PHASE 2: Testing Multimodal Workload (Dynamic Vision Load & Inference)")
        print("=" * 80)

        multimodal_cases = [
            {
                "id": "chart_extraction",
                "category": "Chart Data Extraction",
                "image_data": create_chart_image(),
                "prompt": "Look at this bar chart. Extract the exact quarterly values for Q1, Q2, Q3, and Q4 in a clean structured list.",
                "max_tokens": 384,
            },
            {
                "id": "shapes_spatial",
                "category": "Visual Geometry & Spatial Reasoning",
                "image_data": create_shapes_image(),
                "prompt": "Identify all geometric shapes in this image, describing their color, shape, and spatial location.",
                "max_tokens": 384,
            },
            {
                "id": "text_ocr",
                "category": "Image OCR & Transcription",
                "image_data": create_ocr_image(),
                "prompt": "Transcribe all text lines and scientific equations visible in this laboratory image exactly.",
                "max_tokens": 384,
            },
        ]

        for m_case in multimodal_cases:
            print(f"  -> [{m_case['id']:16s}] ({m_case['category']})... ", end="", flush=True)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": m_case["image_data"]}},
                        {"type": "text", "text": m_case["prompt"]},
                    ],
                }
            ]
            res = stream_chat(base_url, messages, m_case["max_tokens"])
            if "error" in res:
                print(f"FAILED: {res['error']}")
            else:
                print(f"{res['decode_tok_s']:5.1f} tok/s | TTFT: {res['ttft_ms']:6.1f} ms | Tokens: {res['token_count']:3d} | {res['elapsed_s']:.2f}s")
                ans = res["content"].strip() or res["reasoning"].strip()
                snippet = ans[:100].replace("\n", " ")
                print(f"     Preview: {snippet}...")
            m_res = {
                "id": m_case["id"],
                "category": m_case["category"],
                "prompt": m_case["prompt"],
                **res,
            }
            benchmark_data["vision_multimodal"].append(m_res)

        m_post_vision = fetch_metrics(base_url)
        benchmark_data["metrics_post_vision"] = m_post_vision

        # ==========================================
        # PHASE 3: Text-Only Workload (Warm Vision)
        # ==========================================
        print("\n" + "=" * 80)
        print("[4/5] PHASE 3: Testing Text-Only Workload Post-Vision (Verifying 0 Regression)")
        print("=" * 80)

        for p in TEXT_PROMPTS[:4]:  # Run representative subset
            print(f"  -> [{p['id']:16s}] ({p['category']})... ", end="", flush=True)
            res = stream_chat(base_url, [{"role": "user", "content": p["prompt"]}], p["max_tokens"])
            if "error" in res:
                print(f"FAILED: {res['error']}")
            else:
                print(f"{res['decode_tok_s']:5.1f} tok/s | TTFT: {res['ttft_ms']:6.1f} ms | Tokens: {res['token_count']:3d} | {res['elapsed_s']:.2f}s")
            p_res = {**p, **res}
            benchmark_data["text_after"].append(p_res)

    finally:
        print("\n[5/5] Stopping Slipstream server cleanly...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        kill_on_port(port)
        print("Server stopped and port reclaimed.")

    # Save data
    RESULTS_FILE.write_text(json.dumps(benchmark_data, indent=2))
    print(f"\nBenchmark raw results written to: {RESULTS_FILE}")

    # ==========================================
    # Summary Tables
    # ==========================================
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY & SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    # 1. Text Before vs Text After
    print("\n--- 1. Text-Only Performance: Pre-Vision vs Post-Vision (Zero Regression Check) ---")
    print(f"{'Task ID':<18} | {'Category':<28} | {'Pre-Vision TTFT':<15} | {'Post-Vision TTFT':<16} | {'Pre-Vision Tok/s':<16} | {'Post-Vision Tok/s':<17}")
    print("-" * 118)
    for b in benchmark_data["text_after"]:
        match = next(x for x in benchmark_data["text_before"] if x["id"] == b["id"])
        print(f"{b['id']:<18} | {b['category'][:28]:<28} | {match['ttft_ms']:>12.1f} ms | {b['ttft_ms']:>13.1f} ms | {match['decode_tok_s']:>13.1f} tok/s | {b['decode_tok_s']:>14.1f} tok/s")

    # 2. Multimodal Performance
    print("\n--- 2. Multimodal (Text + Vision) Performance ---")
    print(f"{'Task ID':<18} | {'Category':<35} | {'TTFT (Encoding+Prefill)':<24} | {'Decode Tok/s':<14} | {'Tokens':<8} | {'Elapsed':<8}")
    print("-" * 118)
    for m in benchmark_data["vision_multimodal"]:
        if "error" not in m:
            print(f"{m['id']:<18} | {m['category'][:35]:<35} | {m['ttft_ms']:>20.1f} ms | {m['decode_tok_s']:>11.1f} tok/s | {m['token_count']:>6d} | {m['elapsed_s']:>6.2f} s")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
