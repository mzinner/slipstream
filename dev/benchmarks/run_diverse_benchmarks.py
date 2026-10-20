#!/usr/bin/env python3
"""Run a diverse suite of 8 text prompts and 6 vision prompts on Slipstream.

Measures TTFT (ms), decode throughput (tok/s), elapsed time, token count,
and displays side-by-side performance across modalities and categories.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path.home() / "models/swift-qwen38-flash-next-v3/prepared"
RESULTS_FILE = ROOT / "dev/benchmarks/diverse_benchmark_results.json"


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


# ---------------------------------------------------------------------------
# Image Generators
# ---------------------------------------------------------------------------

def to_base64_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def create_financial_bar_chart() -> str:
    """Multi-series bar chart: Revenue vs Expenses for Q1-Q4."""
    img = Image.new("RGB", (420, 260), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # Axes
    d.line([(50, 30), (50, 210), (390, 210)], fill=(40, 40, 40), width=2)
    # Title & Legend
    d.text((120, 10), "Quarterly Financial Performance ($k)", fill=(0, 0, 0))
    d.rectangle([250, 32, 265, 42], fill=(41, 128, 185))
    d.text((272, 30), "Revenue", fill=(0, 0, 0))
    d.rectangle([330, 32, 345, 42], fill=(231, 76, 60))
    d.text((352, 30), "Expenses", fill=(0, 0, 0))

    data = [
        ("Q1", 120, 80),
        ("Q2", 150, 95),
        ("Q3", 180, 110),
        ("Q4", 210, 125),
    ]
    for i, (q, rev, exp) in enumerate(data):
        x = 75 + i * 80
        # Revenue bar (blue)
        h_rev = int(rev * 0.7)
        d.rectangle([x, 210 - h_rev, x + 25, 210], fill=(41, 128, 185))
        d.text((x + 2, 210 - h_rev - 14), str(rev), fill=(41, 128, 185))
        # Expense bar (red)
        h_exp = int(exp * 0.7)
        d.rectangle([x + 28, 210 - h_exp, x + 53, 210], fill=(231, 76, 60))
        d.text((x + 30, 210 - h_exp - 14), str(exp), fill=(231, 76, 60))
        # Quarter label
        d.text((x + 18, 216), q, fill=(0, 0, 0))
    return to_base64_data_url(img)


def create_architecture_diagram() -> str:
    """System architecture diagram showing components and data flow."""
    img = Image.new("RGB", (480, 260), (248, 249, 250))
    d = ImageDraw.Draw(img)
    d.text((150, 10), "E-Commerce Checkout Architecture", fill=(20, 20, 20))

    # Boxes: (x0, y0, x1, y1, label, color)
    boxes = [
        (20, 100, 100, 150, "Client / Web", (52, 152, 219)),
        (130, 100, 220, 150, "API Gateway", (155, 89, 182)),
        (260, 40, 360, 85, "Auth Service", (46, 204, 113)),
        (260, 100, 360, 145, "Order Service", (46, 204, 113)),
        (260, 160, 360, 205, "Payment Svc", (46, 204, 113)),
        (400, 50, 470, 95, "Redis Cache", (230, 126, 34)),
        (400, 130, 470, 185, "PostgreSQL", (52, 73, 94)),
    ]
    for x0, y0, x1, y1, text, col in boxes:
        d.rectangle([x0, y0, x1, y1], fill=col, outline=(30, 30, 30), width=1)
        d.text((x0 + 8, y0 + 16), text, fill=(255, 255, 255))

    # Arrows (lines with simple heads)
    arrows = [
        ((100, 125), (130, 125)),
        ((220, 115), (260, 65)),
        ((220, 125), (260, 125)),
        ((220, 135), (260, 185)),
        ((360, 65), (400, 70)),
        ((360, 125), (400, 150)),
        ((360, 185), (400, 165)),
    ]
    for start, end in arrows:
        d.line([start, end], fill=(50, 50, 50), width=2)
    return to_base64_data_url(img)


def create_pie_chart() -> str:
    """Pie chart showing cloud infrastructure market share."""
    img = Image.new("RGB", (380, 250), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((90, 12), "Cloud Infrastructure Market Share", fill=(0, 0, 0))

    # Slices: (start_deg, end_deg, color, label, pct)
    slices = [
        (0, 137, (52, 152, 219), "AWS", "38%"),
        (137, 245, (46, 204, 113), "Azure", "30%"),
        (245, 317, (241, 196, 15), "GCP", "20%"),
        (317, 360, (231, 76, 60), "Others", "12%"),
    ]
    for s_deg, e_deg, color, _, _ in slices:
        d.pieslice([40, 45, 220, 225], start=s_deg, end=e_deg, fill=color, outline=(255, 255, 255), width=2)

    # Legend
    for i, (_, _, color, label, pct) in enumerate(slices):
        y = 65 + i * 35
        d.rectangle([250, y, 268, y + 18], fill=color)
        d.text((276, y + 2), f"{label}: {pct}", fill=(20, 20, 20))
    return to_base64_data_url(img)


def create_geometric_shapes() -> str:
    """Scatter of colored geometric shapes in 4 quadrants."""
    img = Image.new("RGB", (320, 320), (245, 247, 250))
    d = ImageDraw.Draw(img)
    # Quadrant divider lines
    d.line([(160, 10), (160, 310)], fill=(210, 215, 220), width=1)
    d.line([(10, 160), (310, 160)], fill=(210, 215, 220), width=1)

    # Quadrant 1 (Top-Left): 2 Red Stars
    for cx, cy in [(60, 60), (110, 100)]:
        pts = []
        for a in range(10):
            r = 18 if a % 2 == 0 else 8
            ang = a * math.pi / 5 - math.pi / 2
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        d.polygon(pts, fill=(231, 76, 60), outline=(180, 40, 30))

    # Quadrant 2 (Top-Right): 3 Blue Circles
    for cx, cy in [(200, 50), (260, 70), (220, 120)]:
        d.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(52, 152, 219), outline=(30, 100, 180))

    # Quadrant 3 (Bottom-Left): 2 Green Triangles
    for cx, cy in [(50, 240), (120, 260)]:
        d.polygon([(cx, cy - 18), (cx - 16, cy + 14), (cx + 16, cy + 14)], fill=(46, 204, 113), outline=(30, 140, 70))

    # Quadrant 4 (Bottom-Right): 2 Yellow Squares
    for x, y in [(190, 200), (250, 240)]:
        d.rectangle([x, y, x + 30, y + 30], fill=(241, 196, 15), outline=(190, 150, 10), width=2)
    return to_base64_data_url(img)


def create_invoice_table() -> str:
    """Clean rendered billing/invoice table with headers and numbers."""
    img = Image.new("RGB", (440, 220), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 15), "Invoice #INV-2026-8819", fill=(10, 10, 10))
    d.text((320, 15), "Date: 2026-10-19", fill=(80, 80, 80))

    # Table Header
    d.rectangle([20, 45, 420, 72], fill=(236, 240, 241))
    d.text((30, 52), "SKU", fill=(0, 0, 0))
    d.text((110, 52), "Description", fill=(0, 0, 0))
    d.text((250, 52), "Qty", fill=(0, 0, 0))
    d.text((310, 52), "Price", fill=(0, 0, 0))
    d.text((370, 52), "Total", fill=(0, 0, 0))

    rows = [
        ("SKU-101", "NVMe SSD 2TB", "2", "$140.00", "$280.00"),
        ("SKU-205", "DDR5 RAM 64GB", "1", "$190.00", "$190.00"),
        ("SKU-312", "USB-C Hub 10Gbps", "3", "$45.00", "$135.00"),
    ]
    for i, (sku, desc, qty, price, tot) in enumerate(rows):
        y = 80 + i * 28
        d.line([(20, y - 5), (420, y - 5)], fill=(220, 220, 220), width=1)
        d.text((30, y), sku, fill=(50, 50, 50))
        d.text((110, y), desc, fill=(20, 20, 20))
        d.text((255, y), qty, fill=(50, 50, 50))
        d.text((310, y), price, fill=(50, 50, 50))
        d.text((370, y), tot, fill=(20, 20, 20))

    # Total row
    d.line([(20, 168), (420, 168)], fill=(40, 40, 40), width=2)
    d.text((280, 178), "Grand Total:", fill=(0, 0, 0))
    d.text((370, 178), "$605.00", fill=(180, 20, 20))
    return to_base64_data_url(img)


def create_scientific_plot() -> str:
    """Scientific plot of a damped sinusoidal oscillation."""
    img = Image.new("RGB", (400, 240), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((100, 12), "Damped Oscillator: y(t) = e^(-0.25t) * cos(2t)", fill=(0, 0, 0))

    # Axes
    d.line([(40, 20), (40, 200), (380, 200)], fill=(0, 0, 0), width=1)
    d.line([(40, 110), (380, 110)], fill=(200, 200, 200), width=1)
    d.text((15, 105), "0", fill=(100, 100, 100))
    d.text((15, 40), "+1", fill=(100, 100, 100))
    d.text((15, 175), "-1", fill=(100, 100, 100))
    d.text((370, 205), "t (s)", fill=(0, 0, 0))

    # Curve points
    pts = []
    for px in range(40, 370):
        t = (px - 40) / 30.0
        y_val = math.exp(-0.25 * t) * math.cos(2.0 * t)
        py = 110 - int(y_val * 70)
        pts.append((px, py))
    d.line(pts, fill=(142, 68, 173), width=2)
    return to_base64_data_url(img)


# ---------------------------------------------------------------------------
# Prompts Definition
# ---------------------------------------------------------------------------

DIVERSE_TEXT_PROMPTS = [
    {
        "id": "math_arithmetic",
        "category": "Math Reasoning (Arithmetic)",
        "prompt": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins for her friends every day with four. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. How much in dollars does "
            "she make every day at the farmers' market? Think step by step and conclude with: "
            "Therefore, the final answer is $ANSWER."
        ),
        "max_tokens": 384,
    },
    {
        "id": "math_derivation",
        "category": "Advanced Math (Symbolic Derivation)",
        "prompt": (
            "Define p = sum_{k=1}^inf 1/k^2 and q = sum_{k=1}^inf 1/k^3. "
            "Find a way to write sum_{j=1}^inf sum_{k=1}^inf 1/(j+k)^3 in terms of p and q. "
            "Think step by step and conclude your response with: "
            "Therefore, the final answer is \\boxed{ANSWER}"
        ),
        "max_tokens": 512,
    },
    {
        "id": "logic_puzzle",
        "category": "Constraint Satisfaction Logic",
        "prompt": (
            "Five colleagues (A, B, C, D, E) are assigned to contiguous offices 1 to 5 in a row. "
            "Constraints:\n"
            "1. C is in office 3.\n"
            "2. A is immediately to the left of B (A's office number is B - 1).\n"
            "3. D is not in office 1 or 5.\n"
            "4. E is to the right of C.\n"
            "Deduce the unique office assignment for all five colleagues step by step."
        ),
        "max_tokens": 448,
    },
    {
        "id": "rust_ring_buffer",
        "category": "Systems Programming (Rust)",
        "prompt": (
            "Write a lock-free or single-atomic circular ring buffer struct in Rust `CircularBuffer<T, const N: usize>` "
            "with `push` and `pop` methods. Include safety docstrings and a unit test."
        ),
        "max_tokens": 512,
    },
    {
        "id": "python_algorithms",
        "category": "Algorithms & Coding (Python)",
        "prompt": (
            "Write a Python function `find_cheapest_flight(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int` "
            "using Bellman-Ford or Dijkstra to find the cheapest flight from src to dst with at most k stops. "
            "Include type hints and 2 test assertions."
        ),
        "max_tokens": 448,
    },
    {
        "id": "sql_cohorts",
        "category": "Data Engineering (SQL)",
        "prompt": (
            "Write an ANSI SQL query that computes monthly customer retention cohorts over an `orders(customer_id, order_date, amount)` table. "
            "The output should show: cohort_month, months_since_first_order, active_customers, and retention_rate. "
            "Explain the window functions used."
        ),
        "max_tokens": 448,
    },
    {
        "id": "distributed_systems",
        "category": "Technical Architecture Writing",
        "prompt": (
            "Compare the leader election and log commitment mechanics between the Raft consensus protocol and Multi-Paxos. "
            "Explain how Raft avoids conflicting uncommitted log entries in 3 concise paragraphs."
        ),
        "max_tokens": 448,
    },
    {
        "id": "cpp_code_review",
        "category": "Security & Concurrency Review",
        "prompt": (
            "Review this C++ concurrency snippet:\n"
            "```cpp\n"
            "template<typename K, typename V>\n"
            "class Cache {\n"
            "  std::unordered_map<K, V*> map;\n"
            "  std::mutex mtx;\n"
            "public:\n"
            "  V* get(const K& k) {\n"
            "    std::lock_guard lock(mtx);\n"
            "    return map[k];\n"
            "  }\n"
            "  void set(const K& k, V* v) {\n"
            "    std::lock_guard lock(mtx);\n"
            "    delete map[k];\n"
            "    map[k] = v;\n"
            "  }\n"
            "};\n"
            "```\n"
            "Identify at least 3 serious bugs, undefined behavior, or memory leak risks, and provide the modern RAII C++20 fix."
        ),
        "max_tokens": 512,
    },
]

DIVERSE_VISION_PROMPTS = [
    {
        "id": "financial_barchart",
        "category": "Financial Analytics (Multi-Bar Chart)",
        "image_func": create_financial_bar_chart,
        "prompt": (
            "Analyze this quarterly financial chart showing Revenue and Expenses for Q1 through Q4. "
            "1. Extract the exact Revenue and Expense figures for each quarter.\n"
            "2. Calculate the Net Profit (Revenue - Expenses) for each quarter.\n"
            "3. State which quarter achieved the highest profit margin."
        ),
        "max_tokens": 448,
    },
    {
        "id": "architecture_flowchart",
        "category": "System Architecture (Diagram)",
        "image_func": create_architecture_diagram,
        "prompt": (
            "Look at this system architecture diagram. "
            "1. Trace the path of an incoming user request from Client to Data Stores.\n"
            "2. List all microservices shown in the middle tier.\n"
            "3. Identify which data store is connected to each service."
        ),
        "max_tokens": 448,
    },
    {
        "id": "market_pie_chart",
        "category": "Proportional Reasoning (Pie Chart)",
        "image_func": create_pie_chart,
        "prompt": (
            "Look at this cloud infrastructure market share chart. "
            "1. Extract the percentage share for AWS, Azure, GCP, and Others.\n"
            "2. Compute the combined market share of the top two providers."
        ),
        "max_tokens": 384,
    },
    {
        "id": "geometric_spatial_scatter",
        "category": "Spatial Geometry & Object Counting",
        "image_func": create_geometric_shapes,
        "prompt": (
            "Examine this 4-quadrant geometric diagram carefully: "
            "1. Count how many red stars are in the top-left quadrant.\n"
            "2. Count how many blue circles are in the top-right quadrant.\n"
            "3. What shapes are located in the bottom-left and bottom-right quadrants?"
        ),
        "max_tokens": 384,
    },
    {
        "id": "invoice_table_ocr",
        "category": "Tabular OCR & Arithmetic Verification",
        "image_func": create_invoice_table,
        "prompt": (
            "Transcribe this invoice table into clean Markdown. "
            "Verify whether the item totals (Qty * Price) sum correctly to the Grand Total shown."
        ),
        "max_tokens": 384,
    },
    {
        "id": "scientific_curve_plot",
        "category": "Scientific Plot & Curve Analysis",
        "image_func": create_scientific_plot,
        "prompt": (
            "Analyze the plotted curve in this scientific figure: "
            "1. State the mathematical formula shown in the title.\n"
            "2. Describe the physical phenomenon this equation models.\n"
            "3. What is the value of the function as t -> infinity?"
        ),
        "max_tokens": 384,
    },
]


# ---------------------------------------------------------------------------
# Streaming Client
# ---------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Expanded Diverse Benchmark: Text & Vision")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--cache-gib", type=str, default="28")
    args = parser.parse_args()

    port = args.port
    base_url = f"http://127.0.0.1:{port}"

    print("=" * 85)
    print("SLIPSTREAM V2: EXPANDED DIVERSE BENCHMARK (8 TEXT PROMPTS + 6 VISION PROMPTS)")
    print(f"Model: {MODEL_PATH}")
    print(f"Port:  {port}")
    print("=" * 85)

    kill_on_port(port)
    time.sleep(2)

    env = os.environ.copy()
    env["SPLASH_EXPERT_CACHE_GIB"] = args.cache_gib
    server_cmd = [
        sys.executable,
        str(ROOT / "dev/benchmarks/guarded.py"),
        "--floor-gib", "3.5",
        "--max-seconds", "1800",
        "--",
        str(ROOT / "slipstream"),
        "serve",
        "--model", str(MODEL_PATH),
        "--port", str(port),
    ]

    log_path = "/tmp/slipstream_diverse_bench.log"
    print(f"\n[1/4] Launching Slipstream server (log: {log_path})...")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(server_cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if not wait_for_ready(f"{base_url}/v1/models", timeout=90):
        print("ERROR: Server failed to start! Check /tmp/slipstream_diverse_bench.log")
        proc.kill()
        return 1

    print("Server ready! Beginning evaluation.")
    all_results = {
        "text_prompts": [],
        "vision_prompts": [],
    }

    try:
        # ===================================================================
        # SECTION 1: 8 Diverse Text Prompts
        # ===================================================================
        print("\n" + "=" * 85)
        print("[2/4] SECTION 1: Evaluating 8 Diverse Text Prompts")
        print("=" * 85)

        for item in DIVERSE_TEXT_PROMPTS:
            print(f"  -> [{item['id']:22s}] ({item['category']})... ", end="", flush=True)
            res = stream_chat(base_url, [{"role": "user", "content": item["prompt"]}], item["max_tokens"])
            if "error" in res:
                print(f"FAILED: {res['error']}")
            else:
                print(f"{res['decode_tok_s']:5.1f} tok/s | TTFT: {res['ttft_ms']:6.1f} ms | Tokens: {res['token_count']:3d} | {res['elapsed_s']:.2f}s")
            all_results["text_prompts"].append({**item, **res})

        # ===================================================================
        # SECTION 2: 6 Diverse Multimodal Vision Prompts
        # ===================================================================
        print("\n" + "=" * 85)
        print("[3/4] SECTION 2: Evaluating 6 Diverse Vision Prompts (Images + Prompts)")
        print("=" * 85)

        for item in DIVERSE_VISION_PROMPTS:
            print(f"  -> [{item['id']:22s}] ({item['category']})... ", end="", flush=True)
            img_url = item["image_func"]()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": item["prompt"]},
                    ],
                }
            ]
            res = stream_chat(base_url, messages, item["max_tokens"])
            if "error" in res:
                print(f"FAILED: {res['error']}")
            else:
                print(f"{res['decode_tok_s']:5.1f} tok/s | TTFT: {res['ttft_ms']:6.1f} ms | Tokens: {res['token_count']:3d} | {res['elapsed_s']:.2f}s")
                ans = res["content"].strip() or res["reasoning"].strip()
                snippet = ans[:110].replace("\n", " ")
                print(f"     Preview: {snippet}...")
            # Exclude raw image data from json output to keep it compact
            save_item = {k: v for k, v in item.items() if k != "image_func"}
            all_results["vision_prompts"].append({**save_item, **res})

    finally:
        print("\n[4/4] Stopping Slipstream server cleanly...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        kill_on_port(port)
        print("Server stopped cleanly. Port reclaimed.")

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved detailed results to: {RESULTS_FILE}")

    # ===================================================================
    # Final Presentation Tables
    # ===================================================================
    print("\n" + "=" * 85)
    print("EXPANDED BENCHMARK SUMMARY TABLE")
    print("=" * 85)

    print("\n--- A. Text-Only Performance across 8 Domains ---")
    print(f"{'Prompt ID':<22} | {'Category':<32} | {'TTFT':<10} | {'Decode':<12} | {'Tokens':<8} | {'Elapsed':<8}")
    print("-" * 102)
    tot_text_toks = 0
    tot_text_time = 0.0
    for t in all_results["text_prompts"]:
        if "error" not in t:
            tot_text_toks += t["token_count"]
            tot_text_time += t["elapsed_s"]
            print(f"{t['id']:<22} | {t['category'][:32]:<32} | {t['ttft_ms']:>6.1f} ms | {t['decode_tok_s']:>7.1f} tok/s | {t['token_count']:>6d} | {t['elapsed_s']:>6.2f} s")
    avg_text_speed = sum(t["decode_tok_s"] for t in all_results["text_prompts"] if "error" not in t) / max(1, len(all_results["text_prompts"]))
    print("-" * 102)
    print(f"{'AVERAGE (TEXT)':<57} | {'—':>6}    | {avg_text_speed:>7.1f} tok/s | {tot_text_toks:>6d} | {tot_text_time:>6.2f} s")

    print("\n--- B. Multimodal (Text + Vision) Performance across 6 Image Tasks ---")
    print(f"{'Prompt ID':<22} | {'Category':<35} | {'TTFT':<10} | {'Decode':<12} | {'Tokens':<8} | {'Elapsed':<8}")
    print("-" * 105)
    tot_vis_toks = 0
    tot_vis_time = 0.0
    for v in all_results["vision_prompts"]:
        if "error" not in v:
            tot_vis_toks += v["token_count"]
            tot_vis_time += v["elapsed_s"]
            print(f"{v['id']:<22} | {v['category'][:35]:<35} | {v['ttft_ms']:>6.1f} ms | {v['decode_tok_s']:>7.1f} tok/s | {v['token_count']:>6d} | {v['elapsed_s']:>6.2f} s")
    avg_vis_speed = sum(v["decode_tok_s"] for v in all_results["vision_prompts"] if "error" not in v) / max(1, len(all_results["vision_prompts"]))
    print("-" * 105)
    print(f"{'AVERAGE (VISION)':<60} | {'—':>6}    | {avg_vis_speed:>7.1f} tok/s | {tot_vis_toks:>6d} | {tot_vis_time:>6.2f} s")

    print("\n" + "=" * 85)


if __name__ == "__main__":
    main()
