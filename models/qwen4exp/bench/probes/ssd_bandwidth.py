#!/usr/bin/env python3
"""SSD throughput for expert-sized random reads, bypassing the file cache.

Reads random 2.76 MB experts out of the real layer files with F_NOCACHE (the
macOS equivalent of O_DIRECT, which llama.cpp's --moe-stream-direct uses), at
several levels of parallelism. This is the ceiling for fetching cache misses.
"""

import fcntl
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path.home() / "models/qwen38-flash-next-splash/target"
EXPERT = int(3 * 640 * 2560 * 0.5625)
F_NOCACHE = 48
files = []
for i in range(48):
    fd = os.open(ROOT / f"layer-{i}.bin", os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    files.append((fd, os.fstat(fd).st_size))
align = 16384


def read_one(_):
    fd, size = random.choice(files)
    offset = random.randrange(0, (size - EXPERT) // align) * align
    return len(os.pread(fd, EXPERT, offset))


for threads in (1, 2, 4, 8, 16):
    count = 64 * threads if threads > 1 else 48
    with ThreadPoolExecutor(threads) as pool:
        start = time.perf_counter()
        total = sum(pool.map(read_one, range(count)))
        elapsed = time.perf_counter() - start
    print(
        f"{threads:2d} readers: {total / elapsed / 1e9:5.2f} GB/s  "
        f"({elapsed / count * 1000 * threads:5.2f} ms per expert per reader)",
        flush=True,
    )
