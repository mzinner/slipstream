"""Does the CPU reference in the test match the real module's output?"""

import struct
import sys

import numpy as np

f = open(sys.argv[1], "rb")
assert f.read(8) == b"HCREF001"
ROWS, HIDDEN, HC, LOWRANK = struct.unpack("<4I", f.read(16))


def blob():
    (n,) = struct.unpack("<Q", f.read(8))
    return np.frombuffer(f.read(n), dtype="<u2")


def f32(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


names = ["x", "gain", "down", "up", "inject", "y", "mixed", "inj", "updated"]
T = {n: blob() for n in names}
x = f32(T["x"]).reshape(ROWS, HC * HIDDEN)
gain = f32(T["gain"])
down = f32(T["down"]).reshape(LOWRANK, HC * HIDDEN)
up = f32(T["up"]).reshape(HC * HIDDEN, LOWRANK)
inject = f32(T["inject"]).reshape(HC, HC * HIDDEN)
y = f32(T["y"]).reshape(ROWS, HIDDEN)

# the transcription used by the Metal test's CPU reference
W = HC * HIDDEN
xn = np.empty_like(x)
for s in range(HC):
    sl = slice(s * HIDDEN, (s + 1) * HIDDEN)
    ms = (x[:, sl].astype(np.float64) ** 2).mean(axis=1, keepdims=True)
    xn[:, sl] = x[:, sl] / np.sqrt(ms + 1e-6)
xn = xn * (1.0 + gain)  # upstream gain is an offset from one
low = down @ xn.T  # [LOWRANK, ROWS]
low = low / HC
low = low * (1 / (1 + np.exp(-low)))  # silu
w = 1 / (1 + np.exp(-(up @ low)))  # [W, ROWS]
mixed = (w.T * xn).reshape(ROWS, HC, HIDDEN).mean(axis=1)
inj = 2 / (1 + np.exp(-((inject @ xn.T) / HC)))  # [HC, ROWS]
updated = x + (inj.T[:, :, None] * y[:, None, :]).reshape(ROWS, W)

for name, got in [("mixed", mixed), ("inj", inj.T), ("updated", updated)]:
    want = f32(T[name]).reshape(got.shape)
    denom = np.maximum(np.maximum(abs(got), abs(want)), 1e-3)
    err = float((abs(got - want) / denom).max())
    print(f"  {name:8} worst relative error vs the real module: {err:.5f}")
