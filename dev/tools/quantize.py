#!/usr/bin/env python3
"""Affine 4-bit quantization in the layout the engine's packages carry.

The scheme is the one mlx-community's packages use, which is what the shipped
Splash packages were converted from. Two details of it are not derivable from
a packed file and are worth stating, because getting either wrong produces
weights that load, run, and are quietly worse:

  * A group is anchored at whichever of its extremes is larger in magnitude,
    not at its minimum. That value lands exactly on code zero and the error
    falls on the smaller side. Anchoring at the maximum makes the codes count
    downwards, which is why a scale can be negative.

  * The codes are quantized against the bf16 scale and bias that will be
    stored, not against the float32 ones they were derived from. The reader
    only ever sees the stored values, so quantizing against anything else
    bakes in an error it has no way to undo.

Checked against mlx-community/Qwen3.8-27B-4bit: rounding is optimal, never
exceeding half a step of the group it belongs to.
"""

from __future__ import annotations

import numpy as np


def to_bf16(values):
    """Round float32 to bf16 bit patterns, nearest with ties to even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def quantize_affine(values, group=64, bits=4):
    """float32 [out, in] -> (codes [out, in], scales, biases), the last two bf16."""
    out, inp = values.shape
    if inp % group:
        raise ValueError(f"input {inp} is not a whole number of {group}-wide groups")
    blocks = np.ascontiguousarray(values, dtype=np.float32).reshape(
        out, inp // group, group
    )
    low = blocks.min(axis=2)
    high = blocks.max(axis=2)
    levels = (1 << bits) - 1

    anchor_low = np.abs(low) > np.abs(high)
    bias = np.where(anchor_low, low, high)
    other = np.where(anchor_low, high, low)
    scale = (other - bias) / levels

    scale_bits = to_bf16(scale)
    bias_bits = to_bf16(bias)
    stored_scale = from_bf16(scale_bits)[..., None]
    stored_bias = from_bf16(bias_bits)[..., None]
    # A group whose values are all equal has no range and any scale serves;
    # the bias alone reproduces it.
    safe = np.where(stored_scale == 0, 1.0, stored_scale)
    codes = np.clip(np.rint((blocks - stored_bias) / safe), 0, levels)
    return codes.astype(np.uint8).reshape(out, inp), scale_bits, bias_bits


def dequantize_affine(codes, scales, biases, group=64):
    out, inp = codes.shape
    scale = from_bf16(scales)[..., None]
    bias = from_bf16(biases)[..., None]
    blocks = codes.reshape(out, inp // group, group).astype(np.float32)
    return (blocks * scale + bias).reshape(out, inp)


def pack_nibbles(codes):
    """[..., n] 4-bit codes -> bytes, low nibble first, as the readers expect."""
    flat = np.ascontiguousarray(codes, dtype=np.uint8).reshape(-1)
    if flat.size % 2:
        raise ValueError("a packed run must hold an even number of codes")
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def unpack_nibbles(packed, count):
    bytes_ = np.frombuffer(packed, dtype=np.uint8)
    out = np.empty(count, dtype=np.uint8)
    out[0::2] = bytes_ & 0xF
    out[1::2] = bytes_ >> 4
    return out
