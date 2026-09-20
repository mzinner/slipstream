"""The quantizer must round optimally and reproduce the published scheme."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.quantize import (  # noqa: E402
    dequantize_affine,
    from_bf16,
    pack_nibbles,
    quantize_affine,
    to_bf16,
    unpack_nibbles,
)


class QuantizeTests(unittest.TestCase):
    def test_rounding_never_exceeds_half_a_step(self):
        rng = np.random.default_rng(20260919)
        values = (rng.normal(size=(64, 512)) * 0.05).astype(np.float32)
        codes, scales, biases = quantize_affine(values)
        restored = dequantize_affine(codes, scales, biases)
        step = np.abs(from_bf16(scales)).repeat(64, axis=1)
        error = np.abs(restored - values) / np.maximum(step, 1e-30)
        self.assertLessEqual(float(error.max()), 0.5 + 1e-6)

    def test_group_is_anchored_at_the_larger_extreme(self):
        # A minimum-anchored implementation gets this case wrong while still
        # producing weights that load and run.
        values = np.array([[-0.1, 0.9] + [0.0] * 62], dtype=np.float32)
        _, scales, biases = quantize_affine(values)
        self.assertAlmostEqual(float(from_bf16(biases)[0, 0]), 0.9, places=2)
        self.assertLess(float(from_bf16(scales)[0, 0]), 0.0)

        values = np.array([[-0.9, 0.1] + [0.0] * 62], dtype=np.float32)
        _, scales, biases = quantize_affine(values)
        self.assertAlmostEqual(float(from_bf16(biases)[0, 0]), -0.9, places=2)
        self.assertGreater(float(from_bf16(scales)[0, 0]), 0.0)

    def test_a_flat_group_survives_and_extremes_stay_exact(self):
        values = np.concatenate(
            [
                np.full((1, 64), 0.25, dtype=np.float32),
                np.linspace(-0.4, 0.7, 64, dtype=np.float32).reshape(1, 64),
            ]
        )
        codes, scales, biases = quantize_affine(values)
        restored = dequantize_affine(codes, scales, biases)
        self.assertTrue(np.allclose(restored[0], values[0], atol=1e-3))
        # The anchored extreme is exact up to the bias as stored, which is
        # bf16 - not up to the float32 it came from.
        anchored = from_bf16(to_bf16(np.array([values[1].max()])))[0]
        self.assertEqual(float(restored[1].max()), float(anchored))

    def test_finer_groups_for_rows_that_need_them(self):
        # The per-layer embedding is 160 wide, not a whole number of 64s.
        values = np.random.default_rng(7).normal(size=(8, 160)).astype(np.float32)
        with self.assertRaises(ValueError):
            quantize_affine(values, group=64)
        codes, scales, biases = quantize_affine(values, group=32)
        self.assertEqual(scales.shape, (8, 5))
        restored = dequantize_affine(codes, scales, biases, group=32)
        step = np.abs(from_bf16(scales)).repeat(32, axis=1)
        error = np.abs(restored - values) / np.maximum(step, 1e-30)
        self.assertLessEqual(float(error.max()), 0.5 + 1e-6)

    def test_nibbles_round_trip_low_first(self):
        codes = np.array([1, 2, 3, 4, 15, 0], dtype=np.uint8)
        packed = pack_nibbles(codes)
        self.assertEqual(list(packed[:3]), [0x21, 0x43, 0x0F])
        self.assertTrue(
            np.array_equal(unpack_nibbles(packed.tobytes(), codes.size), codes)
        )

    def test_bf16_round_trip(self):
        values = np.array([0.7, -0.7, 1.9, 0.0, 3.0e-3], dtype=np.float32)
        self.assertTrue(np.allclose(from_bf16(to_bf16(values)), values, rtol=5e-3))


if __name__ == "__main__":
    unittest.main()
