import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models/qwen4exp/bench"))

import cache_plan as plan  # noqa: E402


class ReplayTest(unittest.TestCase):
    def test_a_cold_cache_reads_each_expert_once_then_hits(self):
        sequence = [
            ("D", Counter({1: 2, 2: 1})),
            ("D", Counter({3: 1})),
            ("D", Counter({1: 1, 3: 1})),
        ]
        # The first step warms 1 and 2; the second reads 3; the third hits.
        self.assertEqual(plan.replay(sequence, 4, "engine"), [2, 1, 0])

    def test_each_new_expert_is_one_read(self):
        # One slot, a different expert every step: every step misses once.
        sequence = [("D", Counter({e: 1})) for e in (1, 2, 3)]
        self.assertEqual(plan.replay(sequence, 1, "engine"), [1, 1, 1])

    def test_a_prompt_chunk_is_not_counted_as_a_read(self):
        sequence = [
            ("P", Counter({1: 5, 2: 4})),
            ("D", Counter({1: 1})),
            ("D", Counter({3: 1})),
        ]
        # The prompt loads 1 and 2; the first check step hits, the second
        # must read 3 from the SSD.
        self.assertEqual(plan.replay(sequence, 2, "engine"), [0, 1])

    def test_summary_reports_the_tail(self):
        mean, p95, worst = plan.summary([0, 0, 0, 0, 10])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(p95, 10.0)
        self.assertEqual(worst, 10.0)

    def test_greedy_split_spends_the_budget_where_it_helps(self):
        # Layer 0's reads fall faster with slots than layer 1's.
        curve = {
            0: {4: [10], 6: [5], 8: [1]},
            1: {4: [6], 6: [4], 8: [2]},
        }
        split = plan.greedy_split(curve, np.mean, [4, 6, 8], budget=16, step=2)
        # Layer 0 takes the first two chunks (gains 5 then 4), layer 1 the
        # last two; the budget is spent exactly.
        self.assertEqual(split, {0: 8, 1: 8})
        self.assertEqual(sum(split.values()), 16)

    def test_summed_step_reads_are_the_whole_gate(self):
        # All 48 layers run every step, so the plan is judged on the total:
        # two steps, each layer contributing its own reads.
        totals = plan.summed([[1, 0], [2, 3]])
        self.assertEqual(totals, [3, 3])

    def test_totals_split_moves_chunks_where_the_step_tail_falls(self):
        # Layer 0's occasional big step is what makes the tail; giving it the
        # extra slots cuts the summed series' 95th percentile, while the
        # mean-only fit would take the same chunk for a larger mean gain.
        curve = {
            0: {4: [0, 10, 0, 0], 6: [0, 2, 0, 0]},
            1: {4: [3, 0, 3, 3], 6: [2, 0, 2, 2]},
        }
        split = plan.totals_split(curve, grid=[4, 6], budget=10, step=2)
        # With one chunk to give, the tail (p95 = 8-ish from layer 0's 10)
        # falls more than layer 1's mean would reduce it.
        self.assertEqual(split[0], 6)
        self.assertEqual(sum(split.values()), 10)

    def test_uneven_split_never_costs_more_reads_than_uniform(self):
        # A layer whose reads fall fastest gets the extra slots; the mean
        # over the same budget cannot be worse than giving them out evenly.
        curve = {
            layer: {
                slots: [max(0, 10 - slots) for _ in range(8)] for slots in (4, 6, 8)
            }
            for layer in range(2)
        }
        split = plan.greedy_split(curve, np.mean, [4, 6, 8], budget=16, step=2)
        self.assertEqual(sum(split.values()), 16)
        self.assertLessEqual(
            sum(np.mean(curve[layer][split[layer]]) for layer in split),
            sum(np.mean(curve[layer][6]) for layer in split),
        )


if __name__ == "__main__":
    unittest.main()
