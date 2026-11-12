# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mqa_request_split",
    ROOT / "hcu_sglang/layers/attention/current_dsa/mqa_request_split.py",
)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


class MQARequestSplitTests(unittest.TestCase):
    def test_empty_and_zero_query_requests(self):
        self.assertEqual(planner.plan_mqa_request_slices([], []), ())
        self.assertEqual(planner.plan_mqa_request_slices([0, 0], [4, 5]), ())

    def test_single_request_preserves_full_slice_fast_path(self):
        self.assertEqual(planner.plan_mqa_request_slices([3], [7], 0), ())

    def test_partial_grouping_beats_both_extreme_partitions(self):
        self.assertEqual(
            planner.plan_mqa_request_slices([1, 1, 1], [1, 100, 1], 10),
            ((0, 2, 0, 101), (2, 3, 101, 102)),
        )

    def test_threshold_tie_keeps_more_groups(self):
        self.assertEqual(
            planner.plan_mqa_request_slices([2, 3], [5, 7], 15),
            ((0, 2, 0, 5), (2, 5, 5, 12)),
        )

    def test_full_slice_falls_back_when_extra_launch_costs_more(self):
        self.assertEqual(planner.plan_mqa_request_slices([2, 3], [5, 7], 16), ())

    def test_zero_query_requests_preserve_k_offsets_and_crop_boundaries(self):
        q_lens = [0, 2, 0, 3, 0]
        k_lens = [7, 4, 11, 5, 13]
        self.assertEqual(
            planner.plan_mqa_request_slices(q_lens, k_lens, 0),
            ((0, 2, 7, 11), (2, 5, 22, 27)),
        )
        self.assertEqual(
            planner.plan_mqa_request_slices(q_lens, k_lens, 1000),
            ((0, 5, 7, 27),),
        )

    def test_one_byte_budget_keeps_one_row_and_each_group_k_slice(self):
        self.assertEqual(
            list(planner.iter_mqa_chunks(((0, 3, 4, 9), (3, 5, 12, 14)), 1)),
            [(0, 1, 4, 9), (1, 2, 4, 9), (2, 3, 4, 9),
             (3, 4, 12, 14), (4, 5, 12, 14)],
        )

    def test_zero_budget_keeps_complete_groups(self):
        slices = ((0, 3, 4, 9), (3, 5, 12, 14))
        self.assertEqual(tuple(planner.iter_mqa_chunks(slices, 0)), slices)

    def test_invalid_planner_inputs_raise(self):
        for q_lens, k_lens, threshold, message in (
            ([1], [2], -1, "threshold"),
            ([1], [], 0, "counts differ"),
            ([-1], [2], 0, "lengths"),
            ([1], [-2], 0, "lengths"),
        ):
            with self.subTest(q_lens=q_lens, k_lens=k_lens, threshold=threshold):
                with self.assertRaisesRegex(ValueError, message):
                    planner.plan_mqa_request_slices(q_lens, k_lens, threshold)


if __name__ == "__main__":
    unittest.main()
