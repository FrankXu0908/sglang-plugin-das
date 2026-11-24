from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.attention.current_dsa import backend as B, indexer as I
from lightop import attention as A
from hcu_sglang.layers.attention.current_dsa import mqa_schedule as M


class MQAScheduleTests(unittest.TestCase):
    def test_backend_and_indexer_select_lightop_schedule_provider(self):
        with patch("hcu_sglang.utils.detection.is_hcu", return_value=True):
            self.assertIs(B.resolve_paged_mqa_logits_metadata_fn(), A.get_paged_mqa_logits_metadata)
            self.assertIs(I.resolve_paged_mqa_logits_metadata_fn(), A.get_paged_mqa_logits_metadata)

    def test_refresh_preserves_captured_storage_and_passes_new_lengths(self):
        lens = torch.tensor([[7], [65]], dtype=torch.int32)
        new = torch.arange(10, dtype=torch.int32).reshape(5, 2)
        old = torch.zeros_like(new)
        metadata = SimpleNamespace(paged_mqa_schedule_metadata=old)
        owner = object.__new__(B.DeepseekSparseAttnBackend)
        with patch("hcu_sglang.utils.detection.is_hcu", return_value=True), patch.object(B, "resolve_num_sms", return_value=4), patch.object(A, "get_paged_mqa_logits_metadata", return_value=new) as planner:
            B.DeepseekSparseAttnBackend._refresh_paged_mqa_schedule_metadata(owner, metadata, lens)
        self.assertIs(metadata.paged_mqa_schedule_metadata, old)
        self.assertTrue(torch.equal(old, new))
        self.assertIs(planner.call_args.args[0], lens)
        self.assertEqual(planner.call_args.args[1:], (64, 4))

    def test_non_hcu_preserves_native_resolver(self):
        marker = object()
        with patch.object(M.detection, 'is_hcu', return_value=False), patch.object(M, 'native_resolver', return_value=marker):
            self.assertIs(M.resolve_paged_mqa_logits_metadata_fn(), marker)

    def test_refresh_initializes_empty_schedule(self):
        lens = torch.tensor([[13]], dtype=torch.int32)
        new = torch.ones(5, 2, dtype=torch.int32)
        metadata = SimpleNamespace(paged_mqa_schedule_metadata=None)
        owner = object.__new__(B.DeepseekSparseAttnBackend)
        with patch("hcu_sglang.utils.detection.is_hcu", return_value=True), patch.object(B, "resolve_num_sms", return_value=4), patch.object(A, "get_paged_mqa_logits_metadata", return_value=new):
            B.DeepseekSparseAttnBackend._refresh_paged_mqa_schedule_metadata(owner, metadata, lens)
        self.assertIs(metadata.paged_mqa_schedule_metadata, new)


if __name__ == "__main__":
    unittest.main()
