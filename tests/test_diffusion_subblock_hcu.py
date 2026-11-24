from importlib.util import find_spec
import unittest
from unittest.mock import Mock, patch

import torch

from hcu_sglang.multimodal_gen import subblock_sparse as bsa


class SubBlockHcuTests(unittest.TestCase):
    def test_only_active_prefix_is_sorted(self):
        index = torch.tensor([[[[4, 2, 0, 1, 3], [4, 1, 3, 0, 2]]]], dtype=torch.int32)
        original = index.clone()
        counts = torch.tensor([[[2, 5]]], dtype=torch.int32)
        ordered = bsa._sorted_active_indices(index, counts, 5)
        self.assertEqual(ordered.tolist(), [[[[2, 4, 0, 1, 3], [0, 1, 2, 3, 4]]]])
        self.assertTrue(torch.equal(index, original))

    def test_other_platform_uses_original_runner(self):
        original = Mock()
        device = torch.device("cuda")
        with patch.object(bsa, "is_hcu", return_value=False):
            runner = bsa.around_get_runner(original, device, "sage_fp8")
        self.assertIs(runner, original.return_value)
        original.assert_called_once_with(device, "sage_fp8")

    @unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires HIP")
    def test_partial_key_block_and_heterogeneous_counts(self):
        if find_spec("dcu_bsa") is None:
            self.skipTest("requires the optional dcu_bsa wheel")
        from dcu_bsa.reference import bsa_attn_blk64_fwd

        torch.manual_seed(1101)
        q = torch.randn(1, 129, 2, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 257, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        row = torch.tensor([4, 1, 3, 2, 0], device=q.device, dtype=torch.int32)
        index = row.expand(1, 2, 3, 5).contiguous()
        counts = torch.tensor([2, 5, 1], device=q.device, dtype=torch.int32).expand(1, 2, 3)
        sizes = torch.tensor([64, 64, 64, 64, 1], device=q.device, dtype=torch.int32)
        expected, _ = bsa_attn_blk64_fwd(
            q, k, v, index, 5, block_sizes=sizes, q2k_block_nums=counts, softmax_scale=0.125
        )
        actual = bsa.hcu_sparse_attention(q, k, v, index, 5, 0.125, counts)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
