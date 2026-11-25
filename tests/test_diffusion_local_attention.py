from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from hcu_sglang.multimodal_gen.runtime.layers.attention import local_attention as A


@unittest.skipIf(torch is None, "requires Torch")
class DiffusionLocalAttentionTests(unittest.TestCase):
    def test_mask_semantics_preserve_bias_nonbinary_and_empty_rows(self):
        cases = [
            (torch.ones(2, 4, dtype=torch.int64), "full"),
            (torch.ones(2, 4, dtype=torch.bool), "full"),
            (torch.zeros(2, 4), "full"),
            (torch.ones(2, 4), "bias"),
            (torch.tensor([[1, 0, 1]]), "binary"),
            (torch.tensor([[1, 2, 0]]), "bias"),
            (torch.tensor([[0, 0, 0]]), "bias"),
            (torch.tensor([[1, 0], [0, 0]]), "bias"),
            (torch.tensor([[0.0, -0.5]]), "bias"),
            (torch.tensor([[0.0, -float("inf")]]), "bias"),
        ]
        for mask, expected in cases:
            with self.subTest(mask=mask):
                self.assertEqual(A._mask_kind(mask), expected)

    def test_cpu_preserves_effective_original_and_arguments(self):
        q = torch.ones(1, 3, 2, 8)
        mask = torch.ones(1, 3, dtype=torch.int64)
        original = Mock(return_value=object())
        attention = object()
        with patch.object(A, "_masked_fa", side_effect=AssertionError("unexpected import")):
            self.assertIs(
                A.around_local_attention(original, attention, q, q, q, mask),
                original.return_value,
            )
        original.assert_called_once_with(attention, q, q, q, attn_mask=mask)

    def test_connector_normalizes_only_fully_valid_masks(self):
        batch = SimpleNamespace(
            prompt_attention_mask=torch.ones(2, 4, dtype=torch.int64),
            negative_attention_mask=torch.tensor([[1, 1, 0, 0]]),
        )
        negative = batch.negative_attention_mask
        with patch.object(A, "is_hcu", return_value=True):
            self.assertIs(A.after_ltx_text_connector(batch, None, batch, None), batch)
        self.assertIsNone(batch.prompt_attention_mask)
        self.assertIs(batch.negative_attention_mask, negative)

    def test_non_hcu_connector_is_unchanged(self):
        batch = SimpleNamespace(prompt_attention_mask=torch.ones(1, 4))
        mask = batch.prompt_attention_mask
        with patch.object(A, "is_hcu", return_value=False):
            A.after_ltx_text_connector(batch, None, batch, None)
        self.assertIs(batch.prompt_attention_mask, mask)

    def test_cfg_keeps_both_masks_when_either_branch_has_padding(self):
        for negative_full in (False, True):
            batch = SimpleNamespace(
                do_classifier_free_guidance=True,
                prompt_attention_mask=torch.ones(1, 4, dtype=torch.int64),
                negative_attention_mask=torch.tensor([[1, 1, 1, int(negative_full)]]),
            )
            positive, negative = batch.prompt_attention_mask, batch.negative_attention_mask
            with patch.object(A, "is_hcu", return_value=True):
                A.after_ltx_text_connector(batch, None, batch, None)
            if negative_full:
                self.assertIsNone(batch.prompt_attention_mask)
                self.assertIsNone(batch.negative_attention_mask)
            else:
                self.assertIs(batch.prompt_attention_mask, positive)
                self.assertIs(batch.negative_attention_mask, negative)


if __name__ == "__main__":
    unittest.main()
