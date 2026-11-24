# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as S
import unittest

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from hcu_sglang.forward_batch import ForwardBatchMethods
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput


class CachedPrefillHiddenStatesTests(unittest.TestCase):
    def run_extend(self, *, current_tokens, total_seq_len, output_rows):
        batch = S(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            _original_forward_mode=None,
            _original_batch_size=1,
            _original_num_tokens=current_tokens,
            seq_lens_sum=total_seq_len,
            spec_info=None,
            positions=torch.arange(output_rows),
            seq_lens=torch.tensor([total_seq_len]),
            req_pool_indices=torch.tensor([0]),
            req_pool_indices_cpu=None,
            seq_lens_cpu=None,
        )
        output = S(
            next_token_logits=torch.arange(output_rows),
            hidden_states=torch.arange(output_rows * 4).reshape(output_rows, 4),
        )
        ForwardBatchMethods.post_forward_mlp_sync_batch(batch, output)
        self.assertEqual(output.hidden_states.shape, (current_tokens, 4))
        self.assertEqual(output.next_token_logits.shape, (current_tokens,))
        self.assertTrue(
            torch.equal(
                output.hidden_states,
                torch.arange(current_tokens * 4).reshape(current_tokens, 4),
            )
        )
        draft_input = EagleDraftExtendInput(hidden_states=output.hidden_states)
        self.assertEqual(draft_input.hidden_states.shape[0], current_tokens)

    def test_hot_prefix_uses_current_extend_length(self):
        self.run_extend(current_tokens=64, total_seq_len=1024, output_rows=128)

    def test_cold_prefix_keeps_full_extend(self):
        self.run_extend(current_tokens=1024, total_seq_len=1024, output_rows=1024)

    def test_short_extend_trims_rank_padding(self):
        self.run_extend(current_tokens=10, total_seq_len=1010, output_rows=20)


if __name__ == "__main__":
    unittest.main()
