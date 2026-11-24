import unittest
from unittest.mock import patch

import torch

from hcu_sglang.multimodal_gen import minimax_h3


class MiniMaxH3StagingTests(unittest.TestCase):
    @unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires HIP")
    def test_release_after_return_and_exception(self):
        from sglang.multimodal_gen.runtime.layers import usp

        device = torch.device("cuda", torch.cuda.current_device())
        failure = RuntimeError("request failed")
        for fail in (False, True):
            with self.subTest(fail=fail), torch.inference_mode():
                key = ("h3-test", torch.bfloat16, device.index)
                keep = []

                def loop(*args):
                    view = usp._a2a_staging_buffer(key[0], (128,), key[1], device)
                    view.fill_(4)
                    keep.append(view[:4])
                    self.assertIn(key, usp._A2A_STAGING_BUFFERS)
                    if fail:
                        raise failure
                    return view[:1]

                with patch.object(minimax_h3, "is_hcu", return_value=True):
                    if fail:
                        with self.assertRaises(RuntimeError) as ctx:
                            minimax_h3.around_denoising_full_loop(loop, None, None, None)
                        self.assertIs(ctx.exception, failure)
                    else:
                        result = minimax_h3.around_denoising_full_loop(loop, None, None, None)
                        self.assertEqual(result.item(), 4)
                self.assertNotIn(key, usp._A2A_STAGING_BUFFERS)
                self.assertTrue(keep[0].eq(4).all())
