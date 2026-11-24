import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from hcu_sglang.multimodal_gen import minimax_h3_vae as hooks


class MiniMaxH3VaeOwnershipTests(unittest.TestCase):
    def test_cpu_conversion_preserves_effective_baseline(self):
        pixels = np.empty((2, 8, 8, 3), dtype=np.uint8)
        original = Mock()
        with patch.object(hooks, "is_hcu", return_value=True):
            result = hooks.around_convert_numpy_to_tensor(original, pixels, "cpu")
        original.assert_called_once_with(pixels, "cpu")
        self.assertIs(result, original.return_value)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a device")
    def test_decoded_pixels_match_on_device(self):
        frames = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
        original = Mock(side_effect=AssertionError("unexpected fallback"))
        with patch.object(hooks, "is_hcu", return_value=True):
            for x in (frames, list(frames)):
                packed = hooks.around_convert_numpy_to_tensor(original, x, "cuda")
                expected = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float().div_(255)
                self.assertTrue(torch.equal(packed, expected))
                self.assertTrue(packed.is_contiguous())
