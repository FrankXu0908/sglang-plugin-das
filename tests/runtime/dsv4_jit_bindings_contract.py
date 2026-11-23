import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from hcu_sglang import jit_overlay_loader
from hcu_sglang.models.deepseek.current_v4.hcu_backend import compress, elementwise
from sglang.kernels.jit.utils.compile import loader


class BuildCaptured(Exception):
    pass


class DeepSeekV4JitBindingsContract(unittest.TestCase):
    def test_plugin_sources_with_global_overlay_disabled(self):
        overlay = jit_overlay_loader._OVERLAY_ROOT
        cases = (
            (compress._jit_compress_plan_module, 'c_plan.cuh', {'plan_prefill_out', 'plan_decode_out'}),
            (
                lambda: elementwise._jit_main_q_indexer_rope_hadamard_module(torch.bfloat16),
                'main_norm_rope.cuh',
                {'forward'},
            ),
            (
                lambda: compress._jit_compress_norm_rope_module(
                    torch.bfloat16, 128, 64, 64, False, compress.KVLayout.V4, int8_store=True
                ),
                'fused_norm_rope_v2.cuh',
                {'forward'},
            ),
        )
        for call, filename, names in cases:
            with self.subTest(filename=filename):
                def capture(spec, build_file):
                    self.assertEqual(tuple(map(Path, spec.cuda_files)), (overlay / 'csrc/deepseek_v4' / filename,))
                    self.assertTrue(names <= dict(spec.cuda_wrappers).keys())
                    flags = next(line for line in build_file.splitlines() if line.startswith('cudaflags ='))
                    overlay_flag = f'-I{overlay / "include"}'
                    for include in spec.include_paths:
                        if include != str(overlay / 'include'):
                            self.assertLess(flags.index(overlay_flag), flags.index(f'-I{include}'))
                    raise BuildCaptured

                with patch.dict('os.environ', {'SGLANG_HCU_JIT_OVERLAY': '0'}), patch.object(
                    loader.cache, 'compute_build_key', side_effect=capture
                ):
                    self.assertFalse(jit_overlay_loader._overlay_enabled())
                    with self.assertRaises(BuildCaptured):
                        call()


if __name__ == '__main__':
    unittest.main()
