import unittest
from pathlib import Path
from unittest.mock import patch

from hcu_sglang import jit_overlay_loader
from hcu_sglang.models.deepseek.current_v4.hcu_backend import topk
from sglang.kernels.jit.utils.compile import loader


class BuildCaptured(Exception):
    pass


class DeepSeekV4TopKJitContract(unittest.TestCase):
    def test_plugin_topk_build_uses_owned_sources_with_global_overlay_disabled(self):
        overlay = jit_overlay_loader._OVERLAY_ROOT
        source = overlay / 'csrc/deepseek_v4/topk_v2.cuh'
        header_flag = f'-I{overlay / "include"}'

        def capture(spec, build_file):
            self.assertEqual(tuple(map(Path, spec.cuda_files)), (source,))
            exports = dict(spec.cuda_wrappers)
            for name in ('transform_kpool', 'transform_kpool_ragged'):
                self.assertTrue(exports[f'topk_{name}'].endswith(f'::{name}'))
                self.assertIn(f'static void {name}(', source.read_text())
            flags = next(line for line in build_file.splitlines() if line.startswith('cudaflags ='))
            for include in spec.include_paths:
                if include != str(overlay / 'include'):
                    self.assertLess(flags.index(header_flag), flags.index(f'-I{include}'))
            raise BuildCaptured

        with patch.dict('os.environ', {'SGLANG_HCU_JIT_OVERLAY': '0'}), patch.object(
            loader.cache, 'compute_build_key', side_effect=capture
        ):
            self.assertFalse(jit_overlay_loader._overlay_enabled())
            with self.assertRaises(BuildCaptured):
                topk._jit_topk_v2_module()


if __name__ == '__main__':
    unittest.main()
