from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_detection(hip, properties, available=True):
    # A query that answers implies a visible HIP device. is_hcu() consults
    # is_available() only after the query failed, to decide between a warning
    # and an error, so the failing probes state their case with available.
    torch = SimpleNamespace(version=SimpleNamespace(hip=hip),
                            cuda=SimpleNamespace(get_device_properties=properties,
                                                 is_available=Mock(return_value=available)))
    spec = importlib.util.spec_from_file_location(
        'isolated_hcu_detection', ROOT / 'hcu_sglang/utils/detection.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'torch': torch}):
        spec.loader.exec_module(module)
    return module


class SourceHardwareDecisionTests(unittest.TestCase):
    def test_hip_identity_uses_none_rather_than_truthiness(self):
        query = Mock(return_value=SimpleNamespace(gcnArchName='gfx936'))
        module = load_detection('', query)
        self.assertTrue(module.is_hcu())
        self.assertFalse(module.is_hcu_native_fp8_supported())

    def test_architecture_matrix_and_no_device_queries_on_non_hip(self):
        for hip, arch, expected, fp8 in [
            (None, 'gfx938', False, False),
            ('6.4', 'gfx936:sramecc+:xnack-', True, False),
            ('6.4', 'gfx938:sramecc+:xnack-', True, True),
            ('6.4', 'gfx928', True, False),
            ('6.4', 'gfx942', False, False),
        ]:
            with self.subTest(hip=hip, arch=arch):
                query = Mock(return_value=SimpleNamespace(gcnArchName=arch))
                module = load_detection(hip, query)
                self.assertEqual(module.is_hcu(), expected)
                self.assertEqual(module.is_hcu_native_fp8_supported(), fp8)
                if hip is None:
                    query.assert_not_called()

    def test_empty_architecture_is_an_error_on_a_visible_hip_device(self):
        # #113: an empty gcnArchName is a failed probe. With no device visible
        # it is the usual non-HCU answer; with one visible it must not pass
        # for an answer at all.
        for available in (True, False):
            with self.subTest(available=available):
                query = Mock(return_value=SimpleNamespace(gcnArchName=''))
                module = load_detection('6.4', query, available=available)
                if available:
                    with self.assertRaisesRegex(RuntimeError, 'visible HIP device 0'):
                        module.is_hcu()
                else:
                    with self.assertLogs('isolated_hcu_detection', level='WARNING'):
                        self.assertFalse(module.is_hcu())

    def test_initial_detection_failure_remains_a_non_hcu_result(self):
        module = load_detection('6.4', Mock(side_effect=RuntimeError('no device')),
                                available=False)
        with self.assertLogs('isolated_hcu_detection', level='WARNING'):
            self.assertFalse(module.is_hcu())
        self.assertFalse(module.is_hcu_native_fp8_supported())

    def test_detection_failure_on_a_visible_hip_device_is_surfaced(self):
        # #113: a device is visible but does not answer. That is not "not HCU"
        # and must reach the caller instead of selecting the non-HCU paths.
        module = load_detection('6.4', Mock(side_effect=RuntimeError('no device')))
        with self.assertRaisesRegex(RuntimeError, 'Cannot determine the architecture of visible HIP device 0') as raised:
            module.is_hcu()
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(str(raised.exception.__cause__), 'no device')

    def test_native_fp8_query_failure_is_not_silently_reclassified(self):
        query = Mock(side_effect=[SimpleNamespace(gcnArchName='gfx938'),
                                  RuntimeError('device disappeared')])
        module = load_detection('6.4', query)
        with self.assertRaisesRegex(RuntimeError, 'device disappeared'):
            module.is_hcu_native_fp8_supported()

    def test_native_fp8_requires_the_provider_arch_attribute(self):
        query = Mock(side_effect=[SimpleNamespace(gcnArchName='gfx938'),
                                  SimpleNamespace()])
        module = load_detection('6.4', query)
        with self.assertRaises(AttributeError):
            module.is_hcu_native_fp8_supported()


def load_kernel_compat(torch, hcu=True):
    spec = importlib.util.spec_from_file_location(
        'isolated_kernel_compat', ROOT / 'hcu_sglang/kernel_compat.py')
    module = importlib.util.module_from_spec(spec)
    detector = SimpleNamespace(is_hcu=Mock(return_value=hcu))
    with patch.dict('sys.modules', {
        'torch': torch, 'hcu_sglang.utils.detection': detector,
    }):
        spec.loader.exec_module(module)
    return module


class SourceKernelDispatchTests(unittest.TestCase):
    def test_jit_fp8_macro_follows_the_current_device_architecture(self):
        for arch, hcu, expected in [
            ('gfx928', True, 'E4M3'), ('gfx936', True, 'FNUZ'),
            ('gfx938', True, 'FNUZ'), ('gfx90a', False, 'FNUZ'),
            ('gfx940', False, 'FNUZ'), ('gfx941', False, 'FNUZ'),
            ('gfx942', False, 'FNUZ'), ('gfx1100', False, 'E4M3'),
        ]:
            with self.subTest(arch=arch):
                cuda = SimpleNamespace(current_device=Mock(return_value=3),
                    get_device_properties=Mock(return_value=SimpleNamespace(gcnArchName=arch)))
                module = load_kernel_compat(SimpleNamespace(cuda=cuda), hcu)
                flags = ['-DUSE_ROCM', '-O3', '-DHIP_FP8_TYPE_E4M3=1']
                result = module.after_get_default_target_flags(flags)
                self.assertEqual(result, ['-DUSE_ROCM', '-O3', '-DHIP_FP8_TYPE_'+expected+'=1'])
                cuda.get_device_properties.assert_called_with(3)

    def test_jit_detection_failure_keeps_source_e4m3_fallback(self):
        cuda = SimpleNamespace(current_device=Mock(side_effect=RuntimeError('device lost')))
        module = load_kernel_compat(SimpleNamespace(cuda=cuda))
        self.assertEqual(module.after_get_default_target_flags(
            ['-DUSE_ROCM', '-DHIP_FP8_TYPE_E4M3=1']),
            ['-DUSE_ROCM', '-DHIP_FP8_TYPE_E4M3=1'])

    def test_non_rocm_compile_flags_do_not_query_a_device(self):
        cuda = SimpleNamespace(current_device=Mock(side_effect=AssertionError('unexpected query')))
        module = load_kernel_compat(SimpleNamespace(cuda=cuda))
        flags = ['-DSGL_CUDA_ARCH=900', '-O3']
        self.assertEqual(module.after_get_default_target_flags(flags), flags)
        cuda.current_device.assert_not_called()

    def test_speculative_platform_provider_precedes_the_hcu_kernel(self):
        sentinel = object()
        provider = Mock(return_value=sentinel)
        native = SimpleNamespace(_get_oot_speculative_cache_locs_fn=lambda: provider)
        module = load_kernel_compat(SimpleNamespace())
        request, table, start, end = object(), object(), object(), object()
        with patch.dict('sys.modules', {
            'sglang.kernels.ops.speculative.cache_locs': native,
            'sglang.srt.utils': SimpleNamespace(get_bool_env_var=Mock(
                side_effect=AssertionError('platform provider must win'))),
        }):
            result = module.around_assign_extend_cache_locs_func(
                Mock(side_effect=AssertionError('unexpected original')),
                request, table, start, end, 2, 4, 'cuda:0')
        self.assertIs(result, sentinel)
        provider.assert_called_once_with(
            req_pool_indices=request, req_to_token=table,
            start_offset=start, end_offset=end, batch_size=2,
            draft_token_num=4, device='cuda:0')


if __name__ == '__main__':
    unittest.main()
