# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from hcu_sglang.serve_defaults import apply_qwen4_serve_defaults


class Qwen4ServeDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'renamed-checkpoint'
        self.root.mkdir()
        self.config = {
            'model_type': 'qwen4_exp',
            'text_config': {'dtype': 'bfloat16'},
            'compression_config': {'quant_method': 'slimquant_w4a8'},
        }
        self.args = ['--model-path', str(self.root)]
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        detection = ModuleType('hcu_sglang.utils.detection')
        detection.is_hcu = lambda: True
        self.detection = detection
        patch.dict(sys.modules, {detection.__name__: detection}).start()

    def apply(self, *args):
        (self.root / 'config.json').write_text(json.dumps(self.config))
        return apply_qwen4_serve_defaults(self.args + list(args))

    def test_config_identity_and_explicit_values(self):
        os.environ['SGLANG_QWEN4_ROUTED_INT8'] = '0'
        os.environ['W8A8_SUPPORT_METHODS'] = '1'
        applied = self.apply()
        self.assertEqual(os.environ['SGLANG_QWEN4_COMPACT_PREFILL'], '1')
        self.assertEqual(os.environ['SGLANG_HCU_QSA_DECODE_IMPL'], 'native-paged')
        self.assertEqual(os.environ['SGLANG_QWEN4_ROUTED_INT8'], '0')
        self.assertEqual(os.environ['W8A8_SUPPORT_METHODS'], '1')
        self.assertNotIn('SGLANG_QWEN4_ROUTED_INT8', applied)
        self.assertEqual(self.apply(), {})

    def test_decode_reader_override_preserves_baseline(self):
        os.environ['SGLANG_HCU_QSA_DECODE_IMPL'] = 'triton'
        applied = self.apply('--disaggregation-mode', 'decode')
        self.assertNotIn('SGLANG_HCU_QSA_DECODE_IMPL', applied)
        self.assertEqual(os.environ['SGLANG_HCU_QSA_DECODE_IMPL'], 'triton')

    def test_other_models_and_precision_remain_untouched(self):
        for model_type in ['qwen3_5', 'deepseek_v4', 'glm4_moe', 'qwen4_exp_future']:
            self.config['model_type'] = model_type
            self.assertEqual(self.apply(), {})
            self.assertEqual(dict(os.environ), {})
        self.config['model_type'] = 'qwen4_exp'
        for dtype in ['float16', 'float32']:
            self.assertEqual(self.apply('--dtype=' + dtype), {})
        self.assertEqual(self.apply('--quantization', 'fp8'), {})
        self.config['compression_config'] = {'quant_method': 'fp8'}
        self.assertEqual(self.apply(), {})
        self.config.pop('compression_config')
        self.assertEqual(self.apply(), {})

    def test_non_hcu_remains_untouched(self):
        self.detection.is_hcu = lambda: False
        self.assertEqual(self.apply(), {})
        self.assertEqual(dict(os.environ), {})

    def test_pairing_requires_compatible_prefill_delayer(self):
        self.apply('--disaggregation-mode=prefill', '--enable-prefill-delayer')
        self.assertEqual(os.environ['SGLANG_HCU_PD_PREFILL_DRAIN_MODE'], 'wait')
        self.assertNotIn('SGLANG_HCU_PREFILL_MIXED_TIMEOUT', os.environ)
        self.apply('--disaggregation-mode=prefill', '--enable-prefill-delayer',
                   '--prefill-delayer-queue-min-ratio=0.0')
        self.assertEqual(os.environ['SGLANG_HCU_PREFILL_MIXED_TIMEOUT'], '1')

    def test_decode_does_not_set_prefill_policy(self):
        self.apply('--disaggregation-mode', 'decode')
        self.assertNotIn('SGLANG_HCU_PD_PREFILL_DRAIN_MODE', os.environ)
        self.assertNotIn('SGLANG_HCU_PREFILL_MIXED_TIMEOUT', os.environ)

    def test_cli_forms_and_nested_text_checkpoint(self):
        self.args = ['serve', str(self.root)]
        self.assertTrue(self.apply())
        os.environ.clear()
        self.config = {'model_type': 'qwen4_exp_text', 'torch_dtype': 'bfloat16',
                       'compression_config': {'quant_method': 'slimquant_w4a8'}}
        self.args = ['--model-path=' + str(self.root)]
        self.assertTrue(self.apply())

    def test_missing_or_invalid_checkpoint_is_noop(self):
        self.assertEqual(apply_qwen4_serve_defaults([]), {})
        self.assertEqual(apply_qwen4_serve_defaults(self.args), {})
        (self.root / 'config.json').write_text('invalid')
        self.assertEqual(apply_qwen4_serve_defaults(self.args), {})


if __name__ == '__main__':
    unittest.main()
