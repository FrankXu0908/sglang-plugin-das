# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class HcuRuntimeTests(unittest.TestCase):
    def test_cold_import_keeps_startup_lightweight_and_uses_current_namespace(self):
        code = "import hcu_sglang,sys,json; print(json.dumps({'file':hcu_sglang.__file__,'loaded':[n for n in sys.modules if n.split('.')[0] in ['torch','aiter','sglang_hcu_plugin']]}))"
        report = json.loads(subprocess.check_output(
            [sys.executable, '-S', '-c', code], cwd=ROOT, text=True))
        self.assertTrue(Path(report['file']).is_relative_to(ROOT / 'hcu_sglang'))
        self.assertEqual(report['loaded'], [])

    def test_launcher_preserves_spawn_bootstrap_entry_and_cli_arguments(self):
        from hcu_sglang.__main__ import main
        args = ['hcu-sglang-serve', '--model-path', 'test-model', '--tp-size', '8']
        calls = []
        with patch('runpy.run_module', side_effect=lambda *a, **kw: calls.append((a, kw, sys.argv[1:]))), \
             patch.object(sys, 'argv', args):
            main()
        self.assertEqual(calls, [(('hcu_sglang.launch_server',),
                                  {'run_name': '__main__', 'alter_sys': True}, args[1:])])

    def test_spawn_worker_imports_optional_aiter_api_after_compat_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for package in ['aiter', 'aiter/ops', 'aiter/ops/triton', 'sglang']:
                path = root / package
                path.mkdir(parents=True, exist_ok=True)
                (path / '__init__.py').write_text('')
            (root / 'worker_fixture.py').write_text(
                "from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import fused_gemm_afp4wfp4_split_cat\n"
                "def worker(queue):\n"
                "    queue.put(fused_gemm_afp4wfp4_split_cat.__name__)\n")
            (root / 'sglang/launch_server.py').write_text(
                "import multiprocessing as mp\n"
                "from worker_fixture import worker\n"
                "if __name__ == '__main__':\n"
                "    ctx = mp.get_context('spawn')\n"
                "    queue = ctx.Queue()\n"
                "    process = ctx.Process(target=worker, args=(queue,))\n"
                "    process.start()\n"
                "    process.join(15)\n"
                "    assert process.exitcode == 0, process.exitcode\n"
                "    assert queue.get(timeout=5) == '_unsupported_fused_gemm'\n"
                "    print('spawn bootstrap passed')\n")
            env = {**os.environ, 'PYTHONPATH': str(root) + os.pathsep + str(ROOT)}
            # -S as in the sibling checks: the HCU image's site hooks cost tens of
            # seconds per interpreter, and the spawned worker starts with the same flag.
            result = subprocess.run([sys.executable, '-S', '-m', 'hcu_sglang'], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('spawn bootstrap passed', result.stdout)

    def test_cli_choices_include_all_plugin_quantization_methods_without_loading_configs(self):
        code = """
import argparse
import sys
import types
module = types.ModuleType('sglang.srt.arg_groups.arg_utils')
module.is_record = lambda cls: False
module.field_names = lambda cls: frozenset()
sys.modules[module.__name__] = module
from hcu_sglang.server_args import ensure_choices
from hcu_sglang.layers.quantization.registry import _HCU_QUANTIZATION_NAMES
class Args:
    ATTENTION_BACKEND_CHOICES = ['triton']
    DRAFT_ATTENTION_BACKEND_CHOICES = ['triton']
    QUANTIZATION_CHOICES = ['fp8']
    MOE_A2A_BACKEND_CHOICES = ['none']
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = []
    ServerArgs = type('ServerArgs', (), {})
    @classmethod
    def add_chunked_prefix_cache_attention_backend(cls, name):
        cls.CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(name)
assert ensure_choices(Args)
assert ensure_choices(Args)
parser = argparse.ArgumentParser()
parser.add_argument('--quantization', choices=Args.QUANTIZATION_CHOICES)
for name in _HCU_QUANTIZATION_NAMES:
    assert parser.parse_args(['--quantization', name]).quantization == name
    assert Args.QUANTIZATION_CHOICES.count(name) == 1
assert parser.parse_args(['--quantization', 'fp8']).quantization == 'fp8'
assert Args.MOE_A2A_BACKEND_CHOICES.count('mori-das') == 1
assert 'torch' not in sys.modules
assert not any(name.endswith('slimquant_w4a8_marlin') for name in sys.modules)
"""
        result = subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_choices_support_namespaced_moe_field_metadata(self):
        code = """
import sys
import types
import typing
from typing import Annotated, Literal
arg_utils = types.ModuleType('sglang.srt.arg_groups.arg_utils')
class Arg:
    def __init__(self, choices):
        self.choices = choices
arg_utils.Arg = Arg
arg_utils.is_record = lambda cls: False
arg_utils.field_names = lambda cls: set()
arg_utils.get_type_hints = typing.get_type_hints
sys.modules[arg_utils.__name__] = arg_utils
arg_groups = types.ModuleType('sglang.srt.arg_groups')
arg_groups.__path__ = []
arg_groups.arg_utils = arg_utils
sys.modules[arg_groups.__name__] = arg_groups
fields = types.ModuleType('sglang.srt.arg_groups.fields')
fields.__path__ = []
sys.modules[fields.__name__] = fields
exec_fields = types.ModuleType('sglang.srt.arg_groups.fields.exec_')
metadata = Arg(['none', 'deepep'])
class ExecMoe:
    __annotations__ = {
        'moe_a2a_backend': Annotated[Literal['none', 'deepep'], metadata]
    }
exec_fields.ExecMoe = ExecMoe
sys.modules[exec_fields.__name__] = exec_fields
from hcu_sglang.server_args import ensure_choices
class Args:
    ATTENTION_BACKEND_CHOICES = ['triton']
    DRAFT_ATTENTION_BACKEND_CHOICES = ['triton']
    QUANTIZATION_CHOICES = ['fp8']
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = []
    ServerArgs = type('ServerArgs', (), {})
    @classmethod
    def add_chunked_prefix_cache_attention_backend(cls, name):
        cls.CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(name)
assert ensure_choices(Args)
assert ensure_choices(Args)
resolved = arg_utils.get_type_hints(ExecMoe, include_extras=True)['moe_a2a_backend']
resolved_metadata = typing.get_args(resolved)[1]
assert resolved_metadata.choices.count('mori-das') == 1
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runtime_does_not_import_legacy_namespace(self):
        for path in (ROOT / 'hcu_sglang').rglob('*.py'):
            tree = ast.parse(path.read_bytes())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('sglang_hcu_plugin'), str(path))
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith('sglang_hcu_plugin') for alias in node.names), str(path))

    def test_scoped_launcher_replaces_global_usercustomize(self):
        self.assertFalse((ROOT / 'usercustomize.py').exists())
        self.assertFalse((ROOT / 'sitecustomize.py').exists())
        self.assertTrue((ROOT / 'hcu_sglang/__main__.py').is_file())
        self.assertTrue((ROOT / 'hcu_sglang/jit_overlay/csrc/deep_select/vendor/LICENSE').is_file())
        self.assertTrue(any((ROOT / 'hcu_sglang').rglob('*.hpp')))


if __name__ == '__main__':
    unittest.main()
