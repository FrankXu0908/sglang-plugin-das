from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from hcu_sglang.multimodal_gen.runtime.layers.quantization import fp8 as Q


@unittest.skipIf(torch is None, "requires Torch")
class DiffusionFp8W8A16Tests(unittest.TestCase):
    def setUp(self):
        self.method = SimpleNamespace(use_marlin=False, block_quant=False)
        self.layer = SimpleNamespace(
            weight=torch.tensor(
                [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=torch.float32
            ).to(torch.float8_e4m3fn),
            weight_scale=torch.tensor([0.125], dtype=torch.float32),
        )
        self.original = Mock(return_value=object())

    def test_scaled_weights_bias_and_noncontiguous_batched_input(self):
        x = torch.arange(16, dtype=torch.bfloat16).reshape(2, 2, 4).transpose(1, 2)
        bias = torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16)
        before = self.layer.weight.view(torch.uint8).clone()
        with patch.object(Q, "_needs_w8a16", return_value=True):
            result = Q.around_apply(self.original, self.method, self.layer, x, bias)
            repeated = Q.around_apply(self.original, self.method, self.layer, x, bias)
        expected = torch.nn.functional.linear(
            x, (self.layer.weight.float() * 0.125).to(torch.bfloat16).t(), bias
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
        torch.testing.assert_close(self.layer.weight.view(torch.uint8), before)
        self.assertEqual(set(vars(self.layer)), {"weight", "weight_scale"})
        self.original.assert_not_called()

    def test_empty_batch_without_bias(self):
        with patch.object(Q, "_needs_w8a16", return_value=True):
            output = Q.around_apply(
                self.original, self.method, self.layer, torch.empty(0, 2, dtype=torch.bfloat16)
            )
        self.assertEqual(output.shape, (0, 3))
        self.original.assert_not_called()

    def test_gfx936_triton_dequant_matches_torch(self):
        if not torch.cuda.is_available() or not Q._needs_w8a16(torch.device("cuda", 0)):
            self.skipTest("requires HIP gfx936")
        from hcu_sglang.kernels.ops.quantization.fp8_dequant_triton import (
            dequantize_e4m3fn,
        )

        codes = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
        codes = torch.cat((codes, codes[:3])).reshape(7, 37)
        weight = codes.view(torch.float8_e4m3fn)
        with torch.inference_mode():
            for value in (0.001674107, -0.125, 2**-149, torch.finfo(torch.float32).max):
                with self.subTest(scale=value):
                    scale = torch.tensor(value, device="cuda", dtype=torch.float32)
                    original = (weight.float() * scale).bfloat16()
                    restored = dequantize_e4m3fn(weight, scale)
                    self.assertTrue(torch.equal(restored.isnan(), original.isnan()))
                    finite_or_inf = ~original.isnan()
                    self.assertTrue(torch.equal(
                        restored.view(torch.int16)[finite_or_inf],
                        original.view(torch.int16)[finite_or_inf],
                    ))

    def test_marlin_and_block_preserve_effective_original(self):
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        for flag in ("use_marlin", "block_quant"):
            with self.subTest(flag=flag):
                method = SimpleNamespace(use_marlin=flag == "use_marlin", block_quant=flag == "block_quant")
                with patch.object(Q, "_needs_w8a16", side_effect=AssertionError("unexpected probe")):
                    result = Q.around_apply(self.original, method, self.layer, x)
                self.assertIs(result, self.original.return_value)
                self.original.assert_called_with(method, self.layer, x, None)

    def test_other_devices_preserve_original(self):
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        with patch.object(Q, "_needs_w8a16", return_value=False):
            result = Q.around_apply(self.original, self.method, self.layer, x)
        self.assertIs(result, self.original.return_value)

    def test_hardware_probe_uses_input_device_and_keeps_native_fp8_devices(self):
        Q._needs_w8a16.cache_clear()
        with patch.object(Q.torch.version, "hip", "7.2"), patch.object(
            Q.torch.cuda, "get_device_properties",
            side_effect=lambda device: SimpleNamespace(gcnArchName={
                0: "gfx938", 1: "gfx936:sramecc+:xnack-", 2: "gfx942", 3: "gfx928"
            }[device.index]),
        ) as probe:
            for index, expected in [(0, False), (1, True), (2, False), (3, False), (1, True)]:
                self.assertEqual(Q._needs_w8a16(torch.device("cuda", index)), expected)
            self.assertEqual(probe.call_count, 4)
        Q._needs_w8a16.cache_clear()

    def test_cpu_and_non_hip_do_not_probe_gpu(self):
        Q._needs_w8a16.cache_clear()
        with patch.object(Q.torch.version, "hip", None), patch.object(
            Q.torch.cuda, "get_device_properties", side_effect=AssertionError("unexpected probe")
        ):
            self.assertFalse(Q._needs_w8a16(torch.device("cuda", 4)))
            self.assertFalse(Q._needs_w8a16(torch.device("cpu")))
        Q._needs_w8a16.cache_clear()

    def test_unsupported_storage_dtype_shape_and_scale_keep_original(self):
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        variants = [
            (self.layer, x.float()),
            (self.layer, torch.ones(1, 4, dtype=torch.bfloat16)),
            (SimpleNamespace(weight=self.layer.weight.float(), weight_scale=self.layer.weight_scale), x),
            (SimpleNamespace(weight=self.layer.weight, weight_scale=torch.ones(3)), x),
            (SimpleNamespace(weight=self.layer.weight, weight_scale=self.layer.weight_scale.to(torch.bfloat16)), x),
        ]
        with patch.object(Q, "_needs_w8a16", return_value=True):
            for layer, value in variants:
                with self.subTest(dtype=value.dtype, scale=layer.weight_scale.shape):
                    self.assertIs(
                        Q.around_apply(self.original, self.method, layer, value),
                        self.original.return_value,
                    )

    def test_compute_errors_propagate_without_trying_original(self):
        error = RuntimeError("GEMM failure")
        with patch.object(Q, "_needs_w8a16", return_value=True), patch.object(
            Q.F, "linear", side_effect=error
        ):
            with self.assertRaises(RuntimeError) as caught:
                Q.around_apply(self.original, self.method, self.layer, torch.ones(1, 2, dtype=torch.bfloat16))
        self.assertIs(caught.exception, error)
        self.original.assert_not_called()

    def test_original_errors_are_not_swallowed(self):
        self.original.side_effect = RuntimeError("original failure")
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            Q.around_apply(self.original, self.method, self.layer, torch.ones(1, 2))

    def test_registers_only_apply_without_weight_loading_or_marlin_hooks(self):
        source = Path(importlib.util.find_spec("sglang").origin).parent / "srt/plugins/hook_registry.py"
        spec = importlib.util.spec_from_file_location("fp8_test_hook_registry", source)
        hooks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hooks)
        registry = Mock()
        with patch.dict(sys.modules, {"sglang.srt.plugins.hook_registry": hooks}):
            Q.register_hooks(registry)
        registry.register.assert_called_once()
        args = registry.register.call_args.args
        self.assertEqual(args[0], "sglang.multimodal_gen.runtime.layers.quantization.fp8.Fp8LinearMethod.apply")
        self.assertIs(args[1], Q.around_apply)
        self.assertEqual(args[2].value, "around")


if __name__ == "__main__":
    unittest.main()
