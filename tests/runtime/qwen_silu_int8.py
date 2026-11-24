# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch

from hcu_sglang.environ import register_envs

register_envs()

from hcu_sglang.layers.quantization import compressed_int8
from hcu_sglang.layers.quantization.compressed_int8 import PrequantizedInt8Input
from hcu_sglang.models.qwen import silu_int8


class QwenSiluInt8Tests(TestCase):
    def test_registers_only_when_enabled(self):
        target = "sglang.srt.models.qwen2_moe.Qwen2MoeMLP.forward"
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(silu_int8, "enabled", enabled),
                patch.object(silu_int8.HookRegistry, "register") as register,
            ):
                silu_int8.register()
                hooks = [call.args[1] for call in register.call_args_list if call.args[0] == target]
                self.assertEqual(hooks, [silu_int8.around_forward] if enabled else [])

    def test_prequantized_input_bypasses_standalone_quant(self):
        q = torch.zeros((4, 128), dtype=torch.int8)
        scale = torch.ones((4, 1))
        layer = SimpleNamespace(
            weight=torch.zeros((128, 64), dtype=torch.int8),
            weight_scale=torch.ones((64, 1)),
        )
        with (
            patch.object(compressed_int8, "per_token_quant_int8") as quant,
            patch.object(compressed_int8.ops, "triton_scaled_mm") as gemm,
        ):
            compressed_int8.apply_weights(
                SimpleNamespace(w8a8_strategy=1), layer,
                PrequantizedInt8Input(q, scale, torch.bfloat16), None,
            )
        quant.assert_not_called()
        self.assertIs(gemm.call_args.args[0], q)
        self.assertIs(gemm.call_args.args[2], scale)
        self.assertIs(gemm.call_args.kwargs["out_dtype"], torch.bfloat16)

    @skipUnless(torch.cuda.is_available(), "requires an HCU")
    def test_passes_vendor_int8_to_down_projection_and_handles_static_fallback(self):
        x = torch.ones((1, 5120), dtype=torch.bfloat16, device="cuda")
        gate_up = torch.ones((1, 17408), dtype=x.dtype, device=x.device)
        scheme = object.__new__(silu_int8.CompressedTensorsW8A8Int8)
        scheme.is_static_input_scheme = False
        scheme.input_symmetric = True
        output = object()
        proj = Mock(
            scheme=scheme,
            input_is_parallel=True,
            input_size_per_partition=8704,
            return_value=(output, None),
        )
        layer = SimpleNamespace(
            down_proj=proj,
            gate_up_proj=Mock(return_value=(gate_up, None)),
            act_fn=Mock(),
            _enable_silu_fp4_quant_fusion=False,
        )
        original = Mock(return_value=output)
        with (
            patch.object(silu_int8, "get_forward", return_value=SimpleNamespace(sp_active=False)),
            patch.object(
                silu_int8, "get_exec",
                return_value=SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=None)),
            ),
        ):
            self.assertIs(silu_int8.around_forward(original, layer, x), output)
            original.assert_not_called()
            layer.act_fn.assert_not_called()
            quantized = proj.call_args.args[0]
            self.assertIsInstance(quantized, PrequantizedInt8Input)
            self.assertEqual(quantized.activation.shape, (1, 8704))
            self.assertEqual(quantized.activation.dtype, torch.int8)
            self.assertEqual(quantized.scale.dtype, torch.float32)
            scheme.is_static_input_scheme = True
            self.assertIs(silu_int8.around_forward(original, layer, x), output)
            original.assert_called_once_with(layer, x)
            layer.gate_up_proj.assert_called_once_with(x)


if __name__ == "__main__":
    main()
