from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from hcu_sglang.multimodal_gen.loader import fsdp_load, utils
from sglang.multimodal_gen.runtime.loader import fsdp_load as upstream
from sglang.multimodal_gen.runtime.loader import rank_local_checkpoint


class StreamingStateDictTests(unittest.TestCase):
    def model(self, *, bias=False, dtype=torch.float32):
        model = nn.Linear(2, 2, bias=bias, device="meta")
        model.weight = nn.Parameter(model.weight.to(dtype), requires_grad=False)
        if model.bias is not None:
            model.bias = nn.Parameter(model.bias.to(dtype), requires_grad=False)
        return model

    def load(self, model, checkpoint, **kwargs):
        return fsdp_load.load_model_from_streaming_state_dict(
            model,
            iter(checkpoint.items()),
            torch.device("cpu"),
            torch.float32,
            **kwargs,
        )

    def test_existing_official_helpers_and_api_are_reused(self):
        for name in (
            "_make_param_like",
            "_maybe_dequantize_fp8",
            "_format_dtype_mismatch_summary",
            "maybe_load_fsdp_model",
        ):
            self.assertIs(getattr(fsdp_load, name), getattr(upstream, name))
        self.assertIs(
            fsdp_load._get_param_for_weight_loading,
            rank_local_checkpoint.get_param_for_weight_loading,
        )
        self.assertIs(fsdp_load._QUANTIZED_DTYPES, rank_local_checkpoint.QUANTIZED_DTYPES)
        self.assertIs(
            fsdp_load.iter_hf_to_custom_state_dict, utils.iter_hf_to_custom_state_dict
        )
        if hasattr(upstream, "_resolve_missing_param_init"):
            self.assertIs(
                fsdp_load._resolve_missing_param_init,
                upstream._resolve_missing_param_init,
            )
        else:
            self.assertEqual(
                fsdp_load._resolve_missing_param_init.__module__, fsdp_load.__name__
            )

    def test_real_meta_model_materializes_via_owned_iterator(self):
        model = self.model(bias=True)
        checkpoint = {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([5.0, 6.0]),
        }
        with patch.object(
            fsdp_load,
            "iter_hf_to_custom_state_dict",
            wraps=utils.iter_hf_to_custom_state_dict,
        ) as iterator:
            result = self.load(model, checkpoint, strict=True)
        iterator.assert_called_once()
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertFalse(model.weight.is_meta)
        self.assertEqual(model.weight.device.type, "cpu")
        self.assertFalse(model.weight.requires_grad)
        self.assertTrue(torch.equal(model.weight, checkpoint["weight"]))
        self.assertTrue(torch.equal(model.bias, checkpoint["bias"]))

    def test_plain_weights_match_official_full_state_loader(self):
        checkpoint = {
            "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "bias": torch.tensor([4.0, 5.0]),
        }
        expected, actual = self.model(bias=True), self.model(bias=True)
        mapping = utils.get_param_names_mapping({})
        upstream.load_model_from_full_model_state_dict(
            expected,
            iter(checkpoint.items()),
            torch.device("cpu"),
            torch.float32,
            strict=True,
            param_names_mapping=mapping,
        )
        self.load(actual, checkpoint, strict=True, param_names_mapping=mapping)
        for name, tensor in expected.state_dict().items():
            self.assertTrue(torch.equal(actual.state_dict()[name], tensor))
        self.assertEqual(actual.reverse_param_names_mapping, expected.reverse_param_names_mapping)

    def test_target_dtype_controls_materialization(self):
        model = self.model(dtype=torch.bfloat16)
        checkpoint = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        self.load(model, checkpoint, strict=True)
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(model.weight, checkpoint["weight"].to(torch.bfloat16)))

    def test_custom_weight_loader_runs_and_preserves_metadata(self):
        model = self.model()
        called = []

        def weight_loader(parameter, tensor):
            called.append((parameter.tensor_parallel_rank, tensor.shape))
            parameter.data.copy_(tensor.chunk(2, dim=0)[parameter.tensor_parallel_rank])

        model.weight.weight_loader = weight_loader
        model.weight.tensor_parallel_rank = 1
        checkpoint = {"weight": torch.arange(8, dtype=torch.float32).reshape(4, 2)}
        self.load(model, checkpoint, strict=True)
        self.assertEqual(called, [(1, torch.Size([4, 2]))])
        self.assertTrue(torch.equal(model.weight, checkpoint["weight"][2:]))

    def test_pre_fsdp_parameter_metadata_is_used(self):
        model = self.model()
        original = nn.Parameter(torch.empty(2, 2, device="meta"))
        called = []

        def weight_loader(parameter, tensor):
            called.append(True)
            parameter.data.copy_(tensor + 10)

        original.weight_loader = weight_loader
        model._pre_fsdp_weight_loader_params = {"weight": original}
        self.load(model, {"weight": torch.ones(2, 2)}, strict=True)
        self.assertEqual(called, [True])
        self.assertTrue(torch.equal(model.weight, torch.full((2, 2), 11.0)))

    def test_unexpected_key_fails_before_state_assignment(self):
        model = self.model()
        checkpoint = {"weight": torch.ones(2, 2), "unknown": torch.ones(1)}
        with self.assertRaisesRegex(ValueError, "not found in custom model state dict"):
            self.load(model, checkpoint, strict=True)
        self.assertTrue(model.weight.is_meta)

    def test_nonstrict_unknown_key_is_skipped(self):
        model = self.model()
        checkpoint = {"unknown": torch.ones(1), "weight": torch.ones(2, 2)}
        self.load(model, checkpoint, strict=False)
        self.assertTrue(torch.equal(model.weight, checkpoint["weight"]))

    def test_missing_bias_uses_existing_zero_initialization_policy(self):
        model = self.model(bias=True)
        self.load(model, {"weight": torch.ones(2, 2)}, strict=True)
        self.assertTrue(torch.equal(model.bias, torch.zeros(2)))

    def test_explicit_missing_parameter_initialization_is_preserved(self):
        for mode, expected in (("ones", 1.0), ("zeros", 0.0)):
            with self.subTest(mode=mode):
                model = self.model()
                model.weight.missing_param_init = mode
                self.load(model, {}, strict=True)
                self.assertTrue(torch.equal(model.weight, torch.full((2, 2), expected)))

    def test_synthesized_local_attention_weight_stays_plugin_owned(self):
        model = self.model()
        model.local_attn = nn.Module()
        model.local_attn.proj_l = nn.Linear(2, 2, bias=False, device="meta")
        self.load(model, {"weight": torch.ones(2, 2)}, strict=True)
        self.assertTrue(torch.equal(model.local_attn.proj_l.weight, torch.zeros(2, 2)))

    def test_unknown_missing_weight_fails_instead_of_synthesizing_it(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, "New parameter 'weight' is not supported"):
            self.load(model, {}, strict=True)
        self.assertTrue(model.weight.is_meta)

    def test_quantized_duplicate_takes_precedence_in_both_orders(self):
        quantized = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
        unquantized = torch.full((2, 2), 9.0, dtype=torch.bfloat16)
        for checkpoint in (
            {"quantized": quantized, "unquantized": unquantized},
            {"unquantized": unquantized, "quantized": quantized},
        ):
            with self.subTest(order=list(checkpoint)):
                model = self.model(dtype=torch.int8)
                self.load(
                    model,
                    checkpoint,
                    strict=True,
                    param_names_mapping=lambda name: ("weight", None, None),
                )
                self.assertTrue(torch.equal(model.weight, quantized))
                self.assertFalse(model.weight.requires_grad)

    def test_fp8_scale_is_applied_using_official_dequantizer(self):
        model = nn.Module()
        model.proj = self.model()
        checkpoint = {
            "proj.weight_scale": torch.tensor(2.0),
            "proj.weight": torch.ones(2, 2).to(torch.float8_e4m3fn),
        }
        self.load(model, checkpoint, strict=False)
        self.assertTrue(torch.equal(model.proj.weight, torch.full((2, 2), 2.0)))

    def test_out_of_order_slices_materialize_one_fused_parameter(self):
        model = self.model()
        checkpoint = {
            "second": torch.tensor([[3.0, 4.0]]),
            "first": torch.tensor([[1.0, 2.0]]),
        }
        self.load(
            model,
            checkpoint,
            strict=True,
            param_names_mapping=lambda name: ("weight", {"first": 0, "second": 1}[name], 2),
        )
        self.assertTrue(torch.equal(model.weight, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertEqual(model.reverse_param_names_mapping["weight"], ("first", 0, 2))


if __name__ == "__main__":
    unittest.main()
