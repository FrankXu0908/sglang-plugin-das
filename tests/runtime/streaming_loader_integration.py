from __future__ import annotations

from functools import wraps
from inspect import signature
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hcu_sglang.multimodal_gen.loader import fsdp_load as owned
from hcu_sglang.multimodal_gen.loader import integration
from sglang.multimodal_gen.runtime.loader import fsdp_load
from sglang.multimodal_gen.runtime.loader.component_loaders import component_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.srt.plugins.hook_registry import HookRegistry, _propagate_patch


class TinyDiT(nn.Module):
    param_names_mapping = {}

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2))
        self.post_load_called = False

    def post_load_weights(self):
        self.post_load_called = True


class StreamingLoaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_full = fsdp_load.load_model_from_full_model_state_dict
        self.base_method = component_loader.WeightOverrideComponentLoader.load_state_dict_model
        self.original_method = TransformerLoader.__dict__.get("load_state_dict_model")
        self.method_signature = signature(TransformerLoader.load_state_dict_model)
        self.loader = TransformerLoader(device=torch.device("cpu"))
        self.cached = ModuleType("_hcu_streaming_loader_cached_alias_test")
        self.cached.load = self.original_full
        sys.modules[self.cached.__name__] = self.cached
        HookRegistry.reset()
        integration.register_hooks()
        HookRegistry.apply_hooks()

    def stop_hooks(self):
        wrapped = fsdp_load.load_model_from_full_model_state_dict
        fsdp_load.load_model_from_full_model_state_dict = self.original_full
        _propagate_patch(wrapped, self.original_full, fsdp_load)
        if self.original_method is None:
            if "load_state_dict_model" in TransformerLoader.__dict__:
                delattr(TransformerLoader, "load_state_dict_model")
        else:
            TransformerLoader.load_state_dict_model = self.original_method
        HookRegistry.reset()

    def tearDown(self):
        self.stop_hooks()
        sys.modules.pop(self.cached.__name__, None)
        self.assertIsNone(integration._streaming_model_class.get())

    def args(self, **overrides):
        values = dict(
            tp_size=1,
            sp_degree=1,
            ulysses_degree=1,
            hsdp_replicate_dim=1,
            hsdp_shard_dim=1,
            pin_cpu_memory=False,
            should_use_fsdp_for_component=lambda component: False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def load(self, server_args, *, loader=None, weights_iterator=None):
        if weights_iterator is None:
            weights_iterator = iter((("weight", torch.arange(4.0).reshape(2, 2)),))
        with patch.object(
            component_loader, "get_local_torch_device", return_value=torch.device("cpu")
        ):
            return (loader or self.loader).load_state_dict_model(
                model_cls=TinyDiT,
                init_params={},
                weight_files=[],
                server_args=server_args,
                component_name="transformer",
                dtype=torch.float32,
                component_starts_on_cpu=False,
                weights_iterator=weights_iterator,
            )

    def test_policy_matches_the_existing_tp_sp_conditions(self):
        cases = [
            ({}, False),
            ({"tp_size": 2}, True),
            ({"sp_degree": 2, "ulysses_degree": 2}, True),
            ({"sp_degree": 2, "ulysses_degree": 1}, False),
            ({"sp_degree": 1, "ulysses_degree": 2}, False),
            ({"tp_size": None, "sp_degree": 0, "ulysses_degree": None}, False),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertEqual(
                    integration._should_use_streaming_state_dict_load(self.args(**values)),
                    expected,
                )

    def test_hooks_apply_and_keep_the_official_signature(self):
        self.assertEqual(len(HookRegistry._patched), 2)
        self.assertEqual(signature(TransformerLoader.load_state_dict_model), self.method_signature)
        self.assertNotIn("streaming_state_dict_load", self.method_signature.parameters)
        self.assertNotIn("streaming_state_dict_load", signature(fsdp_load.maybe_load_fsdp_model).parameters)
        self.assertIs(
            component_loader.WeightOverrideComponentLoader.load_state_dict_model,
            self.base_method,
        )

    def test_single_card_public_load_keeps_the_original_path(self):
        with patch.object(
            owned, "load_model_from_streaming_state_dict", wraps=owned.load_model_from_streaming_state_dict
        ) as streaming:
            model = self.load(self.args())
        streaming.assert_not_called()
        self.assertTrue(torch.equal(model.weight, torch.arange(4.0).reshape(2, 2)))
        self.assertTrue(model.post_load_called)
        self.assertFalse(model.training)
        self.assertFalse(model.weight.requires_grad)

    def test_parallel_arguments_select_streaming_through_the_real_loader(self):
        for values in ({"tp_size": 2}, {"sp_degree": 2, "ulysses_degree": 2}):
            with self.subTest(values=values):
                with patch.object(
                    owned,
                    "load_model_from_streaming_state_dict",
                    wraps=owned.load_model_from_streaming_state_dict,
                ) as streaming:
                    model = self.load(self.args(**values))
                streaming.assert_called_once()
                self.assertTrue(torch.equal(model.weight, torch.arange(4.0).reshape(2, 2)))
                self.assertTrue(model.post_load_called)

    def test_plugin_disabled_uses_original_even_with_parallel_arguments(self):
        self.stop_hooks()
        with patch.object(owned, "load_model_from_streaming_state_dict") as streaming:
            model = self.load(self.args(tp_size=2))
        streaming.assert_not_called()
        self.assertTrue(torch.equal(model.weight, torch.arange(4.0).reshape(2, 2)))

    def test_plain_component_loader_is_not_selected(self):
        loader = component_loader.WeightOverrideComponentLoader(device=torch.device("cpu"))
        with patch.object(owned, "load_model_from_streaming_state_dict") as streaming:
            model = self.load(self.args(tp_size=2), loader=loader)
        streaming.assert_not_called()
        self.assertTrue(model.post_load_called)

    def test_cached_module_alias_is_refreshed(self):
        self.assertIs(self.cached.load, fsdp_load.load_model_from_full_model_state_dict)
        self.assertIsNot(self.cached.load, self.original_full)
        model = TinyDiT().to("meta")
        token = integration._streaming_model_class.set(TinyDiT)
        try:
            with patch.object(
                owned,
                "load_model_from_streaming_state_dict",
                wraps=owned.load_model_from_streaming_state_dict,
            ) as streaming:
                self.cached.load(
                    model,
                    iter((("weight", torch.ones(2, 2)),)),
                    torch.device("cpu"),
                    torch.float32,
                    strict=True,
                )
            streaming.assert_called_once()
            self.assertTrue(torch.equal(model.weight, torch.ones(2, 2)))
        finally:
            integration._streaming_model_class.reset(token)

    def test_existing_native_optimizations_preserve_all_arguments(self):
        for extra in (
            {"preconverted_state_dict": ({}, {})},
            {"keep_checkpoint_mapping": True},
            {"allow_device_tensor_assignment": True},
            {"host_spill": object()},
        ):
            with self.subTest(extra=extra):
                calls = []
                sentinel = object()

                @wraps(self.original_full)
                def baseline(*args, **kwargs):
                    calls.append((args, kwargs))
                    return sentinel

                model = TinyDiT()
                iterator = iter(())
                token = integration._streaming_model_class.set(TinyDiT)
                try:
                    with patch.object(owned, "load_model_from_streaming_state_dict") as streaming:
                        result = integration.around_full_state_dict_load(
                            baseline,
                            model,
                            iterator,
                            torch.device("cpu"),
                            torch.float32,
                            strict=False,
                            **extra,
                        )
                    self.assertIs(result, sentinel)
                    self.assertIs(calls[0][0][1], iterator)
                    self.assertEqual(calls[0][1], {"strict": False, **extra})
                    streaming.assert_not_called()
                finally:
                    integration._streaming_model_class.reset(token)

    def test_different_model_type_keeps_the_original_path(self):
        model = nn.Linear(2, 2, bias=False, device="meta")
        token = integration._streaming_model_class.set(TinyDiT)
        try:
            with patch.object(owned, "load_model_from_streaming_state_dict") as streaming:
                fsdp_load.load_model_from_full_model_state_dict(
                    model,
                    iter((("weight", torch.ones(2, 2)),)),
                    torch.device("cpu"),
                    torch.float32,
                    strict=True,
                    param_names_mapping=lambda name: (name, None, None),
                )
            streaming.assert_not_called()
            self.assertTrue(torch.equal(model.weight, torch.ones(2, 2)))
        finally:
            integration._streaming_model_class.reset(token)

    def test_context_is_reset_after_a_checkpoint_exception(self):
        def broken_checkpoint():
            raise RuntimeError("broken checkpoint")
            yield

        with self.assertRaisesRegex(RuntimeError, "broken checkpoint"):
            self.load(self.args(tp_size=2), weights_iterator=broken_checkpoint())
        self.assertIsNone(integration._streaming_model_class.get())
        model = self.load(self.args())
        self.assertTrue(model.post_load_called)

    def test_bitsandbytes_keeps_native_loading_and_resets_nested_context(self):
        seen = []
        config = SimpleNamespace(get_name=lambda: "bitsandbytes")

        def baseline(model_cls, init_params, server_args):
            seen.append(integration._streaming_model_class.get())
            return "native"

        token = integration._streaming_model_class.set(TinyDiT)
        try:
            result = integration.around_transformer_state_dict_model(
                baseline,
                model_cls=TinyDiT,
                init_params={"quant_config": config},
                server_args=self.args(tp_size=2),
            )
            self.assertEqual(result, "native")
            self.assertEqual(seen, [None])
            self.assertIs(integration._streaming_model_class.get(), TinyDiT)
        finally:
            integration._streaming_model_class.reset(token)


if __name__ == "__main__":
    unittest.main()
