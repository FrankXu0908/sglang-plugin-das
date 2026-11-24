# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MAIN = "sglang.srt.models.hunyuan_v3"
NEXT = "sglang.srt.models.hunyuan_v3_nextn"
OWNED = "hcu_sglang.models.hunyuan_v3"
OWNED_NEXT = "hcu_sglang.models.hunyuan_v3_nextn"


class Module:
    def __init__(self):
        self.module_initialized = True


class LazyValue:
    def __init__(self, factory):
        self.factory = factory

    @property
    def value(self):
        return self.factory()


def original_method(self, *args, **kwargs):
    return "original"


class HunyuanMethodHooksTests(unittest.TestCase):
    def setUp(self):
        self.modules = {}

        def add_module(name):
            module = types.ModuleType(name)
            module.__path__ = []
            self.modules[name] = module
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                if parent not in self.modules:
                    add_module(parent)
                setattr(self.modules[parent], attr, module)
            return module

        existing = {
            MAIN: {
                "HYV3FeedForward": ("__init__", "forward"),
                "HYV3MoEFused": ("__init__", "ebias_weight_loader", "forward", "_forward_single_stream", "_forward_dual_stream"),
                "HYV3Attention": ("__init__", "_resolve_hpc_ops_fp8_attn", "forward"),
                "HYV3DecoderLayer": ("__init__", "forward"),
                "HYV3Model": ("__init__", "forward"),
                "HYV3ForCausalLM": ("__init__", "forward", "get_embed_and_head", "set_embed_and_head", "load_weights"),
            },
            NEXT: {
                "HYV3ModelNextN": ("__init__", "forward"),
                "HYV3ForCausalLMNextN": ("__init__", "forward", "get_embed_and_head", "set_embed_and_head", "load_weights"),
            },
        }
        for name, classes in existing.items():
            module = add_module(name)
            for class_name, methods in classes.items():
                cls = type(class_name, (Module,), {
                    method: (lambda self, *args, **kwargs: Module.__init__(self))
                    if method == "__init__" else original_method
                    for method in methods
                })
                setattr(module, class_name, cls)
        self.cached = {
            name: getattr(self.modules[module], name)
            for module, classes in existing.items()
            for name in classes
        }
        registry_module = add_module("sglang.srt.models.registry")
        registry_module.ModelRegistry = types.SimpleNamespace(models={
            name: self.cached[name] for name in ("HYV3ForCausalLM", "HYV3ForCausalLMNextN")
        })
        self.registry = registry_module.ModelRegistry
        self.registry_before = self.registry.models.copy()
        self.original_init = self.cached["HYV3ForCausalLM"].__init__

        self.hooks = {}

        def register(target, function, hook_type):
            self.assertNotIn(target, self.hooks)
            self.hooks[target] = function

        hooks = add_module("sglang.srt.plugins.hook_registry")
        hooks.HookRegistry = types.SimpleNamespace(register=register)
        hooks.HookType = types.SimpleNamespace(REPLACE="replace")
        detection = add_module("hcu_sglang.utils.detection")
        detection.is_hcu = lambda: True
        location = add_module("sglang.srt.eplb.expert_location")
        location.ModelConfigForExpertLocation = lambda **kwargs: types.SimpleNamespace(**kwargs)

        for name, filename in [(OWNED, "hunyuan_v3.py"), (OWNED_NEXT, "hunyuan_v3_nextn.py")]:
            module = add_module(name)
            module.__dict__.update({
                "nn": types.SimpleNamespace(Module=Module),
                "torch": types.SimpleNamespace(no_grad=lambda: lambda function: function),
                "native_hunyuan_v3": self.modules[MAIN],
                "native_hunyuan_v3_nextn": self.modules[NEXT],
                "HYV3DecoderLayer": self.cached["HYV3DecoderLayer"],
                "get_parallel": lambda: types.SimpleNamespace(
                    pp_group=types.SimpleNamespace(is_first_rank=True, is_last_rank=True),
                    enable_dp_lm_head=False,
                ),
                "ParallelLMHead": lambda *args, **kwargs: types.SimpleNamespace(weight="head"),
                "LogitsProcessor": lambda config: lambda *args: ("logits", args),
                "LazyValue": LazyValue,
                "layer_stack": nullcontext,
            })
            tree = ast.parse((ROOT / "hcu_sglang/models" / filename).read_text(encoding="utf-8"))
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            compiled = ast.Module(body=ast.parse("from __future__ import annotations").body + classes, type_ignores=[])
            exec(compile(ast.fix_missing_locations(compiled), filename, "exec"), module.__dict__)

        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        spec = importlib.util.spec_from_file_location("hunyuan_hooks_under_test", ROOT / "hcu_sglang/models/hunyuan.py")
        self.registrar = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.registrar)

    def apply(self):
        for target, function in self.hooks.items():
            parts = target.split(".")
            for end in range(len(parts) - 1, 0, -1):
                name = ".".join(parts[:end])
                if name in self.modules:
                    obj = self.modules[name]
                    for attr in parts[end:-1]:
                        obj = getattr(obj, attr)
                    setattr(obj, parts[-1], function)
                    break
            else:
                self.fail(target)

    def test_preimported_classes_and_registry_entries_keep_identity(self):
        self.registrar.register_hooks()
        self.apply()
        for name, original in self.registry_before.items():
            self.assertIs(self.registry.models[name], original)
        for name, original in self.cached.items():
            module = NEXT if name.endswith("NextN") else MAIN
            self.assertIs(getattr(self.modules[module], name), original)
        self.assertTrue(self.hooks)
        self.assertTrue(all(not isinstance(function, type) for function in self.hooks.values()))

    def test_real_constructor_works_on_cached_class_and_uses_official_component(self):
        self.registrar.register_hooks()
        self.apply()

        def model_init(model, config, quant_config=None, prefix=""):
            Module.__init__(model)
            model.start_layer = model.end_layer = 0
            model.layers = []
            model.prefix = prefix

        self.cached["HYV3Model"].__init__ = model_init
        config = types.SimpleNamespace(vocab_size=8, hidden_size=4, tie_word_embeddings=False)
        instance = self.cached["HYV3ForCausalLM"](config)
        self.assertTrue(instance.module_initialized)
        self.assertIs(type(instance.model), self.cached["HYV3Model"])
        self.assertEqual(instance.model.prefix, ".model")
        self.assertEqual(instance.routed_experts_weights_of_layer, {})

    def test_added_descriptors_bind_to_official_classes(self):
        self.registrar.register_hooks()
        self.apply()
        main = self.cached["HYV3ForCausalLM"]
        nextn = self.cached["HYV3ForCausalLMNextN"]
        self.assertIsInstance(main.__dict__["routed_experts_weights_of_layer"], property)
        config = types.SimpleNamespace(num_hidden_layers=3, num_experts=8)
        for cls in (main, nextn):
            self.assertIsInstance(cls.__dict__["get_model_config_for_expert_location"], classmethod)
            result = cls.get_model_config_for_expert_location(config)
            self.assertEqual((result.num_layers, result.num_logical_experts), (3, 8))
            self.assertIs(cls.get_model_config_for_expert_location.__self__, cls)
        self.assertTrue(callable(self.cached["HYV3Attention"].forward_sp))
        self.assertTrue(callable(self.cached["HYV3MoEFused"].get_moe_weights))

    def test_real_constructor_graph_preserves_classes_for_tp_sp_and_nextn(self):
        self.registrar.register_hooks()
        self.apply()

        def component(*args, **kwargs):
            return types.SimpleNamespace(args=args, kwargs=kwargs, weight="weight")

        def make_layers(count, factory, prefix, **kwargs):
            return [factory(index, f"{prefix}.{index}") for index in range(count)], 0, count

        for sp in (False, True):
            parallel = types.SimpleNamespace(
                pp_group=types.SimpleNamespace(
                    is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1,
                ),
                enable_dp_lm_head=False, attn_tp_rank=0, attn_tp_size=1, hy3_sp=sp,
            )
            for module in [self.modules[OWNED], self.modules[OWNED_NEXT]]:
                module.__dict__.update({
                    "get_parallel": lambda: parallel,
                    "get_model": lambda: types.SimpleNamespace(kv_cache_dtype="bf16"),
                    "get_exec": lambda: types.SimpleNamespace(moe=types.SimpleNamespace(ep_num_redundant_experts=0)),
                    "get_moe_a2a_backend": lambda: types.SimpleNamespace(is_deepep=lambda: False),
                    "get_moe_impl_class": lambda config: lambda **kwargs: types.SimpleNamespace(
                        should_fuse_routed_scaling_factor_in_topk=True,
                    ),
                    "is_cuda": lambda: False,
                    "is_dp_attention_enabled": lambda: False,
                    "is_dense_ffn_fully_dp": lambda: False,
                    "_is_hcu": True,
                    "_use_lightop": True,
                    "make_layers": make_layers,
                    "get_rope_config": lambda config: (10000, None),
                    "get_rope": component,
                    "declare_attn": component,
                    "declare_ffn": component,
                    "append_stages": lambda *args: ("attn_boundary", "ffn_boundary"),
                })
                module.nn.Linear = component
                module.nn.Parameter = lambda value: types.SimpleNamespace(data=value)
                module.torch.empty = component
                module.torch.float32 = "float32"
                module.torch.bfloat16 = "bfloat16"
                for name in ["MergedColumnParallelLinear", "RowParallelLinear", "ReplicatedLinear",
                             "QKVParallelLinear", "VocabParallelEmbedding", "RMSNorm",
                             "RadixAttention", "TopK", "SiluAndMul"]:
                    setattr(module, name, component)
            config = types.SimpleNamespace(
                vocab_size=8, hidden_size=4, intermediate_size=8,
                hidden_act="silu", num_hidden_layers=2, first_k_dense_replace=1,
                num_attention_heads=2, num_key_value_heads=1, rms_norm_eps=1e-5,
                num_experts=2, num_experts_per_tok=1, moe_intermediate_size=4,
                route_norm=True, num_shared_experts=1,
            )
            model = self.cached["HYV3ForCausalLM"](config)
            self.assertIs(type(model.model), self.cached["HYV3Model"])
            for layer in model.model.layers:
                self.assertIs(type(layer), self.cached["HYV3DecoderLayer"])
                self.assertIs(type(layer.self_attn), self.cached["HYV3Attention"])
                self.assertTrue(layer.module_initialized)
                self.assertEqual(hasattr(layer, "ffn_boundary"), not sp)
            dense, moe = model.model.layers
            self.assertIs(type(dense.mlp), self.cached["HYV3FeedForward"])
            self.assertEqual(dense.mlp.down_proj.kwargs["reduce_results"], sp)
            self.assertIs(type(moe.mlp), self.cached["HYV3MoEFused"])
            self.assertIs(type(moe.mlp.shared_mlp), self.cached["HYV3FeedForward"])
            nextn = self.cached["HYV3ForCausalLMNextN"](config)
            self.assertIs(type(nextn.model), self.cached["HYV3ModelNextN"])
            self.assertIs(type(nextn.model.decoder), self.cached["HYV3DecoderLayer"])
            self.assertEqual(nextn.model.decoder.block_type, "moe")
            self.assertEqual(config.first_k_dense_replace, 1)

    def test_real_forward_accepts_pipeline_argument_and_returns_logits(self):
        self.registrar.register_hooks()
        self.apply()
        instance = self.cached["HYV3ForCausalLM"].__new__(self.cached["HYV3ForCausalLM"])
        calls = []
        instance.model = lambda *args: calls.append(args) or "hidden"
        instance.pp_group = types.SimpleNamespace(is_last_rank=True)
        instance.lm_head = "head"
        instance.logits_processor = lambda *args: args
        result = instance.forward(1, 2, "batch", "embeds", "pipeline")
        self.assertEqual(calls, [(1, 2, "batch", "embeds", "pipeline")])
        self.assertEqual(result, (1, "hidden", "head", "batch"))

    def test_non_hcu_does_not_modify_classes_or_registry(self):
        self.modules["hcu_sglang.utils.detection"].is_hcu = lambda: False
        self.registrar.register_hooks()
        self.assertEqual(self.hooks, {})
        self.assertEqual(self.registry.models, self.registry_before)
        self.assertFalse(hasattr(self.cached["HYV3Attention"], "forward_sp"))
        self.assertIs(self.cached["HYV3ForCausalLM"].__init__, self.original_init)

    def test_declared_hook_targets_match_registration(self):
        self.registrar.register_hooks()
        contract = json.loads((ROOT / "tests/contracts/hcu_source_alignment_targets.json").read_text())
        declared = {name for name in contract["expected_hook_targets"] if name.startswith((MAIN + ".", NEXT + "."))}
        self.assertEqual(set(self.hooks), declared)


if __name__ == "__main__":
    unittest.main()
