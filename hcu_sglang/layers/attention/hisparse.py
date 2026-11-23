# Copyright 2023-2026 SGLang Team
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from pathlib import Path

from hcu_sglang.utils.detection import is_hcu


def _matches_glm53_shape(config):
    dimensions = {
        "num_hidden_layers": 78,
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "index_topk": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
    }
    return all(getattr(config, key, None) == value for key, value in dimensions.items())


def _matches_channel_fp8(config):
    if not _matches_glm53_shape(config):
        return False
    quantization = getattr(config, "quantization_config", None)
    if not isinstance(quantization, dict):
        return False
    groups = quantization.get("config_groups", {})
    if quantization.get("quant_method") != "compressed-tensors" or not groups:
        return False
    for group in groups.values():
        weights = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        if not (
            weights.get("type") == activation.get("type") == "float"
            and weights.get("num_bits") == activation.get("num_bits") == 8
            and weights.get("strategy") == "channel"
            and weights.get("dynamic") is False
            and weights.get("block_structure") is None
            and weights.get("group_size") is None
            and weights.get("symmetric") is True
            and activation.get("strategy") == "token"
            and activation.get("dynamic") is True
            and activation.get("block_structure") is None
            and activation.get("group_size") is None
            and activation.get("symmetric") is True
        ):
            return False
    return True


def _matches_channel_w4a8(config):
    quant = getattr(config, "quantization_config", None) or {}
    if not isinstance(quant, dict) or quant.get("quant_method") != "compressed-tensors":
        return False
    has_w4_experts = False
    for group in (quant.get("config_groups") or {}).values():
        weight = group.get("weights") or {}
        act = group.get("input_activations") or {}
        if (weight.get("type"), weight.get("strategy"), weight.get("num_bits")) not in (
            ("int", "channel", 4), ("int", "channel", 8),
        ) or (act.get("type"), act.get("strategy"), act.get("num_bits"), act.get("dynamic")) != (
            "int", "token", 8, True,
        ):
            return False
        if weight["num_bits"] == 4 and any("experts" in target for target in group.get("targets", ())):
            has_w4_experts = True
    return has_w4_experts


def enabled(view=None, *, w4a8_only=False):
    if not is_hcu():
        return False
    from sglang.srt.arg_groups.model_override_base import model_config_of, resolved_view
    from sglang.srt.runtime_context import get_context
    from hcu_sglang.layers.glm53_decode import model_matches

    if view is None:
        try:
            server_args = get_context().server_args
        except ValueError:
            return False
        if server_args is None:
            return False
        view = resolved_view(server_args)
    if not view.enable_hisparse or view.kv_cache_dtype != "fp8_e4m3":
        return False
    config = model_config_of(view).hf_config
    if not model_matches(config):
        return False
    if not w4a8_only and _matches_channel_fp8(config):
        return True
    return (
        view.disaggregation_mode == "decode"
        and _matches_glm53_shape(config)
        and _matches_channel_w4a8(config)
    )


def around_backend_resolution(original, view):
    if not enabled(view):
        return original(view)
    if view.dcp_size > 1:
        raise ValueError("HCU GLM FP8 HiSparse does not support DCP.")
    declared = {}
    for field in ("dsa_prefill_backend", "dsa_decode_backend"):
        backend = getattr(view, field)
        if backend is None:
            declared[field] = "flashmla_kv"
        elif backend != "flashmla_kv":
            raise ValueError(
                "HCU GLM FP8 HiSparse requires both DSA backends to be flashmla_kv; "
                "raw and scaled FP8 KV layouts cannot be mixed."
            )
    return declared


def around_validate_backend(original, server_args, attr, label):
    from sglang.srt.arg_groups.model_override_base import resolved_view

    view = resolved_view(server_args)
    if not enabled(view):
        return original(server_args, attr, label)
    if getattr(view, attr) not in (None, "flashmla_kv"):
        raise ValueError(
            f"HCU GLM FP8 HiSparse requires the flashmla_kv {label} backend."
        )
    return None


def _source():
    return (
        Path(__file__).resolve().parents[2] / "jit_overlay/csrc/kvcacheio/hisparse.cuh"
    )


@lru_cache(maxsize=None)
def _sparse_module(cache_args):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    template_args = make_cpp_args(*cache_args[1:])
    return load_jit(
        "hcu_hisparse_cache",
        *make_cpp_args(*cache_args),
        cuda_files=[str(_source())],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
        extra_include_paths=[str(_source().parents[2] / "include")],
    )


def around_sparse_module(
    original,
    item_size_bytes,
    block_size,
    num_top_k,
    hot_buffer_size,
    is_mla=False,
    is_dsv4_layout=False,
    top_k_block_size=1,
    top_k_is_blocks=False,
    record_miss_plan=False,
    skip_io=False,
):
    values = (
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        top_k_block_size,
        top_k_is_blocks,
        record_miss_plan,
        skip_io,
    )
    if not enabled():
        return original(*values)
    return _sparse_module(values)


@lru_cache(maxsize=None)
def _copy_module(block_size, is_mla, is_dsv4_layout, skip_io):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = (block_size, is_mla, is_dsv4_layout, skip_io)
    template_args = make_cpp_args(*args)
    return load_jit(
        "hcu_hisparse_copy_planned",
        *make_cpp_args(*args),
        cuda_files=[str(_source())],
        cuda_wrappers=[("copy_cache_planned", f"copy_cache_planned<{template_args}>")],
        extra_include_paths=[str(_source().parents[2] / "include")],
    )


def around_copy_module(original, block_size, is_mla, is_dsv4_layout, skip_io):
    if not enabled():
        return original(block_size, is_mla, is_dsv4_layout, skip_io)
    return _copy_module(min(block_size, 256), is_mla, is_dsv4_layout, skip_io)


def around_validate_feature_combos(original, server_args):
    from sglang.srt.arg_groups.model_override_base import resolving_view

    result = original(server_args)
    view = resolving_view(server_args)
    if not enabled(view):
        return result
    if view.disaggregation_mode not in ("null", "decode") or view.page_size != 64:
        raise ValueError(
            "HCU GLM FP8 HiSparse requires unified serving or PD decode, "
            "with page size 64."
        )
    if view.speculative_algorithm in (None, "NONE"):
        return result
    width = view.speculative_num_draft_tokens
    if (
        view.speculative_algorithm not in ("EAGLE", "NEXTN")
        or view.enable_multi_layer_eagle
        or view.speculative_eagle_topk != 1
        or view.speculative_adaptive
        or view.speculative_num_steps is None
        or width is None
        or not 2 <= width <= 6
        or width != view.speculative_num_steps + 1
    ):
        raise ValueError(
            "HCU GLM FP8 HiSparse requires fixed linear EAGLE/NextN with "
            "top-k 1, up to 6 draft tokens and draft tokens equal to steps + 1."
        )
    if view.pp_size != 1 or view.dcp_size != 1:
        raise ValueError("HCU GLM FP8 HiSparse MTP requires PP=1 and DCP=1.")
    if view.enable_two_batch_overlap or view.enable_pdmux:
        raise ValueError("HCU GLM FP8 HiSparse MTP requires one verify transaction.")
    if view.disaggregation_mode == "decode" and (
        view.disaggregation_transfer_backend not in ("mooncake", "fake")
    ):
        raise ValueError("HCU GLM FP8 HiSparse MTP requires Mooncake or fake transfer.")
    return result


def after_capture_prepare(result, runner, *args, **kwargs):
    coordinator = result[0].hisparse_coordinator
    if coordinator is not None and enabled() and coordinator.spec_swap.enabled:
        coordinator.num_real_reqs.zero_()
    return result


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from hcu_sglang.mem_cache.hisparse_capacity import around_allocator_init
    from .hisparse_spec import around_spec_init, around_spec_load

    for target, hook in (
        (
            "sglang.srt.mem_cache.allocator.hisparse.HiSparseTokenToKVPoolAllocator.__init__",
            around_allocator_init,
        ),
        (
            "sglang.srt.managers.hisparse_coordinator.HiSparseSpecSwapManager.__init__",
            around_spec_init,
        ),
        (
            "sglang.srt.managers.hisparse_coordinator.load_cache_to_device_buffer_spec_mla",
            around_spec_load,
        ),
        (
            "sglang.srt.arg_groups.model_hook._dsa_split_backend_resolution",
            around_backend_resolution,
        ),
        (
            "sglang.srt.arg_groups.hisparse_hook.validate_hisparse_dsa_backend",
            around_validate_backend,
        ),
        (
            "sglang.kernels.ops.kvcache.hisparse._jit_sparse_module",
            around_sparse_module,
        ),
        (
            "sglang.kernels.ops.kvcache.hisparse._jit_copy_planned_module",
            around_copy_module,
        ),
        (
            "sglang.srt.arg_groups.hisparse_hook.validate_hisparse_feature_combos",
            around_validate_feature_combos,
        ),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_prepare",
        after_capture_prepare,
        HookType.AFTER,
    )
