# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

from contextvars import ContextVar
from inspect import signature
import logging


logger = logging.getLogger(__name__)
_streaming_model_class: ContextVar[type | None] = ContextVar(
    "hcu_diffusion_streaming_model_class", default=None
)


def _should_use_streaming_state_dict_load(server_args):
    tp_size = getattr(server_args, "tp_size", 1) or 1
    sp_degree = getattr(server_args, "sp_degree", 1) or 1
    ulysses_degree = getattr(server_args, "ulysses_degree", 1) or 1
    return bool(tp_size > 1 or (sp_degree > 1 and ulysses_degree > 1))


def around_transformer_state_dict_model(original_fn, *args, **kwargs):
    from sglang.multimodal_gen.runtime.loader.fsdp_load import (
        _is_bitsandbytes_quant_config,
    )

    arguments = signature(original_fn).bind(*args, **kwargs).arguments
    enabled = _should_use_streaming_state_dict_load(arguments["server_args"])
    if _is_bitsandbytes_quant_config(arguments["init_params"].get("quant_config")):
        enabled = False
    token = _streaming_model_class.set(arguments["model_cls"] if enabled else None)
    try:
        return original_fn(*args, **kwargs)
    finally:
        _streaming_model_class.reset(token)


def around_full_state_dict_load(original_fn, *args, **kwargs):
    model_class = _streaming_model_class.get()
    if model_class is None:
        return original_fn(*args, **kwargs)

    bound = signature(original_fn).bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    if not isinstance(arguments["model"], model_class):
        return original_fn(*args, **kwargs)
    if (
        arguments.get("preconverted_state_dict") is not None
        or arguments.get("keep_checkpoint_mapping", False)
        or arguments.get("allow_device_tensor_assignment", False)
        or arguments.get("host_spill") is not None
    ):
        return original_fn(*args, **kwargs)

    from hcu_sglang.multimodal_gen.loader.fsdp_load import (
        load_model_from_streaming_state_dict,
    )

    logger.info("Loading %s with plugin streaming state-dict load", model_class.__name__)
    return load_model_from_streaming_state_dict(
        model=arguments["model"],
        full_sd_iterator=arguments["full_sd_iterator"],
        device=arguments["checkpoint_load_device"],
        param_dtype=arguments["param_dtype"],
        strict=arguments["strict"],
        cpu_offload=arguments["cpu_offload"],
        param_names_mapping=arguments["param_names_mapping"],
    )


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.multimodal_gen.runtime.loader.component_loaders."
        "transformer_loader.TransformerLoader.load_state_dict_model",
        around_transformer_state_dict_model,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.loader.fsdp_load."
        "load_model_from_full_model_state_dict",
        around_full_state_dict_load,
        HookType.AROUND,
    )
