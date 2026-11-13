# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune
# Copyright 2024 The TorchTune Authors.
# Copyright 2025 The sglang-diffusion Authors.

from collections import Counter, defaultdict
from collections.abc import Callable, Generator
from typing import Any

import torch
from torch import nn
from torch.distributed._tensor import distribute_tensor
from torch.distributed.fsdp import FSDPModule
from torch.nn.modules.module import _IncompatibleKeys

from sglang.multimodal_gen.runtime.loader import fsdp_load as _upstream
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    _DTYPE_MISMATCH_EXAMPLE_LIMIT,
    _format_dtype_mismatch_summary,
    _make_param_like,
    _maybe_dequantize_fp8,
)
from sglang.multimodal_gen.runtime.loader.rank_local_checkpoint import (
    QUANTIZED_DTYPES as _QUANTIZED_DTYPES,
    get_param_for_weight_loading as _get_param_for_weight_loading,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from hcu_sglang.multimodal_gen.loader.utils import (
    get_param_names_mapping,
    iter_hf_to_custom_state_dict,
)

logger = init_logger(__name__)
SYNTHESIZED_ZERO_INIT_PARAM_PATTERNS = ("local_attn.proj_l.",)


def _resolve_missing_param_init(
    param_name: str, actual_param: torch.nn.Parameter | None
) -> str | None:
    missing_param_init = (
        getattr(actual_param, "missing_param_init", None)
        if actual_param is not None
        else None
    )
    if missing_param_init is None and any(
        pattern in param_name for pattern in SYNTHESIZED_ZERO_INIT_PARAM_PATTERNS
    ):
        return "zeros"
    return missing_param_init


_resolve_missing_param_init = getattr(
    _upstream, "_resolve_missing_param_init", _resolve_missing_param_init
)


# TODO(mick): need refactor, to move out checkpoint-specific adjustments
def load_model_from_streaming_state_dict(
    model: FSDPModule | torch.nn.Module,
    full_sd_iterator: Generator[tuple[str, torch.Tensor], None, None],
    device: torch.device,
    param_dtype: torch.dtype | None,
    strict: bool = False,
    cpu_offload: bool = False,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
) -> _IncompatibleKeys:
    """Load weights while streaming HF tensors into their custom names.

    Unlike load_model_from_full_model_state_dict, this avoids keeping the full
    remapped checkpoint state dict resident at once. It still builds the final
    load_state_dict payload, which contains materialized model parameters.
    """
    if param_names_mapping is None:
        param_names_mapping = get_param_names_mapping({})

    meta_sd = model.state_dict()
    param_dict = dict(model.named_parameters())
    is_fsdp_model = isinstance(model, FSDPModule) or any(
        hasattr(p, "device_mesh") for p in meta_sd.values()
    )

    sharded_sd = {}
    skipped_checkpoint_keys: list[str] = []
    reverse_param_names_mapping = {}
    loaded_checkpoint_dtypes: dict[str, torch.dtype] = {}
    fp8_aux_param_sd: dict[str, torch.Tensor] = {}
    non_quantized_dtype_mismatch_counts: Counter[tuple[torch.dtype, torch.dtype]] = (
        Counter()
    )
    non_quantized_dtype_mismatch_examples: dict[
        tuple[torch.dtype, torch.dtype], list[str]
    ] = defaultdict(list)
    quantized_dtype_mismatch_counts: Counter[tuple[torch.dtype, torch.dtype]] = (
        Counter()
    )
    quantized_dtype_mismatch_examples: dict[
        tuple[torch.dtype, torch.dtype], list[str]
    ] = defaultdict(list)

    streamed_sd_iterator = iter_hf_to_custom_state_dict(
        full_sd_iterator,
        param_names_mapping,
        valid_target_names=set(meta_sd.keys()),
    )
    for target_param_name, full_tensor, reverse_entry in streamed_sd_iterator:
        existing_dtype = loaded_checkpoint_dtypes.get(target_param_name)
        if existing_dtype is not None and existing_dtype != full_tensor.dtype:
            existing_is_quantized = existing_dtype in _QUANTIZED_DTYPES
            current_is_quantized = full_tensor.dtype in _QUANTIZED_DTYPES
            if existing_is_quantized and not current_is_quantized:
                logger.debug(
                    "Keeping quantized duplicate for %s: existing=%s new=%s",
                    target_param_name,
                    existing_dtype,
                    full_tensor.dtype,
                )
                continue
            if current_is_quantized and not existing_is_quantized:
                logger.debug(
                    "Replacing non-quantized duplicate for %s: existing=%s new=%s",
                    target_param_name,
                    existing_dtype,
                    full_tensor.dtype,
                )
        loaded_checkpoint_dtypes[target_param_name] = full_tensor.dtype
        reverse_param_names_mapping[target_param_name] = reverse_entry
        if target_param_name.endswith(".weight_scale"):
            fp8_aux_param_sd[target_param_name] = full_tensor

        meta_sharded_param = meta_sd.get(target_param_name)
        if meta_sharded_param is None:
            if strict or is_fsdp_model:
                raise ValueError(
                    f"Parameter {target_param_name} not found in custom model state dict. The hf to custom mapping may be incorrect."
                )
            skipped_checkpoint_keys.append(target_param_name)
            continue

        target_dtype = meta_sharded_param.dtype
        full_tensor = _maybe_dequantize_fp8(
            full_tensor, target_dtype, target_param_name, fp8_aux_param_sd
        )

        if full_tensor.dtype != target_dtype:
            mismatch_key = (full_tensor.dtype, target_dtype)
            if (
                full_tensor.dtype in _QUANTIZED_DTYPES
                or target_dtype in _QUANTIZED_DTYPES
            ):
                quantized_dtype_mismatch_counts[mismatch_key] += 1
                if (
                    len(quantized_dtype_mismatch_examples[mismatch_key])
                    < _DTYPE_MISMATCH_EXAMPLE_LIMIT
                ):
                    quantized_dtype_mismatch_examples[mismatch_key].append(
                        target_param_name
                    )
            else:
                non_quantized_dtype_mismatch_counts[mismatch_key] += 1
                if (
                    len(non_quantized_dtype_mismatch_examples[mismatch_key])
                    < _DTYPE_MISMATCH_EXAMPLE_LIMIT
                ):
                    non_quantized_dtype_mismatch_examples[mismatch_key].append(
                        target_param_name
                    )

        if not hasattr(meta_sharded_param, "device_mesh"):
            full_tensor = full_tensor.to(device=device, dtype=target_dtype)
            actual_param = _get_param_for_weight_loading(
                model, param_dict, target_param_name
            )
            weight_loader = (
                getattr(actual_param, "weight_loader", None)
                if actual_param is not None
                else None
            )
            if weight_loader is not None:
                assert actual_param is not None
                sharded_tensor = torch.empty_like(
                    meta_sharded_param, device=device, dtype=target_dtype
                )
                requires_grad = getattr(meta_sharded_param, "requires_grad", False)
                temp_param = _make_param_like(actual_param, sharded_tensor)
                if not (
                    sharded_tensor.is_floating_point() or sharded_tensor.is_complex()
                ):
                    requires_grad = False
                temp_param.requires_grad = requires_grad
                try:
                    weight_loader(temp_param, full_tensor)
                except AssertionError as exc:
                    raise AssertionError(
                        "Failed to shard/load parameter "
                        f"{target_param_name}: full_tensor.shape={tuple(full_tensor.shape)}, "
                        f"meta_sharded_param.shape={tuple(meta_sharded_param.shape)}, "
                        f"temp_param.shape={tuple(temp_param.shape)}, "
                        f"param_cls={type(actual_param).__name__}"
                    ) from exc
                sharded_tensor = temp_param.data
            else:
                sharded_tensor = full_tensor

            if cpu_offload and not is_fsdp_model:
                sharded_tensor = sharded_tensor.cpu()
        else:
            full_tensor = full_tensor.to(device=device, dtype=target_dtype)
            actual_param = _get_param_for_weight_loading(
                model, param_dict, target_param_name
            )
            weight_loader = (
                getattr(actual_param, "weight_loader", None)
                if actual_param is not None
                else None
            )
            if weight_loader is not None:
                assert actual_param is not None
                tp_sharded_tensor = torch.empty(
                    tuple(actual_param.shape),
                    device=device,
                    dtype=target_dtype,
                )
                temp_param = _make_param_like(actual_param, tp_sharded_tensor)
                if not (
                    tp_sharded_tensor.is_floating_point()
                    or tp_sharded_tensor.is_complex()
                ):
                    temp_param.requires_grad = False
                try:
                    weight_loader(temp_param, full_tensor)
                except AssertionError as exc:
                    raise AssertionError(
                        "Failed to TP-shard/load FSDP parameter "
                        f"{target_param_name}: full_tensor.shape={tuple(full_tensor.shape)}, "
                        f"meta_sharded_param.shape={tuple(meta_sharded_param.shape)}, "
                        f"temp_param.shape={tuple(temp_param.shape)}, "
                        f"param_cls={type(actual_param).__name__}"
                    ) from exc
                full_tensor = temp_param.data
            sharded_tensor = distribute_tensor(
                full_tensor,
                meta_sharded_param.device_mesh,
                meta_sharded_param.placements,
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.to("cpu")

        sharded_sd[target_param_name] = nn.Parameter(
            sharded_tensor, requires_grad=False
        )

    model.reverse_param_names_mapping = reverse_param_names_mapping

    if non_quantized_dtype_mismatch_counts:
        logger.debug(
            "Casting checkpoint tensors to target dtype during load: %s",
            _format_dtype_mismatch_summary(
                non_quantized_dtype_mismatch_counts,
                non_quantized_dtype_mismatch_examples,
            ),
            main_process_only=True,
            local_main_process_only=True,
        )

    if quantized_dtype_mismatch_counts:
        logger.warning(
            "Dtype mismatches detected for quantized parameters during load: %s",
            _format_dtype_mismatch_summary(
                quantized_dtype_mismatch_counts,
                quantized_dtype_mismatch_examples,
            ),
            main_process_only=True,
            local_main_process_only=True,
        )

    if skipped_checkpoint_keys:
        logger.warning(
            "Checkpoint keys not loaded (no matching model parameter) %s",
            (
                skipped_checkpoint_keys[:20]
                if len(skipped_checkpoint_keys) > 20
                else skipped_checkpoint_keys
            ),
        )
        if len(skipped_checkpoint_keys) > 20:
            logger.warning(
                "... and %d more skipped keys.",
                len(skipped_checkpoint_keys) - 20,
            )

    unused_keys = set(meta_sd.keys()) - set(sharded_sd.keys())
    if unused_keys:
        logger.warning("Found unloaded parameters in meta state dict: %s", unused_keys)

    LEGACY_ALLOWED_NEW_PARAM_PATTERNS = [
        "gate_compress",
        "wcscales",
        "wtscale",
        "input_scale",
        "weight_scale",
        "bias",
        "norm_q",
        "norm_k",
        "weight_scale",
    ]
    for new_param_name in unused_keys:
        meta_sharded_param = meta_sd.get(new_param_name)
        meta_sharded_param_dtype = meta_sharded_param.dtype
        actual_param = param_dict.get(new_param_name)
        missing_param_init = _resolve_missing_param_init(new_param_name, actual_param)

        if missing_param_init is None and not any(
            pattern in new_param_name for pattern in LEGACY_ALLOWED_NEW_PARAM_PATTERNS
        ):
            logger.error(
                "Unsupported new parameter: %s. Allowed legacy patterns: %s",
                new_param_name,
                LEGACY_ALLOWED_NEW_PARAM_PATTERNS,
            )
            raise ValueError(
                f"New parameter '{new_param_name}' is not supported. "
                "Checkpoint-specific synthesized parameters should either match "
                f"{LEGACY_ALLOWED_NEW_PARAM_PATTERNS} or declare missing_param_init."
            )

        if missing_param_init == "ones" or any(
            p in new_param_name
            for p in (
                "wcscales",
                "wtscale",
                "input_scale",
                "weight_scale",
                "norm_q",
                "norm_k",
            )
        ):
            init_like = torch.ones_like
        elif missing_param_init == "zeros" or missing_param_init is None:
            init_like = torch.zeros_like
        else:
            raise ValueError(
                f"Unsupported missing_param_init={missing_param_init!r} for {new_param_name}"
            )

        if not hasattr(meta_sharded_param, "device_mesh"):
            sharded_tensor = init_like(
                meta_sharded_param, device=device, dtype=meta_sharded_param_dtype
            )
            if cpu_offload and not is_fsdp_model:
                sharded_tensor = sharded_tensor.cpu()
        else:
            full_tensor = init_like(
                meta_sharded_param, device=device, dtype=meta_sharded_param_dtype
            )
            sharded_tensor = distribute_tensor(
                full_tensor,
                meta_sharded_param.device_mesh,
                meta_sharded_param.placements,
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
        sharded_sd[new_param_name] = nn.Parameter(sharded_tensor)

    return model.load_state_dict(sharded_sd, strict=strict, assign=True)


def __getattr__(name):
    return getattr(_upstream, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_upstream)))
