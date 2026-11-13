# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Callable, Iterator
from importlib import import_module
from typing import Any

import torch


def iter_hf_to_custom_state_dict(
    hf_param_sd: dict[str, torch.Tensor] | Iterator[tuple[str, torch.Tensor]],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
    valid_target_names: set[str] | None = None,
) -> Iterator[tuple[str, torch.Tensor, tuple[str, Any, Any]]]:
    """Yield custom-formatted checkpoint tensors without building a full dict."""
    to_merge_params = defaultdict(dict)  # type: ignore
    if isinstance(hf_param_sd, dict):
        hf_param_sd = hf_param_sd.items()  # type: ignore
    for source_param_name, full_tensor in hf_param_sd:  # type: ignore
        target_param_name, merge_index, num_params_to_merge = param_names_mapping(
            source_param_name
        )
        if (
            valid_target_names is not None
            and target_param_name != source_param_name
            and source_param_name in valid_target_names
            and target_param_name not in valid_target_names
        ):
            target_param_name = source_param_name
            merge_index = None
            num_params_to_merge = None
        if target_param_name == "" or target_param_name is None:  # type: ignore[comparison-overlap]
            continue
        reverse_entry = (source_param_name, merge_index, num_params_to_merge)
        if merge_index is not None:
            to_merge_params[target_param_name][merge_index] = full_tensor
            if len(to_merge_params[target_param_name]) == num_params_to_merge:
                sorted_tensors = [
                    to_merge_params[target_param_name][i]
                    for i in range(num_params_to_merge)
                ]
                full_tensor = torch.cat(sorted_tensors, dim=0)
                del to_merge_params[target_param_name]
            else:
                continue
        yield target_param_name, full_tensor, reverse_entry


def __getattr__(name):
    return getattr(
        import_module("sglang.multimodal_gen.runtime.loader.utils"), name
    )


def __dir__():
    upstream = import_module("sglang.multimodal_gen.runtime.loader.utils")
    return sorted(set(globals()) | set(dir(upstream)))
