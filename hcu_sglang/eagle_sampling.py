# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from hcu_sglang.utils.detection import is_hcu

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


_sample_context = ContextVar("hcu_glm_ct_mtp_target_sampling", default=None)
lightop_top_k_top_p_sampling_from_probs = None


def _use_native_target_sampling(verify_input, batch, kwargs):
    if (
        batch.forward_mode.is_idle()
        or batch.sampling_info.is_all_greedy
        or verify_input.tree_topk != 1
        or kwargs.get("watermark_state") is not None
    ):
        return False
    from sglang.srt.configs.model_config import is_glm_moe_dsa
    from sglang.srt.runtime_context import get_model, get_spec, process_model_config

    if not is_hcu() or get_spec().speculative_use_rejection_sampling:
        return False
    model_config = process_model_config()
    if not is_glm_moe_dsa(model_config.hf_config):
        return False
    quantization = get_model().quantization
    if quantization is None:
        from hcu_sglang.layers.attention.hisparse import _matches_channel_fp8

        if not _matches_channel_fp8(model_config.hf_config):
            return False
        quantization = model_config.quantization
    return quantization in ("compressed-tensors", "compressed_tensors")


def around_eagle_sample(original, verify_input, batch, logits_output, *args, **kwargs):
    if _sample_context.get() is not None or not _use_native_target_sampling(
        verify_input, batch, kwargs
    ):
        return original(verify_input, batch, logits_output, *args, **kwargs)
    token = _sample_context.set((verify_input, batch.sampling_info, logits_output))
    try:
        return original(verify_input, batch, logits_output, *args, **kwargs)
    finally:
        _sample_context.reset(token)


def around_verify_policy(original, **kwargs):
    if _sample_context.get() is not None:
        return True
    return original(**kwargs)


def around_verify_tree_greedy(original, *args, **kwargs):
    context = _sample_context.get()
    if context is None:
        return original(*args, **kwargs)
    verify_input, sampling_info, logits_output = context
    kwargs["target_predict"] = sample_mtp_target_ids(
        logits_output.next_token_logits,
        sampling_info,
        verify_input.draft_token_num,
        verify_input.positions,
    ).reshape(kwargs["candidates"].shape)
    result = original(*args, **kwargs)

    from sglang.srt.speculative import eagle_utils

    if not eagle_utils._is_hip:
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        group = parallel.attn_tp_group if is_dp_attention_enabled() else parallel.tp_group
        if group.world_size > 1:
            for tensor in result:
                group.broadcast(tensor, src=0)
    return result


def sample_mtp_target_ids(
    next_token_logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    draft_token_num: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Sample linear MTP target tokens on HCU before tree verification."""
    from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    target_probs = F.softmax(next_token_logits / expanded_temperature, dim=-1)
    expanded_top_ks = torch.repeat_interleave(
        sampling_info.top_ks, draft_token_num, dim=0
    )
    expanded_top_ps = torch.repeat_interleave(
        sampling_info.top_ps, draft_token_num, dim=0
    )

    # LightOp implements the common top-k-first/top-p path. Keep the PyTorch
    # path for min-p and request-specific seeds.
    if (
        lightop_top_k_top_p_sampling_from_probs is None
        or sampling_info.sampling_seed is not None
        or sampling_info.need_min_p_sampling
    ):
        expanded_min_ps = torch.repeat_interleave(
            sampling_info.min_ps, draft_token_num, dim=0
        )
        expanded_sampling_seed = (
            None
            if sampling_info.sampling_seed is None
            else torch.repeat_interleave(
                sampling_info.sampling_seed, draft_token_num, dim=0
            )
        )
        return top_k_top_p_min_p_sampling_from_probs_torch(
            target_probs,
            expanded_top_ks,
            expanded_top_ps,
            expanded_min_ps,
            sampling_info.need_min_p_sampling,
            expanded_sampling_seed,
            positions,
        ).to(torch.long)

    return lightop_top_k_top_p_sampling_from_probs(
        target_probs.contiguous(),
        expanded_top_ks,
        expanded_top_ps,
        filter_apply_order="top_k_first",
        deterministic=True,
    ).to(torch.long)


def register_hooks():
    global lightop_top_k_top_p_sampling_from_probs
    if not is_hcu():
        return
    from sglang.srt.speculative import eagle_utils
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from lightop.sampling import top_k_top_p_sampling_from_probs

    if hasattr(eagle_utils, "sample_mtp_target_ids"):
        return
    lightop_top_k_top_p_sampling_from_probs = top_k_top_p_sampling_from_probs
    for target, hook in (
        ("eagle_utils.eagle_sample", around_eagle_sample),
        ("eagle_worker_common.eagle_sample", around_eagle_sample),
        ("eagle_utils._verify_uses_greedy", around_verify_policy),
        ("eagle_utils.verify_tree_greedy_func", around_verify_tree_greedy),
    ):
        HookRegistry.register(
            "sglang.srt.speculative." + target, hook, HookType.AROUND
        )
