# Copyright 2023-2026 SGLang Team
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HIP speculative swap for GLM-5.3 Channel FP8 widths beyond upstream's limit."""

from __future__ import annotations

from functools import lru_cache
import weakref

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from sglang.kernels.ops.kvcache.hisparse import HiSparseSpecState

from .hisparse import _source, enabled


def around_spec_init(original, self, coordinator, *, num_draft_tokens):
    if num_draft_tokens <= 4 or not enabled():
        return original(self, coordinator, num_draft_tokens=num_draft_tokens)

    import torch
    from sglang.kernels.ops.kvcache.hisparse import HiSparseSpecState

    if coordinator.is_dsv4_hisparse or not 5 <= num_draft_tokens <= 6 or coordinator.top_k < 1024:
        raise ValueError("Expected GLM DSA HiSparse with 5-6 draft tokens and top_k >= 1024.")
    workspace_capacity = num_draft_tokens * coordinator.top_k
    if workspace_capacity > 16384:
        raise ValueError("HIP HiSparse speculative swap supports at most 16384 occurrences.")
    self._coordinator = weakref.proxy(coordinator)
    self.enabled = True
    self.num_draft_tokens = num_draft_tokens
    self._scratch_reqs = set()
    layer_num = coordinator.mem_pool_device.layer_num
    max_num_req_slots = coordinator.req_to_token_pool.req_to_token.shape[0]
    if max_num_req_slots > coordinator.device_buffer_size:
        raise ValueError(
            "HiSparse spec request capacity must not exceed device_buffer_size."
        )

    self.scratch_capacity = min(
        workspace_capacity,
        max(
            self.MIN_SCRATCH_CAPACITY,
            workspace_capacity - coordinator.device_buffer_size,
        ),
    )
    hash_size = 1 << max(1, (2 * coordinator.device_buffer_size - 1).bit_length())
    self._cache_index = torch.full(
        (layer_num, max_num_req_slots, 2, hash_size),
        -1,
        dtype=torch.int64,
        device=coordinator.device,
    )
    self._cache_policy = torch.zeros(
        (layer_num, max_num_req_slots + 1, coordinator.device_buffer_size),
        dtype=torch.int32,
        device=coordinator.device,
    )
    metadata_width = max(
        self.NUM_COUNTERS_PER_REQUEST * max_num_req_slots,
        self.NUM_METADATA_VALUES_PER_OCCURRENCE * workspace_capacity,
    )
    self._scratch_state = torch.full(
        (max_num_req_slots + 1, metadata_width),
        -1,
        dtype=torch.int32,
        device=coordinator.device,
    )
    self._scratch_state[0].zero_()
    self.req_to_scratch = torch.zeros(
        (layer_num, max_num_req_slots, self.scratch_capacity),
        dtype=torch.int32,
        device=coordinator.device,
    )
    self.states = tuple(
        HiSparseSpecState(
            cache_index=self._cache_index[layer_id],
            cache_policy=self._cache_policy[layer_id],
            scratch_locs=self.req_to_scratch[layer_id],
            scratch_state=self._scratch_state,
        )
        for layer_id in range(layer_num)
    )

    tokens = torch.arange(
        coordinator.device_buffer_size,
        dtype=torch.int64,
        device=coordinator.device,
    )
    self._hash_slots = ((tokens * self.HASH_MULTIPLIER) & (hash_size - 1)).to(
        torch.long
    )
    self._hash_entries = (tokens << 32) | tokens
    self.top_k_device_locs = torch.full(
        (max_num_req_slots, num_draft_tokens, coordinator.top_k),
        -1,
        dtype=torch.int32,
        device=coordinator.device,
    )


@lru_cache(maxsize=None)
def _jit_spec_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    num_steps: int,
    record_miss_plan: bool,
):
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args

    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        item_size_bytes,
        num_steps,
        record_miss_plan,
        is_arch_support_pdl(),
    )
    return load_jit(
        "hcu_hisparse_spec",
        *template_args,
        cuda_files=[str(_source().with_name("hisparse_spec.cuh"))],
        extra_include_paths=[str(_source().parents[2] / "include")],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_spec",
                f"load_cache_to_device_buffer_spec<{template_args}>",
            )
        ],
    )


def around_spec_load(original, **values):
    steps = values["top_k_tokens"].size(1)
    if steps in (2, 3, 4) or not enabled():
        return original(**values)
    return _load_spec(**values)


def _load_spec(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state: HiSparseSpecState,
    num_real_reqs: torch.Tensor,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
) -> None:
    """Resolve all speculative steps and swap unique misses in one launch pair.

    Optional miss-plan outputs use the same protocol as the single-step HiSparse
    kernel, so shared-index layers can replay only the Host-to-GPU copies with
    ``copy_cache_planned_mla``.
    """
    import torch

    _, num_steps, num_top_k = top_k_tokens.shape
    if not 1 <= num_steps <= 6:
        raise ValueError(
            "HiSparse speculative swap requires "
            f"1-6 steps, "
            f"got {num_steps}."
        )
    hot_buffer_size = state.cache_policy.size(1)
    page_size = device_buffer_tokens.size(1) - hot_buffer_size
    item_size_bytes = host_cache.stride(0) * host_cache.element_size()
    record_miss_plan = miss_src is not None
    if record_miss_plan:
        if miss_dst is None or miss_count is None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        if miss_src.dtype != torch.int64 or miss_dst.dtype != torch.int32:
            raise ValueError("miss_src must be int64 and miss_dst must be int32.")
        if miss_count.dtype != torch.int32:
            raise ValueError("miss_count must be int32.")
        plan_capacity = num_steps * num_top_k
        batch_size = top_k_tokens.size(0)
        if (
            miss_src.ndim != 2
            or miss_dst.ndim != 2
            or miss_src.size(0) < batch_size
            or miss_dst.size(0) < batch_size
            or miss_src.size(1) < plan_capacity
            or miss_dst.size(1) < plan_capacity
        ):
            raise ValueError(
                "speculative miss_src/miss_dst must have shape "
                f"[batch, >= steps * top_k] (capacity {plan_capacity})."
            )
        if miss_count.ndim != 1 or miss_count.numel() < batch_size:
            raise ValueError("speculative miss_count must have shape [batch].")
        if miss_src.stride(0) != miss_dst.stride(0):
            raise ValueError("miss_src/miss_dst row strides must match.")
    else:
        if miss_dst is not None or miss_count is not None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        empty = torch.empty(0)
        miss_src = miss_dst = miss_count = empty

    module = _jit_spec_module(
        item_size_bytes,
        64,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
    )

    module.load_cache_to_device_buffer_spec(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        state.cache_index,
        state.cache_policy,
        state.scratch_locs,
        state.scratch_state,
        num_real_reqs,
        page_size,
        miss_src,
        miss_dst,
        miss_count,
    )
