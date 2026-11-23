# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from enum import Enum
from threading import Lock
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
)
from sglang.srt.utils import direct_register_custom_op


_BACKEND_ENV = "SGLANG_HCU_MORI_DAS_BACKEND"
_GROUP_NAME = "mori_ep_group"
_INITIALIZED_GROUPS: set[str] = set()
_OPS: dict[tuple, object] = {}
_LOCK = Lock()


class DispatchDtype(Enum):
    INT8 = "int8"


class _Stage(Enum):
    INITIAL = "initial"
    AFTER_DISPATCH_A = "after_dispatch_a"
    AFTER_DISPATCH_B = "after_dispatch_b"
    AFTER_COMBINE_A = "after_combine_a"


def _m_grouped_i8_gemm_nt_masked(
    a0: torch.Tensor,
    a1: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> None:
    from deepgemm import m_grouped_i8_gemm_nt_masked
    from hcu_sglang.layers.moe.deepep_fp8 import _get_deepgemm_shuffle_unique

    shuffle_unique, _ = _get_deepgemm_shuffle_unique()
    m_grouped_i8_gemm_nt_masked(
        (a0, a1),
        (b0, b1),
        output,
        masked_m,
        expected_m,
        shuffle_unique=shuffle_unique,
    )


def _m_grouped_i8_gemm_nt_masked_fake(
    a0: torch.Tensor,
    a1: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> None:
    return None


if not hasattr(torch.ops.sglang, "mori_grouped_i8_gemm_nt_masked"):
    direct_register_custom_op(
        op_name="mori_grouped_i8_gemm_nt_masked",
        op_func=_m_grouped_i8_gemm_nt_masked,
        mutates_args=["output"],
        fake_impl=_m_grouped_i8_gemm_nt_masked_fake,
    )


def is_enabled() -> bool:
    return os.environ.get(_BACKEND_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _enable() -> None:
    os.environ[_BACKEND_ENV] = "1"


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _group_size(group) -> int:
    size = getattr(group, "world_size", None)
    if size is not None:
        return int(size)
    size = getattr(group, "size", None)
    if callable(size):
        return int(size())
    if size is not None:
        return int(size)
    return torch.distributed.get_world_size(group=_torch_group(group))


def _torch_group(group):
    cpu_group = getattr(group, "cpu_group", None)
    return group if cpu_group is None else cpu_group


def _group_name(group) -> str:
    torch_group = _torch_group(group)
    try:
        name = torch.distributed.distributed_c10d._get_process_group_name(torch_group)
    except Exception:
        name = "None"
    if name == "None":
        name = os.environ.get("SGLANG_MORI_DAS_GROUP_NAME", _GROUP_NAME)
        try:
            torch.distributed.distributed_c10d._register_process_group(name, torch_group)
        except Exception as exc:
            if "already registered" not in str(exc):
                raise
    return name


def _initialize_shmem(group, mori) -> str:
    name = _group_name(group)
    with _LOCK:
        if name in _INITIALIZED_GROUPS:
            return name
        result = mori.shmem.shmem_torch_process_group_init(name)
        if result not in (None, 0):
            raise RuntimeError(f"MoriDAS shmem initialization failed: {result}")
        _INITIALIZED_GROUPS.add(name)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier(group=_torch_group(group))
    return name


def _kernel_type(mori, name: str):
    try:
        return getattr(mori.ops.EpDispatchCombineKernelType, name)
    except AttributeError as exc:
        available = sorted(
            item
            for item in dir(mori.ops.EpDispatchCombineKernelType)
            if item.startswith("InterNode")
        )
        raise ValueError(
            f"Unsupported SGLANG_MORI_DAS_KERNEL_TYPE={name!r}; available: {available}"
        ) from exc


def _configure_chunked_sync_impl() -> str | None:
    """Forward the plugin-owned selector to Mori worker processes.

    SGLang does not preserve arbitrary environment variables when it creates
    scheduler workers, while SGLANG_* configuration is propagated.  Translate
    the public plugin setting immediately before constructing the Mori op.
    """
    value = os.environ.get("SGLANG_MORI_DAS_CHUNKED_SYNC_IMPL")
    if value is None:
        return os.environ.get("MORI_CHUNKED_SYNC_IMPL")
    value = value.strip()
    if value not in {"0", "1", "2"}:
        raise ValueError(
            "SGLANG_MORI_DAS_CHUNKED_SYNC_IMPL must be 0 (single chunk), "
            "1 (single chunk dispatch overlap), or 2 (multi chunk)"
        )
    os.environ["MORI_CHUNKED_SYNC_IMPL"] = value
    return value


def _build_op(
    group,
    router_topk: int,
    num_local_experts: int,
    hidden_size: int,
    params_dtype: torch.dtype,
):
    chunked_sync_impl = _configure_chunked_sync_impl()
    import mori

    group_name = _initialize_shmem(group, mori)
    world_size = _group_size(group)
    rank = int(getattr(group, "rank_in_group", 0))
    max_tokens = _int_env("SGLANG_MORI_DAS_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128)
    block_num = _int_env("SGLANG_MORI_DAS_BLOCK_NUM", 32)
    rdma_block_num = _int_env("SGLANG_MORI_DAS_RDMA_BLOCK_NUM", 16)
    warps = _int_env("SGLANG_MORI_DAS_WARP_NUM_PER_BLOCK", 8)
    chunk_size = _int_env("SGLANG_MORI_DAS_CHUNK_SIZE", 32)
    num_qp = _int_env("SGLANG_MORI_DAS_NUM_QP_PER_PE", 4)
    group_size = _int_env("SGLANG_MORI_DAS_GROUP_SIZE", 0) or hidden_size
    kernel_name = os.environ.get(
        "SGLANG_MORI_DAS_KERNEL_TYPE", "InterNodeV1LLChunkedSync"
    )
    dispatch_dtype = os.environ.get(
        "SGLANG_MORI_DAS_DISPATCH_DTYPE", DispatchDtype.INT8.value
    ).lower()
    if dispatch_dtype != DispatchDtype.INT8.value:
        raise ValueError("The pluginized MoriDAS backend currently supports INT8 dispatch only")
    if block_num <= 0 or not 0 < rdma_block_num < block_num:
        raise ValueError("MoriDAS requires 0 < RDMA block count < total block count")
    if hidden_size % group_size:
        raise ValueError("SGLANG_MORI_DAS_GROUP_SIZE must divide the hidden size")
    if kernel_name == "InterNodeV1LLChunkedSync" and (chunk_size != 32 or warps != 8):
        raise ValueError("InterNodeV1LLChunkedSync requires chunk size 32 and 8 warps")

    key = (
        group_name,
        rank,
        world_size,
        hidden_size,
        str(params_dtype),
        max_tokens,
        num_local_experts,
        router_topk,
        block_num,
        rdma_block_num,
        warps,
        num_qp,
        kernel_name,
        group_size,
        chunk_size,
        chunked_sync_impl,
        _bool_env("SGLANG_MORI_DAS_COMBINE_ZERO_COPY"),
    )
    with _LOCK:
        if key in _OPS:
            return _OPS[key]
        config = mori.ops.EpDispatchCombineConfig(
            data_type=params_dtype or torch.bfloat16,
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_size,
            scale_dim=hidden_size // group_size,
            scale_type_size=torch.float32.itemsize,
            max_token_type_size=(params_dtype or torch.bfloat16).itemsize,
            max_num_inp_token_per_rank=max_tokens,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=router_topk,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_num_per_block=warps,
            num_qp_per_pe=num_qp,
            kernel_type=_kernel_type(mori, kernel_name),
            dispatch_quant_type="int8",
            chunk_size=chunk_size,
            combine_zero_copy=_bool_env("SGLANG_MORI_DAS_COMBINE_ZERO_COPY"),
        )
        _OPS[key] = mori.ops.EpDispatchCombineOp(config)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier(group=_torch_group(group))
        return _OPS[key]


class MoriDasDispatcher(BaseDispatcher):
    """Synchronous MoriDAS dispatcher with split-phase scheduling boundaries.

    The A/B methods only expose the same state transitions as DeepEP and the
    invasive MoriDAS implementation. They do not enable TBO/SBO: the plugin
    still creates one dispatcher and executes the same synchronous Mori ops.
    """

    def __init__(
        self,
        group,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: Optional[torch.dtype] = None,
        mori_op=None,
        **_,
    ):
        super().__init__()
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype or torch.bfloat16
        self.mori_op = mori_op or _build_op(
            group,
            router_topk,
            num_local_experts,
            hidden_size,
            self.params_dtype,
        )
        for method in ("dispatch_standard_moe", "combine_standard_moe"):
            if not hasattr(self.mori_op, method):
                raise TypeError(f"MoriDAS op must implement {method}()")
        self._num_tokens = 0
        self._out_dtype = self.params_dtype
        self._stage = _Stage.INITIAL

    def dispatch(self, hidden_states, topk_output):
        self.dispatch_a(hidden_states, topk_output)
        return self.dispatch_b()

    def dispatch_a(self, hidden_states, topk_output) -> None:
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        self._num_tokens = hidden_states.shape[0]
        self._out_dtype = hidden_states.dtype
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights
        if hidden_states.shape[0] == 0:
            hidden_states = hidden_states.new_zeros((1, hidden_states.shape[-1]))
            topk_ids = torch.full(
                (1, self.router_topk), -1, dtype=torch.int32, device=topk_ids.device
            )
            topk_weights = torch.zeros(
                (1, self.router_topk), dtype=topk_weights.dtype, device=topk_weights.device
            )

        self._dispatch_intermediate_state = (
            hidden_states,
            topk_ids,
            topk_weights,
        )

    @torch._dynamo.disable()
    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        hidden_states, topk_ids, topk_weights = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state

        packed, counts, _, _ = self.mori_op.dispatch_standard_moe(
            input=hidden_states,
            scales=None,
            weights=topk_weights,
            indices=topk_ids,
        )
        if not isinstance(packed, (tuple, list)) or len(packed) != 2:
            raise RuntimeError("INT8 MoriDAS dispatch must return (hidden_states, scales)")
        packed_hidden, packed_scale = packed
        expected_m = (
            hidden_states.shape[0] * _group_size(self.group) * self.router_topk
            + self.num_experts
        ) // self.num_experts
        return DeepEPLLDispatchOutput(
            hidden_states=packed_hidden,
            hidden_states_scale=packed_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            masked_m=counts,
            expected_m=min(packed_hidden.shape[1], expected_m),
        )

    def combine(self, combine_input: DeepEPLLCombineInput):
        self.combine_a(combine_input)
        return self.combine_b()

    def combine_a(self, combine_input: DeepEPLLCombineInput) -> None:
        if not isinstance(combine_input, DeepEPLLCombineInput):
            raise TypeError("MoriDAS combine requires DeepEPLLCombineInput")
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        self._combine_intermediate_state = (
            combine_input.hidden_states,
            combine_input.topk_ids,
            combine_input.topk_weights,
        )

    @torch._dynamo.disable()
    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        hidden_states, topk_ids, topk_weights = self._combine_intermediate_state
        del self._combine_intermediate_state
        output = self.mori_op.combine_standard_moe(
            input=hidden_states,
            indices=topk_ids,
            weights=topk_weights,
        )
        output = output[: self._num_tokens]
        return output if output.dtype == self._out_dtype else output.to(self._out_dtype)

    def _update_stage(self, old_stage: _Stage, new_stage: _Stage) -> None:
        if self._stage is not old_stage:
            raise RuntimeError(
                f"Invalid MoriDAS dispatcher stage: expected {old_stage.value}, "
                f"got {self._stage.value}"
            )
        self._stage = new_stage

    def requires_registered_combine_input_buffer(self) -> bool:
        config = getattr(self.mori_op, "config", None)
        kernel = getattr(config, "kernel_type", None)
        name = getattr(kernel, "name", str(kernel).rsplit(".", 1)[-1])
        return name in {"InterNodeV1LLChunked", "InterNodeV1LLChunkedSync"}

    def get_standard_moe_combine_input_buffer(self, dtype, hidden_dim):
        getter = getattr(self.mori_op, "get_standard_moe_combine_input_buffer", None)
        if getter is None:
            raise RuntimeError("The selected Mori kernel requires a registered combine buffer")
        return getter(dtype=dtype, hidden_dim=hidden_dim)


def run_moe_core(layer, dispatch_output):
    """Run the slimquant Marlin masked MoE core for MoriDAS output."""
    hidden_states = dispatch_output.hidden_states
    hidden_states_scale = dispatch_output.hidden_states_scale
    masked_m = dispatch_output.masked_m
    expected_m = min(hidden_states.shape[1], dispatch_output.expected_m)
    if hidden_states_scale is None or masked_m is None:
        raise RuntimeError(
            "MoriDAS INT8 dispatch requires activation scales and expert counts"
        )

    num_groups, m, _ = hidden_states.shape
    w13_scale = layer.w13_weight_scale
    gateup = torch.empty(
        (num_groups, m, w13_scale.size(1)),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    torch.ops.sglang.mori_grouped_i8_gemm_nt_masked(
        hidden_states,
        hidden_states_scale,
        layer.w13_weight,
        w13_scale,
        gateup,
        masked_m,
        expected_m,
    )
    swiglu_limit = layer.moe_runner_config.swiglu_limit
    if swiglu_limit is not None and envs.SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT.get():
        try:
            from lightop.activation import fuse_silu_mul_clamp_quant
        except ImportError:
            from lightop import fuse_silu_mul_clamp_quant

        q_a2, q_a2_scale = fuse_silu_mul_clamp_quant(
            input=gateup,
            limit=swiglu_limit,
            mask_m=masked_m,
            expect_m=expected_m,
        )
    else:
        if swiglu_limit is not None:
            half = gateup.shape[-1] // 2
            gateup[..., :half].clamp_(max=swiglu_limit)
            gateup[..., half:].clamp_(min=-swiglu_limit, max=swiglu_limit)
        q_a2, q_a2_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(gateup, masked_m)

    n2 = layer.w2_weight_scale.size(1)
    if layer.dispatcher.requires_registered_combine_input_buffer():
        down_output = layer.dispatcher.get_standard_moe_combine_input_buffer(
            dtype=torch.bfloat16,
            hidden_dim=n2,
        )
    else:
        down_output = torch.empty(
            (num_groups, m, n2), device=q_a2.device, dtype=torch.bfloat16
        )
    expected_shape = (num_groups, m, n2)
    if tuple(down_output.shape) != expected_shape:
        raise RuntimeError(
            f"MoriDAS combine buffer shape mismatch: expected {expected_shape}, "
            f"got {tuple(down_output.shape)}"
        )
    torch.ops.sglang.mori_grouped_i8_gemm_nt_masked(
        q_a2,
        q_a2_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        down_output,
        masked_m,
        expected_m,
    )
    return DeepEPLLCombineInput(
        hidden_states=down_output,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


def around_missing_backend(original, cls, value):
    if value == "mori-das":
        _enable()
        return cls.CUSTOMIZED
    return original(cls, value)


def around_is_deepep(original, backend):
    if is_enabled() and backend.is_customized():
        return True
    return original(backend)


def around_create_dispatcher(original, moe_runner_config, quant_method):
    from sglang.srt.layers.moe import get_moe_a2a_backend

    backend = get_moe_a2a_backend()
    if not (is_enabled() and backend.is_customized()):
        return original(moe_runner_config, quant_method)
    from sglang.srt.runtime_context import get_parallel

    return MoriDasDispatcher(
        group=get_parallel().moe_ep_group,
        router_topk=moe_runner_config.top_k,
        num_experts=moe_runner_config.num_experts,
        num_local_experts=moe_runner_config.num_local_experts,
        hidden_size=moe_runner_config.hidden_size,
        params_dtype=moe_runner_config.params_dtype,
    )


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.moe.utils.MoeA2ABackend._missing_",
        around_missing_backend,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.utils.MoeA2ABackend.is_deepep",
        around_is_deepep,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.fused_moe_triton.layer.create_moe_dispatcher",
        around_create_dispatcher,
        HookType.AROUND,
    )
