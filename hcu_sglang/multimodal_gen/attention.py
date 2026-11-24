"""HCU multimodal attention backend compatibility."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def around_cuda_backend_selection(
    original: Callable[..., str], cls, selected_backend, head_size: int, dtype
) -> str:
    if is_hcu() and getattr(selected_backend, "name", None) == "SLA_ATTN":
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("HCU SLA attention requires fp16 or bf16 inputs.")
        if head_size not in (64, 128):
            raise ValueError("HCU SLA attention supports head sizes 64 and 128.")
        return (
            "hcu_sglang.multimodal_gen.rocm_sparse_linear_attn."
            "RocmSparseLinearAttentionBackend"
        )
    return original(cls, selected_backend, head_size, dtype)


def around_rocm_backend_selection(
    original: Callable[..., str], cls, selected_backend, head_size: int, dtype
) -> str:
    backend_name = getattr(selected_backend, "name", None)
    if is_hcu() and backend_name == "SUBBLOCK_SPARSE_ATTN":
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "subblock_sparse_attn.SubBlockSparseAttentionBackend"
        )
    if is_hcu() and backend_name == "SLA_ATTN":
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("ROCm SLA attention requires fp16 or bf16 inputs.")
        if head_size not in (64, 128):
            raise ValueError("ROCm SLA attention supports head sizes 64 and 128.")
        return (
            "hcu_sglang.multimodal_gen.rocm_sparse_linear_attn."
            "RocmSparseLinearAttentionBackend"
        )
    if (
        is_hcu()
        and backend_name in (None, "FA", "FA2")
        and dtype in (torch.float16, torch.bfloat16)
        and importlib.util.find_spec("flash_attn") is not None
    ):
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "flash_attn_2.FlashAttention2Backend"
        )
    return original(cls, selected_backend, head_size, dtype)


def around_aiter_forward_varlen(
    original: Callable[..., Any],
    attention_impl,
    query,
    key,
    value,
    *,
    cu_seqlens,
    max_seqlen: int,
    cu_seqlens_host=None,
):
    import aiter
    from sglang.multimodal_gen.runtime.platforms.aiter import USE_AITER_GFX942

    if not is_hcu() or USE_AITER_GFX942 or hasattr(aiter, "flash_attn_varlen_func"):
        return original(
            attention_impl,
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )

    attention_func = importlib.import_module(
        "aiter.ops.triton.mha"
    ).flash_attn_varlen_func
    cu_seqlens = cu_seqlens.to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    output = attention_func(
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=attention_impl.softmax_scale,
        causal=attention_impl.causal,
        config={
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "waves_per_eu": 2,
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 1,
        },
    )
    return output[0] if isinstance(output, tuple) else output


def around_minimal_a2a_forward(
    original: Callable[..., Any], attention_op, query, key, value, *args, **kwargs
):
    if is_hcu() and attention_op.pg is None:
        from sglang.multimodal_gen.runtime.distributed import get_sp_group
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_batch = getattr(get_forward_context(), "forward_batch", None)
        if bool(getattr(forward_batch, "enable_sequence_shard", False)):
            try:
                sp_group = get_sp_group()
                process_group = getattr(sp_group, "ulysses_group", None)
                world_size = getattr(sp_group, "ulysses_world_size", 0)
                if process_group is not None and world_size >= 2:
                    stream = attention_op.stream
                    if stream is None and query.is_cuda:
                        stream = torch.cuda.Stream(device=query.device)
                    attention_op.set_context_parallel_group(
                        process_group, None, stream
                    )
            except Exception:
                pass
    return original(attention_op, query, key, value, *args, **kwargs)


def mark_qwen_image21_target(_result, attention, *args, **kwargs) -> None:
    attention.target_attn._hcu_qwen_image21_direct_recv = True


def around_replicated_kv_prefix_split(
    original, attention, q, k_rep, v_rep, k_shard, v_shard, ctx_attn_metadata
):
    if torch.version.hip is not None and getattr(
        attention, "_hcu_qwen_image21_direct_recv", False
    ):
        from hcu_sglang.multimodal_gen.qwen_image21_direct_recv import try_direct_recv

        out = try_direct_recv(
            original, attention, q, k_rep, v_rep, k_shard, v_shard, ctx_attn_metadata
        )
        if out is not None:
            return out
    return original(attention, q, k_rep, v_rep, k_shard, v_shard, ctx_attn_metadata)


def register_hooks(registry=None) -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if registry is not None:
        HookRegistry = registry

    from hcu_sglang.multimodal_gen.subblock_sparse import (
        around_build_dense_impl,
        around_get_runner,
    )

    HookRegistry.register(
        "sglang.multimodal_gen.runtime.layers.attention.backends."
        "subblock_sparse_attn.SubBlockSparseAttentionImpl._build_dense_impl",
        around_build_dense_impl,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.layers.attention.backends."
        "subblock_sparse_attn._get_subblock_sparse_attention_runner",
        around_get_runner,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.platforms.cuda."
        "CudaPlatformBase.get_attn_backend_cls_str",
        around_cuda_backend_selection,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.platforms.rocm."
        "RocmPlatform.get_attn_backend_cls_str",
        around_rocm_backend_selection,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.layers.attention.backends.aiter."
        "AITerImpl.forward_varlen",
        around_aiter_forward_varlen,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.layers.attention.turbo_layer."
        "MinimalA2AAttnOp.forward",
        around_minimal_a2a_forward,
        HookType.AROUND,
    )
    if registry is not None:
        HookRegistry.register(
            "sglang.multimodal_gen.runtime.models.dits.qwen_image21."
            "QwenImage21Attention.__init__",
            mark_qwen_image21_target,
            HookType.AFTER,
        )
        HookRegistry.register(
            "sglang.multimodal_gen.runtime.layers.attention.layer."
            "USPAttention._forward_with_replicated_kv_prefix_split",
            around_replicated_kv_prefix_split,
            HookType.AROUND,
        )
