from functools import lru_cache
from types import MethodType

import torch

from hcu_sglang.utils.detection import is_hcu


def _sorted_active_indices(index, block_counts, num_blocks):
    if block_counts is None:
        return index.sort(dim=-1).values
    order = torch.arange(index.shape[-1], device=index.device)
    active = order < block_counts.unsqueeze(-1)
    ordered = index.masked_fill(~active, num_blocks).sort(dim=-1).values
    return torch.where(active, ordered, index)


def hcu_sparse_attention(q, k, v, index, topk, softmax_scale, block_counts=None):
    from dcu_bsa import bsa_attn_blk64_fwd
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
        _cached_block_sizes,
    )

    sizes = _cached_block_sizes(k.shape[1], k.device)
    out, _ = bsa_attn_blk64_fwd(
        q,
        k,
        v,
        _sorted_active_indices(index, block_counts, sizes.numel()),
        topk,
        block_sizes=sizes,
        q2k_block_nums=block_counts,
        softmax_scale=softmax_scale,
    )
    return out


@lru_cache(maxsize=None)
def _resolve_hcu_runner(device, compute_mode):
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch not in ("gfx936", "gfx938") or compute_mode != "bf16":
        raise ValueError(
            f"DCU SubBlock requires gfx936/gfx938 and bf16, got {arch}/{compute_mode}"
        )
    return hcu_sparse_attention


def around_get_runner(original, device, compute_mode="bf16"):
    if device.type == "cuda" and is_hcu():
        return _resolve_hcu_runner(device, compute_mode)
    return original(device, compute_mode)


def _hcu_fa2_forward_varlen(
    dense_impl, query, key, value, *, cu_seqlens, max_seqlen, cu_seqlens_host=None
):
    from flash_attn import flash_attn_varlen_func

    cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
    output = flash_attn_varlen_func(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=getattr(dense_impl, "dropout_p", 0.0),
        softmax_scale=dense_impl.softmax_scale,
        causal=dense_impl.causal,
    )
    return output[0] if isinstance(output, tuple) else output


def around_build_dense_impl(original, impl, *, causal):
    if not is_hcu():
        return original(impl, causal=causal)

    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        component_attn_backend_context_manager,
    )

    with component_attn_backend_context_manager(None, "subblock_sparse_attn.dense"):
        dense_impl = original(impl, causal=causal)

    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )

    if type(dense_impl).forward_varlen is AttentionImpl.forward_varlen:
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn_2 import (
            FlashAttention2Impl,
        )

        if isinstance(dense_impl, FlashAttention2Impl):
            dense_impl.forward_varlen = MethodType(_hcu_fa2_forward_varlen, dense_impl)
    return dense_impl
