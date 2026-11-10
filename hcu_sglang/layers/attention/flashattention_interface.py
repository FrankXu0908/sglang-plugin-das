"""HCU FlashAttention interface compatible with official SGLang call sites."""

from __future__ import annotations

import json
import os
from typing import Optional, Union

import torch

from hcu_sglang.utils.detection import is_hcu, is_hcu_native_fp8_supported


def _note_fa(kind: str) -> None:
    path = os.environ.get("HCU_ATTN_TRACE")
    if not path:
        return
    counts = _note_fa.counts
    n = counts.get(kind, 0) + 1
    counts[kind] = n
    if n != 1 and n % 200 != 0:
        return
    line = json.dumps({"pid": os.getpid(), "kind": kind, "n": n}, ensure_ascii=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


_note_fa.counts = {}


def _layout_enabled() -> bool:
    from sglang.srt.utils import get_bool_env_var

    return get_bool_env_var("SGLANG_KV_LAYOUT_HCU_FA", default="true")


def is_native_fp8(dtype: torch.dtype) -> bool:
    return is_hcu_native_fp8_supported() and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )


def _apply_out(result, out, return_softmax_lse):
    if out is None:
        return result
    if return_softmax_lse:
        attn_out, lse, *rest = result
        out.copy_(attn_out)
        return (out, lse, *rest)
    out.copy_(result)
    return out


def _restore_hcu_cache_layout(k_cache: torch.Tensor, v_cache: torch.Tensor):
    if not _layout_enabled() or k_cache.ndim != 4 or v_cache.ndim != 4:
        return k_cache, v_cache
    num_pages, page_size, num_heads, head_dim = k_cache.shape
    _, value_page_size, value_heads, value_dim = v_cache.shape
    if page_size != value_page_size or num_heads != value_heads:
        return k_cache, v_cache
    # Uniform BHSD. The NHD carrier is only a view; both caches go back to
    # [pages, heads, page, dim] before the wheel call with layout="bhsd".
    return (
        k_cache.view(num_pages, num_heads, page_size, head_dim),
        v_cache.view(num_pages, value_heads, page_size, value_dim),
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk: Optional[int] = None,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    sfq=None,
    sfk=None,
    sfv=None,
    rel_bias=None,
    rel_bias_prep_cache=None,
    ver=3,
    out=None,
    max_seqlen_k: Optional[int] = None,
):
    from flash_attn import (
        flash_attn_varlen_func as hcu_flash_attn_varlen_func,
        flash_attn_with_kvcache as hcu_flash_attn_with_kvcache,
        vllm_flash_attn_with_kvcache,
    )

    k_cache, v_cache = _restore_hcu_cache_layout(k_cache, v_cache)
    if _layout_enabled() and max_seqlen_q is not None and max_seqlen_q > 1:
        page_capacity = (
            page_table.shape[1] * k_cache.shape[2]
            if max_seqlen_k is None and page_table is not None
            else max_seqlen_k
        )
        _note_fa("fa_varlen_bhsd")
        result = hcu_flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=page_capacity,
            seqused_k=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=page_table,
            fa_version=2,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            layout="bhsd",
        )
        return _apply_out(result, out, return_softmax_lse)

    q_4d = q
    if q.ndim == 3:
        query_len = max_seqlen_q or 1
        q_4d = q.contiguous().view(-1, query_len, q.shape[-2], q.shape[-1])

    if _layout_enabled():
        _note_fa("fa_decode_bhsd")
        result = vllm_flash_attn_with_kvcache(
            q=q_4d,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            num_splits=num_splits,
            layout=1,
        )
        return _apply_out(result, out, return_softmax_lse)

    if not is_native_fp8(k_cache.dtype):
        k_cache = k_cache.to(q.dtype)
        v_cache = v_cache.to(q.dtype)
    result = hcu_flash_attn_with_kvcache(
        q=q_4d,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_softmax_lse=return_softmax_lse,
        num_splits=num_splits,
    )
    return _apply_out(result, out, return_softmax_lse)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    sfq=None,
    sfk=None,
    sfv=None,
    rel_bias=None,
    rel_bias_prep_cache=None,
    ver=3,
    out=None,
):
    from flash_attn import flash_attn_varlen_func as hcu_flash_attn_varlen_func

    result = hcu_flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=return_softmax_lse,
        softcap=softcap,
    )
    return _apply_out(result, out, return_softmax_lse)


def around_flash_attn_with_kvcache(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return flash_attn_with_kvcache(*args, **kwargs)


def around_flash_attn_varlen_func(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return flash_attn_varlen_func(*args, **kwargs)
