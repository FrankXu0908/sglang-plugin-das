from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from hcu_sglang.utils.detection import is_hcu


@lru_cache(maxsize=1)
def _masked_fa():
    try:
        import flash_attn
    except ImportError:
        return None
    return getattr(flash_attn, "flash_attn_with_mask_func", None)


@torch.library.custom_op("hcu_diffusion::masked_attention", mutates_args=())
def _masked_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, scale: float,
) -> torch.Tensor:
    return _masked_fa()(
        q, k, v, mask, dropout_p=0.0, softmax_scale=scale,
        causal=False, bhsd=False,
    )


@_masked_attention.register_fake
def _masked_attention_fake(q, k, v, mask, scale):
    return torch.empty_like(q, memory_format=torch.contiguous_format)


@torch.compiler.disable
def _mask_kind(mask):
    if torch.is_floating_point(mask):
        if bool((mask == 0).all()):
            return "full"
        return "bias"
    if bool((mask == 1).all()):
        return "full"
    if not bool(((mask == 0) | (mask == 1)).all()):
        return "bias"
    if not bool(mask.bool().any(dim=-1).all()):
        return "bias"
    return "binary"


def after_ltx_text_connector(result, stage, batch, server_args):
    if not is_hcu():
        return result
    normalized = []
    for name in ("prompt_attention_mask", "negative_attention_mask"):
        mask = getattr(result, name, None)
        if isinstance(mask, torch.Tensor) and _mask_kind(mask) == "full":
            normalized.append(name)
    if getattr(result, "do_classifier_free_guidance", False) and len(normalized) != 2:
        return result
    for name in normalized:
        setattr(result, name, None)
    return result


def _supports_masked_fa(q, k, v):
    return (
        not torch.is_grad_enabled()
        and q.ndim == k.ndim == v.ndim == 4
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype
        and q.device == k.device == v.device
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
        and k.shape[1:3] == v.shape[1:3]
        and k.shape[2] > 0
        and q.shape[2] % k.shape[2] == 0
        and q.shape[1] > 0 and k.shape[1] > 0
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and _masked_fa() is not None
    )


def around_local_attention(original, attention, q, k, v, attn_mask=None):
    if q.device.type != "cuda" or not is_hcu():
        return original(attention, q, k, v, attn_mask=attn_mask)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return original(attention, q, k, v, attn_mask=attn_mask)
    if attn_mask is None:
        return original(attention, q, k, v, attn_mask=None).contiguous()

    from sglang.multimodal_gen.runtime.layers.attention.layer import (
        get_request_skip_softmax_params,
    )

    if (
        not attention.is_cross_attention
        and get_request_skip_softmax_params() is not None
    ):
        return original(attention, q, k, v, attn_mask=attn_mask)
    if torch.cuda.is_current_stream_capturing():
        return original(attention, q, k, v, attn_mask=attn_mask).contiguous()

    target = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
    mask_shape = attn_mask.shape
    if attn_mask.ndim == 2:
        mask_shape = (mask_shape[0], 1, 1, mask_shape[1])
    elif attn_mask.ndim == 3:
        mask_shape = (mask_shape[0], 1, mask_shape[1], mask_shape[2])
    if len(mask_shape) != 4 or any(
        actual not in (1, expected) for actual, expected in zip(mask_shape, target)
    ):
        return original(attention, q, k, v, attn_mask=attn_mask)

    kind = _mask_kind(attn_mask)
    if (
        kind == "full"
        and not attention._attn_impl_ctor_kwargs.get("causal", False)
        and not attention._attn_impl_ctor_kwargs.get("dropout_p", 0.0)
        and not getattr(attention.attn_impl, "causal", False)
        and not getattr(attention.attn_impl, "dropout", 0.0)
    ):
        return original(attention, q, k, v, attn_mask=None).contiguous()
    if kind not in ("full", "binary") or not _supports_masked_fa(q, k, v):
        return original(attention, q, k, v, attn_mask=attn_mask).contiguous()

    mask = (
        attn_mask == 0 if torch.is_floating_point(attn_mask) else attn_mask.bool()
    ).to(device=q.device)
    if mask.ndim == 2:
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    q_len, k_len = q.shape[1], k.shape[1]
    q_pad, k_pad = -q_len % 128, -k_len % 64
    if mask.shape[-1] == k_len and k_pad:
        mask = F.pad(mask, (0, k_pad), value=False)
    if mask.shape[-2] == q_len and q_pad:
        mask = F.pad(mask, (0, 0, 0, q_pad), value=False)
    mask = mask.expand(q.shape[0], q.shape[2], q_len + q_pad, k_len + k_pad)
    if mask.stride(-1) != 1:
        return original(attention, q, k, v, attn_mask=attn_mask).contiguous()
    if q_pad:
        q = F.pad(q, (0, 0, 0, 0, 0, q_pad))
    if k_pad:
        k = F.pad(k, (0, 0, 0, 0, 0, k_pad))
        v = F.pad(v, (0, 0, 0, 0, 0, k_pad))
    out = _masked_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), mask,
        float(attention.softmax_scale),
    )
    return out[:, :q_len].contiguous()


def register_hooks(registry=None):
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if registry is not None:
        HookRegistry = registry
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.layers.attention.layer.LocalAttention.forward",
        around_local_attention, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.pipelines_core.stages."
        "model_specific_stages.ltx_2.text_connector.LTX2TextConnectorStage.forward",
        after_ltx_text_connector, HookType.AFTER,
    )
