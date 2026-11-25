from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=None)
def _needs_w8a16(device: torch.device) -> bool:
    if not torch.version.hip or device.type != "cuda":
        return False
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx936"


def around_apply(original, method, layer, x, bias=None):
    if method.use_marlin or method.block_quant:
        return original(method, layer, x, bias)
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or not _needs_w8a16(x.device)
    ):
        return original(method, layer, x, bias)
    weight = layer.weight
    scale = layer.weight_scale
    if (
        weight.dtype != torch.float8_e4m3fn
        or weight.ndim != 2
        or weight.device != x.device
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or scale.device != weight.device
        or x.ndim == 0
        or x.shape[-1] != weight.shape[0]
    ):
        return original(method, layer, x, bias)
    weight_nk = weight.t()
    if (
        weight.is_cuda
        and weight_nk.is_contiguous()
        and 0 < weight.numel() <= 2**31
        and scale.ndim <= 2
        and not (torch.is_grad_enabled() and (weight.requires_grad or scale.requires_grad))
    ):
        from hcu_sglang.kernels.ops.quantization.fp8_dequant_triton import (
            dequantize_e4m3fn,
        )

        restored = dequantize_e4m3fn(weight_nk, scale)
    else:
        restored = (weight.float() * scale).to(x.dtype).t()
    return F.linear(x, restored, bias)


def register_hooks(registry=None):
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if registry is not None:
        HookRegistry = registry
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.layers.quantization.fp8."
        "Fp8LinearMethod.apply",
        around_apply,
        HookType.AROUND,
    )
