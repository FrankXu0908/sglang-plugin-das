# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import lru_cache

import torch
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_exec, get_forward
from sglang.srt.environ import envs

from hcu_sglang.layers.quantization.compressed_int8 import PrequantizedInt8Input
from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)
enabled = envs.SGLANG_HCU_QWEN_SILU_INT8_FUSION.get() and is_hcu()


def register():
    if enabled:
        HookRegistry.register(
            "sglang.srt.models.qwen2_moe.Qwen2MoeMLP.forward",
            around_forward,
            HookType.AROUND,
        )


@lru_cache(maxsize=1)
def _kernel():
    try:
        from lightop.activation import fuse_silu_mul_quant
    except ImportError:
        return None
    return fuse_silu_mul_quant


def around_forward(original, self, x):
    proj = self.down_proj
    scheme = getattr(proj, "scheme", None)
    if (
        torch.compiler.is_compiling()
        or self._enable_silu_fp4_quant_fusion
        or type(scheme) is not CompressedTensorsW8A8Int8
        or scheme.is_static_input_scheme
        or not scheme.input_symmetric
        or not proj.input_is_parallel
        or proj.input_size_per_partition % 32
        or not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or not x.shape[0]
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or get_forward().sp_active
        or get_exec().deterministic.rl_on_policy_target is not None
    ):
        return original(self, x)

    kernel = _kernel()
    if kernel is None:
        return original(self, x)

    gate_up, _ = self.gate_up_proj(x)
    if (
        isinstance(gate_up, torch.Tensor)
        and gate_up.shape == (x.shape[0], 2 * proj.input_size_per_partition)
        and gate_up.is_contiguous()
        and gate_up.device == x.device
        and gate_up.dtype == torch.bfloat16
        and gate_up.storage_offset() % 32 == 0
    ):
        q, scale = kernel(gate_up)
        output, _ = proj(PrequantizedInt8Input(q, scale, gate_up.dtype))
        _log_dispatch()
        return output

    output, _ = proj(self.act_fn(gate_up))
    return output


@lru_cache(maxsize=1)
def _log_dispatch():
    logger.info("Qwen MLP LightOp SiLU + mul + dynamic INT8 quant fused before down_proj")
