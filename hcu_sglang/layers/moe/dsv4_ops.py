from typing import Optional, Tuple

import torch
from sglang.kernels.ops.moe import dsv4 as N
from sglang.kernels.jit.utils import is_hip_runtime
from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()
_jit_hash_topk_module = N._jit_hash_topk_module
_jit_mask_topk_module = N._jit_mask_topk_module
_jit_mega_moe_pre_dispatch_module = N._jit_mega_moe_pre_dispatch_module


def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scoring_func == "sqrtsoftplus"
    if is_hip_runtime() and not _is_hcu:
        from sglang.kernels.ops.moe.triton_hash_topk import hash_topk_triton

        return hash_topk_triton(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts,
            routed_scaling_factor,
            scoring_func,
        )
    else:
        num_tokens = router_logits.size(0)
        topk_routed = tid2eid.size(1)
        topk_fused = topk_routed + num_fused_shared_experts
        topk_ids = torch.empty(
            (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
        )
        topk_weights = torch.empty(
            (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
        )
        if num_tokens == 0:
            # Nothing to route; the runtime rejects a launch with no blocks.
            return topk_weights, topk_ids
        module = _jit_hash_topk_module()
        module.hash_topk(
            router_logits,
            input_ids,
            tid2eid,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
        )
        return topk_weights, topk_ids


def mask_topk_ids(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor):
    if topk_ids.size(0) == 0:
        # Nothing to mask; the runtime rejects a launch with no blocks.
        return None
    return _jit_mask_topk_module().run(topk_ids, num_token_non_padded)


def mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int = 32,
    *,
    shared_x_sf: Optional[torch.Tensor] = None,
    shared_block_m: int = 0,
) -> None:
    if x.shape[0] == 0 and x.stride(0) != x.shape[1]:
        x = x.new_empty(x.shape)
    # shared_block_m must match the consuming MoE's paged scale layout
    module = _jit_mega_moe_pre_dispatch_module(
        quant_group_size, shared_x_sf is not None
    )
    module.run(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
        shared_x_sf,
        shared_block_m,
    )


def around_hash_topk(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return hash_topk(*args, **kwargs)


def around_mask_topk_ids(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return mask_topk_ids(*args, **kwargs)


def around_mega_moe_pre_dispatch(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return mega_moe_pre_dispatch(*args, **kwargs)


def around_fused_moe_kernel_sequence(original, *args, **kwargs):
    """Use the HIP-safe DSV4 SwiGLU clamp path for non-EP MoE.

    Upstream uses ``filter_expert=False`` when every expert is local.  Its
    non-filtered DSV4 clamp kernel is CUDA/XPU-only, while the filtered Triton
    activation path is supported on HCU and is equivalent when the local
    expert range covers all experts.  EP decode already has
    ``filter_expert=True`` and is therefore unchanged.
    """
    if (
        _is_hcu
        and kwargs.get("swiglu_limit") is not None
        and kwargs.get("filter_expert") is False
    ):
        kwargs["filter_expert"] = True
    return original(*args, **kwargs)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for name, hook in [
        ("hash_topk", around_hash_topk),
        ("mask_topk_ids", around_mask_topk_ids),
        ("mega_moe_pre_dispatch", around_mega_moe_pre_dispatch),
    ]:
        HookRegistry.register("sglang.kernels.ops.moe.dsv4." + name, hook, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence",
        around_fused_moe_kernel_sequence,
        HookType.AROUND,
    )
