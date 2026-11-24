from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput, DeepEPNormalDispatchOutput,
    )

from typing import Any, Dict, Optional
import torch
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from hcu_sglang.layers.quantization.compressed_marlin import SlimQuantCompressedTensorsMarlinConfig
from hcu_sglang.layers.quantization.hcu_deepgemm_w8a8_utils import (
    is_hcu_pair32_int8, use_int8_deepgemm_asm, run_w8a8_int8_contiguous_gemm,
)
from hcu_sglang.layers.moe.deepep_fp8 import (
    _ep_scatter_with_optional_lightop, _ep_gather_with_optional_lightop,
    get_offloader, copy_list_to_gpu_no_ce,
)
from sglang.srt.utils import get_int_env_var
from lightop.activation import fuse_silu_mul_quant, fuse_silu_mul_quant_ep
from deepgemm import m_grouped_i8_gemm_nt_contiguous
try:
    from lightop.activation import fuse_silu_mul_clamp_quant
except ImportError:
    from lightop import fuse_silu_mul_clamp_quant

def fuse_silu_mul_quant_ep_wrapper(
    input: torch.Tensor,
    tokens_per_expert: Optional[torch.Tensor] = None,
    num_local_tokens_tensor: Optional[torch.Tensor] = None,
    topk: int = 1,
    expect_m: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fuse_silu_mul_quant_ep(
        input, tokens_per_expert, num_local_tokens_tensor, topk, expect_m
    )

def fuse_silu_mul_quant_ep_fake(
    input: torch.Tensor,
    tokens_per_expert: Optional[torch.Tensor] = None,
    num_local_tokens_tensor: Optional[torch.Tensor] = None,
    topk: int = 1,
    expect_m: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    E, T, H = input.shape
    d = H // 2
    output = torch.empty(E, T, d, dtype=torch.int8, device=input.device)
    scales = torch.empty((E, T, 1), device=input.device, dtype=torch.float32)
    return output, scales

from sglang.srt.utils import direct_register_custom_op

if not hasattr(torch.ops.sglang, "fuse_silu_mul_quant_ep"):
    direct_register_custom_op(
        op_name="fuse_silu_mul_quant_ep",
        op_func=fuse_silu_mul_quant_ep_wrapper,
        mutates_args=[],
        fake_impl=fuse_silu_mul_quant_ep_fake,
    )

_w8a8_int8_ep_workspace_tokens = get_int_env_var("SGLANG_REUSE_W8A8_INT8_EP_MOE_WORKSPACE", 32768)
_w8a8_int8_ep_workspace: Dict[Any, torch.Tensor] = {}

def _ep_moe_workspace_empty(
    name: str,
    shape: tuple,
    *,
    device: torch.device,
    dtype: torch.dtype,
    zero: bool = False,
) -> torch.Tensor:
    if _w8a8_int8_ep_workspace_tokens <= 0:
        return (
            torch.zeros(shape, device=device, dtype=dtype)
            if zero
            else torch.empty(shape, device=device, dtype=dtype)
        )

    key = (str(device), dtype, name)
    buf = _w8a8_int8_ep_workspace.get(key)
    if (
        buf is None
        or buf.device != device
        or buf.dtype != dtype
        or buf.shape[1:] != shape[1:]
        or buf.shape[0] < shape[0]
    ):
        alloc_shape = (
            max(shape[0], _w8a8_int8_ep_workspace_tokens),
            *shape[1:],
        )
        buf = torch.empty(alloc_shape, device=device, dtype=dtype)
        _w8a8_int8_ep_workspace[key] = buf

    out = buf.narrow(0, 0, shape[0])
    if zero:
        out.zero_()
    return out

def _apply_swiglu_limit_inplace(
    gateup_output: torch.Tensor, swiglu_limit: Optional[float]
) -> None:
    if swiglu_limit is None:
        return

    half = gateup_output.shape[-1] // 2
    gateup_output[..., :half].clamp_(max=swiglu_limit)
    gateup_output[..., half:].clamp_(min=-swiglu_limit, max=swiglu_limit)

def _is_w8a8_int8_deepgemm_quant(
    self, quant_config: Optional[QuantizationConfig]
) -> bool:
    if isinstance(quant_config, W8A8Int8Config):
        return True
    if isinstance(quant_config, SlimQuantCompressedTensorsMarlinConfig):
        return True

    scheme = getattr(self, "scheme", None)
    return (
        scheme is not None
        and type(scheme).__name__ == "CompressedTensorsW8A8Int8MoE"
    )

def forward_groupgemm_w8a8_int8_contiguous(
    self,
    dispatch_output: DeepEPNormalDispatchOutput,
):
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output

    assert self.quant_method is not None
    assert self.moe_runner_config.activation == "silu"
    if hidden_states_scale is None:
        raise RuntimeError(
            "DeepEP W8A8 INT8 dispatch must return activation scales."
        )
    if num_recv_tokens_per_expert is None:
        return hidden_states.bfloat16()

    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states.bfloat16()

    M, K = hidden_states.size()
    w13_shape = getattr(self, "_w8a8_int8_w13_weight_shape", None)
    if w13_shape is not None:
        N = w13_shape[1]
    else:
        N = self.w13_weight_scale.size(1)
    w13_weight_int8 = (self.w13_weight_deepgemm, self.w13_weight_scale)
    w2_weight_int8 = (self.w2_weight_deepgemm, self.w2_weight_scale)

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    input_tensor = [
        _ep_moe_workspace_empty(
            "w8a8_int8_contiguous_input",
            (all_tokens, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ),
        _ep_moe_workspace_empty(
            "w8a8_int8_contiguous_input_scale",
            (all_tokens, hidden_states_scale.shape[-1]),
            device=hidden_states.device,
            dtype=torch.float32,
        ),
    ]
    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)

    counts_are_aligned = all(
        count % 256 == 0 for count in num_recv_tokens_per_expert
    )
    m_indices, output_index = _ep_scatter_with_optional_lightop(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        input_tensor[0],
        input_tensor[1],
        all_tokens,
        counts_are_aligned=counts_are_aligned,
    )

    # Keep this as a flat storage so both output views stay contiguous.
    bf16_workspace_tokens = max(all_tokens, _w8a8_int8_ep_workspace_tokens)
    bf16_gemm_workspace = _ep_moe_workspace_empty(
        "w8a8_int8_contiguous_bf16_gemm",
        (bf16_workspace_tokens * max(N, K),),
        device=hidden_states_device,
        dtype=torch.bfloat16,
    )
    gateup_output = bf16_gemm_workspace.narrow(0, 0, all_tokens * N).view(
        all_tokens, N
    )
    if not counts_are_aligned:
        gateup_output.zero_()

    if (
        self.deepep_mode.is_normal()
        or self.use_int8_w8a8_deepgemm_asm
        or self.use_pair32_w8a8_int8_deepgemm
    ):
        run_w8a8_int8_contiguous_gemm(
            self,
            input_tensor,
            w13_weight_int8,
            gateup_output,
            m_indices,
        )
    else:
        m_grouped_i8_gemm_nt_contiguous(
            input_tensor,
            w13_weight_int8,
            gateup_output,
            m_indices,
        )
    del input_tensor

    swiglu_limit = self.moe_runner_config.swiglu_limit
    if (
        swiglu_limit is not None
        and envs.SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT.get()
    ):
        q_a2_all, q_a2_scale = fuse_silu_mul_clamp_quant(
            gateup_output, float(swiglu_limit)
        )
    else:
        _apply_swiglu_limit_inplace(gateup_output, swiglu_limit)
        q_a2_all, q_a2_scale = fuse_silu_mul_quant(gateup_output)
    del gateup_output

    down_output = bf16_gemm_workspace.narrow(0, 0, all_tokens * K).view(
        all_tokens, K
    )

    if (
        self.deepep_mode.is_normal()
        or self.use_int8_w8a8_deepgemm_asm
        or self.use_pair32_w8a8_int8_deepgemm
    ):
        run_w8a8_int8_contiguous_gemm(
            self,
            (q_a2_all, q_a2_scale),
            w2_weight_int8,
            down_output,
            m_indices,
        )
    else:
        m_grouped_i8_gemm_nt_contiguous(
            (q_a2_all, q_a2_scale),
            w2_weight_int8,
            down_output,
            m_indices,
        )

    gather_out = torch.empty(
        hidden_states_shape,
        device=hidden_states_device,
        dtype=torch.bfloat16,
    )

    _ep_gather_with_optional_lightop(
        down_output, topk_ids, topk_weights, output_index, gather_out
    )
    del down_output

    return gather_out

def forward_groupgemm_w8a8_int8_masked(
    self,
    dispatch_output: DeepEPLLDispatchOutput,
):
    hidden_states, hidden_states_scale, _, _, masked_m, expected_m = dispatch_output
    assert self.quant_method is not None
    assert self.moe_runner_config.activation == "silu"
    if hidden_states_scale is None:
        raise RuntimeError(
            "DeepEP W8A8 INT8 dispatch must return activation scales."
        )

    num_groups, m, _ = hidden_states.size()
    expected_m = min(m, expected_m)

    n1 = self.w13_weight_scale.size(1)
    gateup_output = torch.empty(
        (num_groups, m, n1),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    if self.use_int8_w8a8_deepgemm_asm:
        from deepgemm import m_grouped_w8a8_gemm_nt_masked

        m_grouped_w8a8_gemm_nt_masked(
            (hidden_states, hidden_states_scale),
            (self.w13_weight_deepgemm_masked, self.w13_weight_scale),
            gateup_output,
            masked_m,
            expected_m,
        )
    elif self.use_pair32_w8a8_int8_deepgemm:
        from deepgemm import m_grouped_i8_gemm_nt_masked

        w13_weight_int8 = (self.w13_weight_deepgemm_masked, self.w13_weight_scale)
        if hidden_states.dtype != torch.int8 or w13_weight_int8[0].dtype != torch.int8:
            raise RuntimeError(
                "pair32 masked INT8 GEMM requires INT8 dispatch and weights, "
                f"got activation={hidden_states.dtype}, weight={w13_weight_int8[0].dtype}"
            )
        m_grouped_i8_gemm_nt_masked(
            (hidden_states.contiguous(), hidden_states_scale.contiguous()),
            (w13_weight_int8[0].contiguous(), w13_weight_int8[1].contiguous()),
            gateup_output,
            masked_m.contiguous(),
            expected_m,
            shuffle_unique=0,
        )
    else:
        from deepgemm.m_group_gemm import m_grouped_w8a8_gemm_nt_masked_ll

        w13_weight_int8 = (self.w13_weight_deepgemm, self.w13_weight_scale)
        m_grouped_w8a8_gemm_nt_masked_ll(
            (hidden_states, hidden_states_scale),
            w13_weight_int8,
            gateup_output,
            masked_m,
            expected_m,
        )

    if self.moe_runner_config.swiglu_limit is None:
        q_a2_all, q_a2_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(
            gateup_output, masked_m
        )
    else:
        if not envs.SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT.get():
            _apply_swiglu_limit_inplace(
                gateup_output, self.moe_runner_config.swiglu_limit
            )
            q_a2_all, q_a2_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(
                gateup_output, masked_m
            )
        else:
            q_a2_all, q_a2_scale = fuse_silu_mul_clamp_quant(
                input=gateup_output,
                limit=self.moe_runner_config.swiglu_limit,
                mask_m=masked_m,
                expect_m=expected_m,
            )

    del gateup_output

    n2 = self.w2_weight_scale.size(1)
    down_output = torch.empty(
        (num_groups, m, n2),
        device=q_a2_all.device,
        dtype=torch.bfloat16,
    )

    if self.use_int8_w8a8_deepgemm_asm:
        from deepgemm import m_grouped_w8a8_gemm_nt_masked

        m_grouped_w8a8_gemm_nt_masked(
            (q_a2_all, q_a2_scale),
            (self.w2_weight_deepgemm_masked, self.w2_weight_scale),
            down_output,
            masked_m,
            expected_m,
        )
    elif self.use_pair32_w8a8_int8_deepgemm:
        from deepgemm import m_grouped_i8_gemm_nt_masked

        w2_weight_int8 = (self.w2_weight_deepgemm_masked, self.w2_weight_scale)
        m_grouped_i8_gemm_nt_masked(
            (q_a2_all.contiguous(), q_a2_scale.contiguous()),
            (w2_weight_int8[0].contiguous(), w2_weight_int8[1].contiguous()),
            down_output,
            masked_m.contiguous(),
            expected_m,
            shuffle_unique=0,
        )
    else:
        w2_weight_int8 = (self.w2_weight_deepgemm, self.w2_weight_scale)
        m_grouped_w8a8_gemm_nt_masked_ll(
            (q_a2_all, q_a2_scale),
            w2_weight_int8,
            down_output,
            masked_m,
            expected_m,
        )

    return down_output


def configure_layer(layer, quant_config):
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.utils import get_bool_env_var
    layer.use_int8_w8a8_deepgemm = is_hcu() and get_bool_env_var("SGLANG_USE_DEEPGEMM_MOE") and _is_w8a8_int8_deepgemm_quant(layer, quant_config)
    if layer.use_int8_w8a8_deepgemm:
        layer.use_int8_w8a8_deepgemm_asm = use_int8_deepgemm_asm()
        layer.use_pair32_w8a8_int8_deepgemm = is_hcu_pair32_int8() and not layer.use_int8_w8a8_deepgemm_asm
        layer.use_w8a8_marlin = False


def run_moe_core(layer, dispatch_output):
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from hcu_sglang.layers.moe import deepep_current as D
    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        output = forward_groupgemm_w8a8_int8_contiguous(layer, dispatch_output)
        wrapper = D.DeepEPNormalCombineInput
    elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
        output = forward_groupgemm_w8a8_int8_masked(layer, dispatch_output)
        wrapper = D.DeepEPLLCombineInput
    else:
        raise ValueError("HCU INT8 DeepGEMM requires DeepEP normal or low-latency dispatch")
    return wrapper(hidden_states=output, topk_ids=dispatch_output.topk_ids, topk_weights=dispatch_output.topk_weights)
