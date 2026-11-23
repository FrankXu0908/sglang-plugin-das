"""HCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
from __future__ import annotations

import logging
from functools import lru_cache

from hcu_sglang.models.deepseek.deepseek_v2 import constants as C
from sglang.srt import utils as U
from hcu_sglang.models.deepseek.deepseek_v2.methods import moe_graph as G
import sglang.srt.models.deepseek_v2 as N0


_logger = logging.getLogger(__name__)


def __init__(
    self,
    config: N0.PretrainedConfig,
    layer_id: int,
    quant_config: N0.Optional[N0.QuantizationConfig] = None,
    prefix: str = "",
    alt_stream: N0.Optional[N0.torch.cuda.Stream] = None,
    routed_quant_stream: N0.Optional[N0.torch.cuda.Stream] = None,
    is_nextn: bool = False,
    is_deepseek_v4: bool = False,
    vl_correction_bias: bool = False,
    reduce_results: bool = True,
):
    super(N0.DeepseekV2MoE, self).__init__()
    # False when a stage boundary completes this output's sum.
    self.reduce_results = reduce_results
    self.tp_size = N0.get_parallel().tp_size
    self.moe_ep_size = N0.get_parallel().moe_ep_size
    self.routed_scaling_factor = config.routed_scaling_factor
    self.n_shared_experts = config.n_shared_experts
    self._deferred_finalize_max_tokens = (
        N0.envs.SGLANG_MOE_DEFERRED_FINALIZE_MAX_TOKENS.get()
    )

    n_shared_experts = (
        0 if config.n_shared_experts is None else int(config.n_shared_experts)
    )
    _fusion_disabled = N0.is_shared_experts_fusion_disabled()

    # num_fused_shared_experts drives weight remapping in deepseek_weight_loader:
    # mlp.shared_experts → mlp.experts.256 when > 0.
    self.num_fused_shared_experts = 0 if _fusion_disabled else n_shared_experts

    # DeepEP and MegaMOE shared expert fusion: shared expert is fused into
    # the same MoE kernel as a local expert at each EP rank. Expert layout
    # is expanded from 256 routed to 256+EP_size (e.g. 272 for EP=16).
    _uses_per_rank_shared_slots = N0.has_per_rank_fused_shared_slots(
        self.num_fused_shared_experts
    )

    if _uses_per_rank_shared_slots:
        # 256 routed + EP_size shared slots = 272 experts total (for EP=16)
        num_experts_for_moe = config.n_routed_experts + self.moe_ep_size
        top_k_for_moe = config.num_experts_per_tok + 1  # 8 routed + 1 shared
        # Interleaving for DeepEP/MegaMOE dispatch is handled by TopK internally.
    else:
        num_experts_for_moe = (
            config.n_routed_experts + self.num_fused_shared_experts
        )
        top_k_for_moe = config.num_experts_per_tok + self.num_fused_shared_experts

    self.config = config
    self.layer_id = layer_id
    from sglang.srt.configs.model_config import is_glm_moe_dsa

    if (
        C._is_hcu
        and is_glm_moe_dsa(config)
        and not N0.envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
    ):
        alt_stream = None
        routed_quant_stream = None
    self.alt_stream = alt_stream
    self.routed_quant_stream = routed_quant_stream
    self.is_nextn = is_nextn
    self.is_deepseek_v4 = is_deepseek_v4
    n_hash_layers = getattr(config, "num_hash_layers", 0)
    self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)

    if self.tp_size > config.n_routed_experts:
        raise ValueError(
            f"Tensor parallel size {self.tp_size} is greater than "
            f"the number of experts {config.n_routed_experts}."
        )

    if config.hidden_act != "silu":
        raise ValueError(
            f"Unsupported activation: {config.hidden_act}. "
            "Only silu is supported for now."
        )

    self.gate = N0.MoEGate(
        config=config,
        quant_config=quant_config,
        prefix=N0.add_prefix("gate", prefix),
        is_hash_moe=self.is_hash,
        is_deepseek_v4=is_deepseek_v4,
        vl_correction_bias=vl_correction_bias,
    )

    # scaling factor for fused shared experts on AMD-platform.
    # DeepEP/MegaMOE doesn't need this: shared expert is only computed on home rank
    # (not all-reduced), so no 1/ep_size correction is needed.
    fused_shared_experts_scaling_factor = None
    if (
        self.moe_ep_size > 1
        and self.num_fused_shared_experts > 0
        and not _uses_per_rank_shared_slots
    ):
        # if enable_ep_moe tp_szie == ep_size, every gpu get shared experts gemm output
        # so we scale with 1 / self.moe_ep_size in ep mode which will make it equalation as in tp mode
        # with fused_shared_experts
        fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

    self.experts = N0.get_moe_impl_class(quant_config)(
        num_experts=num_experts_for_moe + N0.get_exec().moe.ep_num_redundant_experts,
        num_fused_shared_experts=self.num_fused_shared_experts,
        top_k=top_k_for_moe,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        layer_id=self.layer_id,
        quant_config=quant_config,
        routed_scaling_factor=self.routed_scaling_factor,
        routing_method_type=getattr(
            config, "routing_method_type", N0.RoutingMethodType.DeepSeekV3
        ),
        swiglu_limit=getattr(config, "swiglu_limit", None),
        prefix=N0.add_prefix("experts", prefix),
    )

    if self.is_hash and not (is_nextn and is_deepseek_v4):
        self.topk = N0.HashTopK(
            topk=config.num_experts_per_tok + self.num_fused_shared_experts,
            num_experts=config.n_routed_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            vocab_size=config.vocab_size,
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            layer_id=self.layer_id,
        )
    else:
        # Default: grouped noaux_tc top-k. Covers V3/V3.2/GLM-5/Glm4MoeLite.
        topk_kwargs = dict(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
            # and requires the output format to be standard (except trtllm). We use quant_config to determine the output format.
            output_format=(
                N0.TopKOutputFormat.STANDARD
                if (quant_config is None)
                and (not N0.get_moe_runner_backend().is_flashinfer_trtllm())
                else None
            ),
        )
        # DSV4 override: ungrouped sqrtsoftplus + fp4 expert layout flag.
        if is_deepseek_v4:
            topk_kwargs.update(
                use_grouped_topk=False,
                scoring_func=config.scoring_func,
                sqrtsoftplus_log1p=(
                    getattr(config, "model_type", None) == "deepseek_v41"
                ),
                fused_gate_packed_ids=(
                    getattr(config, "model_type", None) == "deepseek_v41"
                ),
                is_fp4_experts=getattr(quant_config, "is_fp4_experts", False),
                apply_routed_scaling_factor_on_output=(
                    True
                    if N0._use_aiter
                    else self.experts.should_fuse_routed_scaling_factor_in_topk
                ),
            )
        self.topk = N0.TopK(**topk_kwargs)

    self.shared_experts_is_int8 = False
    self.shared_experts_is_fp8 = False
    self.shared_experts_weight_block_size = None
    self._shared_expert_tp1 = False
    self.mega_shared_l1_weights = None
    self.mega_shared_l2_weights = None
    # Shared experts: skip when fused into MoE kernel
    # (self.num_fused_shared_experts > 0) or when DeepEP/MegaMOE fusion is enabled.
    if (
        config.n_shared_experts is not None
        and config.n_shared_experts > 0
        and self.num_fused_shared_experts == 0
        and not _uses_per_rank_shared_slots
    ):
        intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        # Disable TP for shared experts for A2A/FP4 allgather paths, or when
        # explicitly requested for DSV4 checkpoints whose shared scales are
        # not divisible by the global TP size.
        _shared_expert_use_tp1 = (
            N0.get_moe_a2a_backend().is_deepep()
            or N0.get_moe_a2a_backend().is_pplx()
            or N0.get_moe_a2a_backend().is_mooncake()
            or N0.get_moe_a2a_backend().is_nixl()
            or N0.get_moe_a2a_backend().is_mori()
            or N0.get_moe_a2a_backend().is_ascend_fuseep()
            or N0.get_moe_a2a_backend().is_flashinfer()
            or N0.get_moe_a2a_backend().is_megamoe()
            or N0.get_moe_a2a_backend().is_flashinfer_megamoe()
            or N0.get_moe_a2a_backend().is_deepep_v2()
            or N0.should_use_flashinfer_cutlass_moe_fp4_allgather()
            or N0.envs.SGLANG_SHARED_EXPERT_TP1.get()
        )
        self.shared_experts = N0.DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            swiglu_limit=getattr(config, "swiglu_limit", None),
            prefix=N0.add_prefix("shared_experts", prefix),
            parallel_group="replicated" if _shared_expert_use_tp1 else "tp",
        )
        self.shared_experts._glm_prefill_shared_overlap = (
            C._is_hcu
            and not is_nextn
            and "GlmMoeDsaForCausalLM" in (getattr(config, "architectures", None) or ())
        )
        # Flags must be set before weight load so
        # process_weights_after_loading sees them and builds the
        # [Up, Gate]-interleaved weight + scale.
        from sglang.srt.layers.quantization.fp4_utils import (
            get_fp4_gemm_runner_backend,
        )
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptFp4LinearMethod,
        )

        fc1_n = self.shared_experts.gate_up_proj.output_size_per_partition
        if (
            N0.get_platform().is_sm100
            # The fused kernel bypasses the LoRA wrappers.
            and not (N0.get_lora().enable_lora or N0.get_lora().lora_paths)
            and isinstance(
                self.shared_experts.gate_up_proj.quant_method,
                ModelOptFp4LinearMethod,
            )
            and self.shared_experts.gate_up_proj.quant_method.quant_mode == "w4a4"
            and get_fp4_gemm_runner_backend().supports_swiglu_fusion()
            and isinstance(
                self.shared_experts.down_proj.quant_method,
                ModelOptFp4LinearMethod,
            )
            and fc1_n % 128 == 0
            and self.shared_experts.swiglu_limit is None
            and not N0.check_cuda_graph_backend(N0.Phase.PREFILL, N0.Backend.TC_PIECEWISE)
        ):
            self.shared_experts.gate_up_proj._interleave_for_swiglu_fusion = True
            self.shared_experts._enable_nvfp4_gemm_swiglu_fusion = True
            self.shared_experts.down_proj._accepts_prequantized_fp4 = True
        self._shared_expert_tp1 = _shared_expert_use_tp1
        is_packed_weight = (
            hasattr(self.shared_experts.gate_up_proj.quant_method, "quant_config")
            and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name()
            in {
                "awq",
                "awq_marlin",
                "moe_wna16",
            }
        )
        shared_gate_up_weight = getattr(
            self.shared_experts.gate_up_proj, "weight", None
        )
        if shared_gate_up_weight is None:
            shared_gate_up_weight = getattr(
                self.shared_experts.gate_up_proj, "qweight", None
            )
        if shared_gate_up_weight is None:
            raise ValueError(
                "shared expert gate/up projection has no weight storage"
            )
        self.shared_experts_is_int8 = (
            not is_packed_weight and shared_gate_up_weight.dtype == N0.torch.int8
        )
        self.shared_experts_is_fp8 = (
            not is_packed_weight
            and shared_gate_up_weight.dtype == N0.torch.float8_e4m3fn
        )
        if self.shared_experts_is_fp8 and not N0._is_npu:
            # DeepSeek-V4 passes _DeepseekV4ConfigAlias here.  Unlike the
            # generic HF config, that compatibility view intentionally has
            # no quantization_config attribute.  Identify compressed-tensors
            # from either source before accessing its block-size metadata.
            config_quantization = getattr(
                config, "quantization_config", None
            ) or getattr(config, "compression_config", None)
            is_compressed_tensors = (
                isinstance(config_quantization, dict)
                and config_quantization.get("quant_method") == "compressed-tensors"
            ) or (
                self.shared_experts.gate_up_proj.quant_method.__class__.__module__.startswith(
                    "sglang.srt.layers.quantization.compressed_tensors"
                )
            )
            if is_compressed_tensors:
                # Compressed-tensors stores the per-layer block shape on the
                # quant method/layer, not on the shared config object.
                # This path does not consume shared_experts_weight_block_size.
                pass
            else:
                # Some FP8 methods (notably the HCU w8a8_fp8 path) expose
                # their settings through ``quantization_config`` rather
                # than a legacy ``quant_config`` attribute.  The shared
                # expert block-size fast path only applies when both
                # methods actually provide block-size metadata.
                gate_up_quant_config = getattr(
                    self.shared_experts.gate_up_proj.quant_method,
                    "quant_config",
                    None,
                )
                down_proj_quant_config = getattr(
                    self.shared_experts.down_proj.quant_method,
                    "quant_config",
                    None,
                )
                gate_up_block_size = getattr(
                    gate_up_quant_config, "weight_block_size", None
                )
                down_proj_block_size = getattr(
                    down_proj_quant_config, "weight_block_size", None
                )
                if (
                    gate_up_block_size is not None
                    or down_proj_block_size is not None
                ):
                    assert gate_up_block_size == down_proj_block_size
                    self.shared_experts_weight_block_size = gate_up_block_size

    self.top_k = config.num_experts_per_tok

    if (
        N0.get_moe_a2a_backend().is_deepep()
        or N0.get_moe_a2a_backend().is_mooncake()
        or N0.get_moe_a2a_backend().is_nixl()
        or N0.get_moe_a2a_backend().is_mori()
        or N0.get_moe_a2a_backend().is_ascend_fuseep()
        or N0.get_moe_a2a_backend().is_deepep_v2()
    ):
        # TODO: we will support tp < ep in the future
        self.ep_size = N0.get_parallel().moe_ep_size
        self.num_experts = (
            config.n_routed_experts + N0.get_exec().moe.ep_num_redundant_experts
        )
        self.renormalize = config.norm_topk_prob
        self.topk_group = config.topk_group
        self.num_expert_group = config.n_group
        self.correction_bias = (
            self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None
        )

    self._enable_a2a_moe = (
        N0.get_moe_a2a_backend().is_deepep()
        or N0.get_moe_a2a_backend().is_mooncake()
        or N0.get_moe_a2a_backend().is_nixl()
        or N0.get_moe_a2a_backend().is_mori()
        or N0.get_moe_a2a_backend().is_ascend_fuseep()
        or N0.get_moe_a2a_backend().is_flashinfer()
        or N0.get_moe_a2a_backend().is_deepep_v2()
    )
    self._fuse_shared_experts_inside_sbo = N0.SboFlags.fuse_shared_experts_inside_sbo()
    # SGLANG_OPT_MOE_QUANT_ONCE eligibility, resolved lazily on first
    # forward (weights and runner are final by then). None = undecided.
    self._moe_quant_once: N0.Optional[bool] = None

    self._shared_expert_input_quant_args = None


def forward(
    self,
    hidden_states: N0.torch.Tensor,
    forward_batch: N0.Optional[N0.ForwardBatch] = None,
    gemm_output_zero_allocator: N0.BumpAllocator = None,
    input_ids: N0.Optional[N0.torch.Tensor] = None,
    input_ids_global: N0.Optional[N0.torch.Tensor] = None,
    skip_shared_experts: bool = False,
    return_moe_output: bool = False,
    leave_output_sum: bool = False,
) -> N0.Union[N0.torch.Tensor, N0.MoEOutput]:
    """``return_moe_output`` asks for the pieces unmerged, as `MoEOutput`; only
    the TP paths honor it. Distinct from `get_forward().defer_moe_finalize`,
    the V2/V3 layer-boundary handoff. ``leave_output_sum`` returns those
    pieces with the selected finalize kernel, for
    a boundary that owns reduction and the replicated shared add."""
    from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe

    if should_use_mega_moe(self, hidden_states):
        if N0.is_in_breakable_cuda_graph():
            # MegaMoE drives rank-coupled symmetric-buffer collectives whose
            # token count and expert routing change per batch, so capturing
            # it bakes in one batch's dispatch (garbled replay output). Run
            # it as an eager node, same as DeepEP NORMAL.
            return G._bcg_forward_mega_moe(
                self,
                hidden_states,
                forward_batch,
                input_ids_global,
            )
        return forward_mega_moe(
            self,
            hidden_states,
            forward_batch,
            input_ids_global=input_ids_global,
        )

    num_token_non_padded = (
        forward_batch.moe_num_token_non_padded()
        if forward_batch is not None
        else None
    )
    use_vision_topk = self.gate.e_score_correction_bias_vl is not None
    if use_vision_topk and N0._is_hip:
        use_vision_topk = N0._hip_moe.batch_has_images(forward_batch)
    if not self._enable_a2a_moe:
        if self._can_dual_stream_graph(hidden_states):
            fwd = N0.get_forward()
            if leave_output_sum:
                routed, shared = G.dsv4_flashinfer_moe_dual_stream_outputs(
                    hidden_states,
                    self.layer_id,
                    fwd.mlp_reduce_scatter,
                    self._shared_expert_tp1,
                )
                return G.MergedMoEOutput(
                    routed,
                    shared if self._shared_expert_tp1 else None,
                    self.experts,
                    1.0,
                    self._shared_expert_tp1,
                )
            return N0.dsv2_flashinfer_moe_dual_stream_graph(
                hidden_states,
                self.layer_id,
                fwd.mlp_reduce_scatter,
            )
        elif (
            self.alt_stream is not None
            and self.num_fused_shared_experts == 0
            and hidden_states.shape[0] > 0
            and N0.get_is_capture_mode()
            and not N0.is_in_breakable_cuda_graph()
            and not C._is_hcu
        ):
            return self.forward_normal_dual_stream(
                hidden_states,
                gemm_output_zero_allocator,
                input_ids,
                input_ids_global=input_ids_global,
                num_token_non_padded=num_token_non_padded,
                use_vision_topk=use_vision_topk,
                return_moe_output=return_moe_output,
                leave_output_sum=leave_output_sum,
            )
        else:
            return self.forward_normal(
                hidden_states,
                gemm_output_zero_allocator,
                input_ids,
                input_ids_global=input_ids_global,
                skip_shared_experts=skip_shared_experts,
                num_token_non_padded=num_token_non_padded,
                use_vision_topk=use_vision_topk,
                return_moe_output=return_moe_output,
                leave_output_sum=leave_output_sum,
            )
    else:
        glm_opt_prefill = N0.envs.SGLANG_GLM_OPT_PREFILL.get()
        indirect_enabled = (
            glm_opt_prefill
            or N0.envs.SGLANG_HCU_GLM53_FP8_K1_INDIRECT.get()
            or N0.envs.SGLANG_HCU_GLM53_FP8_K1_AUTO_BUILD.get()
        )
        prefill_ops_enabled = (
            glm_opt_prefill
            or N0.envs.SGLANG_HCU_GLM53_FP8_EXPERT_INVERSE.get()
            or N0.envs.SGLANG_HCU_GLM53_FP8_EP_GATHER_K8.get()
        )
        if indirect_enabled or prefill_ops_enabled:
            from contextlib import ExitStack

            with ExitStack() as scopes:
                if indirect_enabled:
                    from hcu_sglang.layers.moe.glm53_fp8_k1_indirect import forward_scope

                    scopes.enter_context(forward_scope(self, forward_batch, hidden_states))
                if prefill_ops_enabled:
                    from hcu_sglang.layers.moe.glm53_fp8_prefill_ops import prefill_scope

                    scopes.enter_context(prefill_scope(self, forward_batch, hidden_states))
                return self.forward_deepep(
                    hidden_states, forward_batch, input_ids_global=input_ids_global
                )
        return self.forward_deepep(
            hidden_states, forward_batch, input_ids_global=input_ids_global
        )


def forward_normal(
    self,
    hidden_states: N0.torch.Tensor,
    gemm_output_zero_allocator: N0.BumpAllocator = None,
    input_ids: N0.Optional[N0.torch.Tensor] = None,
    input_ids_global: N0.Optional[N0.torch.Tensor] = None,
    skip_shared_experts: bool = False,
    num_token_non_padded: N0.Optional[N0.torch.Tensor] = None,
    use_vision_topk: bool = False,
    return_moe_output: bool = False,
    leave_output_sum: bool = False,
) -> N0.Union[N0.torch.Tensor, N0.MoEOutput]:
    if hasattr(self, "shared_experts") and N0.use_intel_amx_backend(
        self.shared_experts.gate_up_proj
    ):
        return self.forward_cpu(hidden_states)
    dispatch_info = (
        N0.ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
        if N0.get_exec().moe.enable_eplb and not self.is_nextn
        else None
    )
    defer_shared = not self.experts.moe_runner_config.inplace
    # PoC (SGLANG_DP_SHARED_EXPERT_LOCAL): shared expert is computed on the LOCAL
    # hidden in the decoder layer (before the dp gather) and added after the
    # reduce_scatterv. When set, never compute/add it here (on the global buffer).
    shared_output = None
    if hidden_states.shape[0] > 0:
        # Quantize-once (SGLANG_OPT_MOE_QUANT_ONCE): only worthwhile when
        # the shared expert also runs here on the same tensor.
        pre_quant_input = (
            None
            if skip_shared_experts
            else self._maybe_quant_moe_input_once(hidden_states)
        )
        if (
            not defer_shared
            and not self._fuse_shared_experts_inside_sbo
            and not skip_shared_experts
        ):
            shared_output = self._forward_shared_experts(
                hidden_states,
                gemm_output_zero_allocator,
                pre_quant_input=pre_quant_input,
            )
        # router_logits: (num_tokens, n_experts)
        if N0._is_hip:
            router_logits, router_logits_partials = N0._hip_moe.forward_gate(
                self,
                hidden_states,
                gemm_output_zero_allocator,
                fused_gate=not use_vision_topk,
            )
        else:
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            router_logits_partials = None
        topk_kwargs = (
            {"input_ids": input_ids_global}
            if getattr(self, "is_hash", False)
            else {}
        )
        if router_logits_partials is not None:
            topk_kwargs["router_logits_partials"] = router_logits_partials
        if use_vision_topk:
            topk_output = N0.vision_topk(
                self,
                router_logits,
                input_ids_global,
                num_token_non_padded=num_token_non_padded,
            )
        else:
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
    else:
        pre_quant_input = None
        shared_output = None
        topk_output = self.topk.empty_topk_output(
            hidden_states.device, layer_id=self.layer_id
        )

    if self._fuse_shared_experts_inside_sbo and not skip_shared_experts:
        shared_output = None

        def _pre_combine_hook(
            dispatcher: N0.BaseDispatcher, combine_input: N0.CombineInput
        ):
            nonlocal shared_output
            self.alt_stream.wait_stream(N0.torch.cuda.current_stream())
            with N0.torch.cuda.stream(self.alt_stream):
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator
                )

            pre_combine_hook_handle.remove()

        def _post_combine_hook(
            dispatcher: N0.BaseDispatcher, hidden_states: N0.torch.Tensor
        ):
            nonlocal shared_output
            N0.torch.cuda.current_stream().wait_stream(self.alt_stream)
            post_combine_hook_handle.remove()

        pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
            _pre_combine_hook
        )
        post_combine_hook_handle = (
            self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
        )

    if pre_quant_input is not None:
        final_hidden_states = self.experts(
            hidden_states,
            topk_output,
            pre_quant_input=pre_quant_input,
        )
    else:
        final_hidden_states = self.experts(
            hidden_states,
            topk_output,
        )
    if (
        not N0._is_cuda
        and not N0._is_musa
        and not C._is_hcu
        and not N0._is_xpu
        and not N0._use_aiter
        or isinstance(self.experts.quant_method, N0.KTEPWrapperMethod)
    ):
        # fused in biased_grouped_topk so we can skip here
        final_hidden_states *= self.routed_scaling_factor

    if (
        defer_shared
        and hidden_states.shape[0] > 0
        and not self._fuse_shared_experts_inside_sbo
        and not skip_shared_experts
    ):
        shared_output = self._forward_shared_experts(
            hidden_states,
            gemm_output_zero_allocator,
            pre_quant_input=pre_quant_input,
        )

    if return_moe_output or leave_output_sum:
        return N0.MoEOutput(
            routed=final_hidden_states,
            shared=shared_output,
            experts=self.experts,
            routed_scaling_factor=self.routed_scaling_factor,
            shared_is_replicated=self._shared_expert_tp1,
        )

    final_hidden_states = N0.maybe_fuse_routed_scale_and_shared_add(
        self.experts,
        final_hidden_states,
        None if self._shared_expert_tp1 else shared_output,
        self.routed_scaling_factor,
    )
    if self.reduce_results:
        final_hidden_states = N0.post_experts_all_reduce(final_hidden_states)
    # TP1 shared experts are replicated: add them after this block's
    # all-reduce, or on TP rank 0 only when the stage boundary sums.
    if (
        shared_output is not None
        and self._shared_expert_tp1
        and self._adds_replicated_output()
    ):
        final_hidden_states += shared_output
    return final_hidden_states


@lru_cache(maxsize=None)
def _glm_prefill_shared_expert_stream(device):
    return N0.torch.cuda.Stream(device=device)


def _try_glm_prefill_shared_stream(moe, batch, hidden_states):
    shared_experts = getattr(moe, "shared_experts", None)
    if (
        not getattr(shared_experts, "_glm_prefill_shared_overlap", False)
        or moe._fuse_shared_experts_inside_sbo
        or not N0.get_moe_a2a_backend().is_deepep()
        or not isinstance(moe.experts.dispatcher, N0.MaybeTboDeepEPDispatcher)
        or not batch.forward_mode.is_extend_without_speculative()
        or batch.can_run_tbo
        or getattr(batch, "tbo_parent_token_range", None) is not None
    ):
        return None

    from hcu_sglang.layers.moe.deepep_fp8 import _use_hcu_legacy_channel_fp8_deepep
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch
    from sglang.srt.layers.moe.utils import DeepEPMode, get_deepep_mode
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )

    if (
        get_deepep_mode().resolve(get_is_extend_in_batch()) != DeepEPMode.NORMAL
        or not _use_hcu_legacy_channel_fp8_deepep(moe.experts)
        or is_batch_invariant_mode_enabled()
        or N0.get_exec().deterministic.enable_deterministic_inference
        or N0.torch.compiler.is_compiling()
        or N0.get_is_capture_mode()
        or N0.is_in_breakable_cuda_graph()
        or is_in_tc_piecewise_cuda_graph()
        or N0.torch.cuda.is_current_stream_capturing()
    ):
        return None

    shared_stream = moe.alt_stream
    if shared_stream is None:
        shared_stream = _glm_prefill_shared_expert_stream(hidden_states.device)

    return shared_stream


def forward_deepep(
    self,
    hidden_states: N0.torch.Tensor,
    forward_batch: N0.ForwardBatch,
    input_ids_global: N0.Optional[N0.torch.Tensor] = None,
) -> N0.torch.Tensor:
    shared_output = None
    shared_event = None
    prefill_stream = None
    prefill_dispatch_hooks = ()
    sbo_enabled_flag = self._fuse_shared_experts_inside_sbo and not self.is_nextn
    sbo_overlap_dispatch_flag = (
        sbo_enabled_flag and N0.SboFlags.enable_dispatch_shared_one_stream_overlap()
    )
    sbo_overlap_combine_flag = (
        sbo_enabled_flag and N0.SboFlags.enable_combine_shared_two_stream_overlap()
    )

    if hidden_states.shape[0] > 0:
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states, forward_batch=forward_batch)
        if not sbo_enabled_flag and self.num_fused_shared_experts == 0:
            if N0.envs.SGLANG_GLM_PREFILL_SHARED_OVERLAP.get():
                prefill_stream = _try_glm_prefill_shared_stream(
                    self, forward_batch, hidden_states
                )
            if prefill_stream is None and self.alt_stream is not None:
                self.alt_stream.wait_stream(N0.torch.cuda.current_stream())
                with N0.torch.cuda.stream(self.alt_stream):
                    shared_output = self._forward_shared_experts(hidden_states)
                    shared_output.record_stream(self.alt_stream)
                    shared_event = self.alt_stream.record_event()
                if N0.is_in_breakable_cuda_graph():
                    # The MoE call below is an eager break, so record
                    # and wait must share one capture; joining here means
                    # the shared experts overlap nothing. The alt stream
                    # is kept for record_stream: without that marking the
                    # allocator recycles shared_output across the break.
                    N0.torch.cuda.current_stream().wait_event(shared_event)
            elif prefill_stream is None:
                shared_output = self._forward_shared_experts(hidden_states)
        topk_kwargs = (
            {"input_ids": input_ids_global}
            if getattr(self, "is_hash", False)
            else {}
        )
        if self.gate.e_score_correction_bias_vl is not None:
            topk_output = N0.vision_topk(
                self,
                router_logits,
                input_ids_global,
                num_token_non_padded=forward_batch.moe_num_token_non_padded(),
            )
        else:
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.moe_num_token_non_padded(),
                expert_location_dispatch_info=(
                    N0.ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    )
                    if not self.is_nextn
                    else None
                ),
                **topk_kwargs,
            )
    else:
        topk_output = self.topk.empty_topk_output(
            hidden_states.device, layer_id=self.layer_id
        )

    if sbo_overlap_dispatch_flag:
        shared_output = None

        def _deepep_dispatch_hook(dispatcher: N0.BaseDispatcher):
            nonlocal shared_output
            shared_output = self._forward_shared_experts(hidden_states)
            for handle in deepep_dispatch_hook_handle:
                handle.remove()

        def _post_dispatch_hook(
            dispatcher: N0.BaseDispatcher, dispatch_output: N0.DispatchOutput
        ):
            combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                N0.compute_overlap_args(dispatch_output, self.alt_stream)
            )
            dispatcher.set_overlap_args(
                combine_overlap_args=combine_overlap_args,
                meta_overlap_args=meta_overlap_args,
            )
            self.experts.set_overlap_args(
                down_gemm_overlap_args=down_gemm_overlap_args,
                meta_overlap_args=meta_overlap_args,
            )
            post_dispatch_hook_handle.remove()

        def _post_combine_hook(
            dispatcher: N0.BaseDispatcher, hidden_states: N0.torch.Tensor
        ):
            dispatcher.clear_overlap_args()
            self.experts.clear_overlap_args()
            post_combine_hook_handle.remove()

        assert isinstance(self.experts.dispatcher, N0.MaybeTboDeepEPDispatcher)
        deepep_dispatch_hook_handle = (
            self.experts.dispatcher.register_deepep_dispatch_hook(
                _deepep_dispatch_hook
            )
        )
        post_dispatch_hook_handle = (
            self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
        )
        post_combine_hook_handle = (
            self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
        )

    elif sbo_overlap_combine_flag:
        shared_output = None

        def _post_dispatch_hook(
            dispatcher: N0.BaseDispatcher, dispatch_output: N0.DispatchOutput
        ):
            combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                N0.compute_overlap_args(dispatch_output, self.alt_stream)
            )
            dispatcher.set_overlap_args(
                combine_overlap_args=combine_overlap_args,
                meta_overlap_args=meta_overlap_args,
            )
            self.experts.set_overlap_args(
                down_gemm_overlap_args=down_gemm_overlap_args,
                meta_overlap_args=meta_overlap_args,
            )

            post_dispatch_hook_handle.remove()

        def _pre_combine_hook(
            dispatcher: N0.BaseDispatcher, combine_input: N0.CombineInput
        ):
            nonlocal shared_output

            if (
                e := dispatcher.meta_overlap_args.get("record_event_after_down")
            ) is not None:
                e.record()

            # TODO reduce sm for non-deepgemm
            with N0.deep_gemm_wrapper.configure_deep_gemm_num_sms(
                dispatcher.meta_overlap_args["compute_num_sms"]
            ):
                shared_output = self._forward_shared_experts(hidden_states)

            pre_combine_hook_handle.remove()

        def _post_combine_hook(
            dispatcher: N0.BaseDispatcher, hidden_states: N0.torch.Tensor
        ):
            dispatcher.clear_overlap_args()
            self.experts.clear_overlap_args()
            post_combine_hook_handle.remove()

        post_dispatch_hook_handle = (
            self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
        )
        pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
            _pre_combine_hook
        )
        post_combine_hook_handle = (
            self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
        )
    elif N0.envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.get():
        # On GB200: Shared experts overlapped on alt_stream, down gemm overlapped with DeepEP Combine

        def _post_dispatch_hook(
            dispatcher: N0.BaseDispatcher, dispatch_output: N0.DispatchOutput
        ):
            combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                N0.compute_overlap_args(dispatch_output, self.alt_stream)
            )
            dispatcher.set_overlap_args(
                combine_overlap_args=combine_overlap_args,
                meta_overlap_args=meta_overlap_args,
            )
            self.experts.set_overlap_args(
                down_gemm_overlap_args=down_gemm_overlap_args,
                meta_overlap_args=meta_overlap_args,
            )

            post_dispatch_hook_handle.remove()

        def _pre_combine_hook(
            dispatcher: N0.BaseDispatcher, combine_input: N0.CombineInput
        ):
            if (
                e := dispatcher.meta_overlap_args.get("record_event_after_down")
            ) is not None:
                e.record()
            pre_combine_hook_handle.remove()

        def _post_combine_hook(
            dispatcher: N0.BaseDispatcher, hidden_states: N0.torch.Tensor
        ):
            dispatcher.clear_overlap_args()
            self.experts.clear_overlap_args()
            post_combine_hook_handle.remove()

        post_dispatch_hook_handle = (
            self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
        )
        pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
            _pre_combine_hook
        )
        post_combine_hook_handle = (
            self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
        )

    if prefill_stream is not None:

        def _launch_prefill_shared(_dispatcher):
            nonlocal shared_output, shared_event
            quant_ready = N0.torch.cuda.current_stream().record_event()
            prefill_stream.wait_event(quant_ready)
            with N0.torch.cuda.stream(prefill_stream):
                shared_output = self._forward_shared_experts(hidden_states)
                hidden_states.record_stream(prefill_stream)
                shared_output.record_stream(prefill_stream)
                shared_event = prefill_stream.record_event()
            if (
                self.layer_id == getattr(self.config, "first_k_dense_replace", 0)
                and not getattr(self, "_glm_shared_overlap_reported", False)
            ):
                _logger.info("GLM prefill selected shared expert overlap")
                self._glm_shared_overlap_reported = True

        prefill_dispatch_hooks = self.experts.dispatcher.register_deepep_dispatch_hook(
            _launch_prefill_shared
        )
    try:
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
    finally:
        for handle in prefill_dispatch_hooks:
            handle.remove()
    if prefill_stream is not None and shared_event is None:
        shared_output = self._forward_shared_experts(hidden_states)

    if shared_event is not None and not N0.is_in_breakable_cuda_graph():
        current_stream = N0.torch.cuda.current_stream()
        current_stream.wait_event(shared_event)
        if prefill_stream is not None:
            shared_output.record_stream(current_stream)

    if shared_output is not None:
        x = shared_output
        # aiter moe call will handle routed_scaling_factor in the function
        # so add _use_aiter condition to eliminate to use self.routed_scaling_factor in add_ call
        if self.experts.should_fuse_routed_scaling_factor_in_topk or N0._use_aiter:
            x.add_(final_hidden_states)
        else:
            x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
        final_hidden_states = x
    else:
        if not (
            self.experts.should_fuse_routed_scaling_factor_in_topk or N0._use_aiter
        ):
            final_hidden_states *= self.routed_scaling_factor

    return final_hidden_states


def forward_normal_dual_stream(
    self,
    hidden_states: N0.torch.Tensor,
    gemm_output_zero_allocator: N0.BumpAllocator = None,
    input_ids: N0.Optional[N0.torch.Tensor] = None,
    input_ids_global: N0.Optional[N0.torch.Tensor] = None,
    num_token_non_padded: N0.Optional[N0.torch.Tensor] = None,
    use_vision_topk: bool = False,
    return_moe_output: bool = False,
    leave_output_sum: bool = False,
) -> N0.Union[N0.torch.Tensor, N0.MoEOutput]:
    # Note(kpham-sgl): issue order satisfies 3 constraints:
    # - no stream explosion: main (routed) issued before alt block -> capture reuses 1 alt stream;
    # - PDL overlap: routed is the last main-stream kernel (fuses w/ residual add);
    # - dispose_tensor: disabled during capture (CaptureFlags.disable_dispose_tensor) so the routed
    #   deep_gemm does not free hidden_states, which the shared expert reads on the alt stream.
    use_flashinfer_trtllm_bypass = N0.get_forward().flashinfer_trtllm_bypass
    current_stream = N0.torch.cuda.current_stream()
    # Quantize-once (SGLANG_OPT_MOE_QUANT_ONCE) must happen on the main
    # stream BEFORE the alt-stream fork so both consumers see it.
    pre_quant_input = (
        None
        if use_flashinfer_trtllm_bypass
        else self._maybe_quant_moe_input_once(hidden_states)
    )
    self.alt_stream.wait_stream(current_stream)
    should_quant_routed_input_mxfp8 = (
        not use_flashinfer_trtllm_bypass
        and pre_quant_input is None
        and self._should_quant_routed_input_mxfp8(hidden_states)
    )
    if should_quant_routed_input_mxfp8:
        self.routed_quant_stream.wait_stream(current_stream)
    has_shared_output = (
        hidden_states.shape[0] > 0 and self.num_fused_shared_experts == 0
    )
    dispatch_info = (
        N0.ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
        if N0.get_exec().moe.enable_eplb and not self.is_nextn
        else None
    )

    # router_logits: (num_tokens, n_experts)
    if N0._is_hip:
        router_logits, router_logits_partials = N0._hip_moe.forward_gate(
            self,
            hidden_states,
            gemm_output_zero_allocator,
            fused_gate=not use_flashinfer_trtllm_bypass and not use_vision_topk,
        )
    else:
        router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
        router_logits_partials = None
    if use_flashinfer_trtllm_bypass:
        topk_output = N0.BypassedTopKOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=self.topk.topk_config,
        )
    else:
        topk_kwargs = (
            {"input_ids": input_ids_global}
            if getattr(self, "is_hash", False)
            else {}
        )
        if router_logits_partials is not None:
            topk_kwargs["router_logits_partials"] = router_logits_partials
        if use_vision_topk:
            topk_output = N0.vision_topk(
                self,
                router_logits,
                input_ids_global,
                num_token_non_padded=num_token_non_padded,
            )
        else:
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
    # Issued after the router so the main chain stays on the main stream at replay.
    routed_pre_quant_input = pre_quant_input
    if should_quant_routed_input_mxfp8:
        with N0.torch.cuda.stream(self.routed_quant_stream):
            x_q, x_sf = self.experts.quant_method.quantize_routed_input(
                hidden_states, N0.routed_hidden_size(self.experts)
            )
            ready = self.routed_quant_stream.record_event()
        routed_pre_quant_input = N0.Mxfp8RoutedInputPreQuant(x_q, x_sf, ready)
    deferred_finalize = (
        has_shared_output
        and not self._shared_expert_tp1
        and topk_output.format == N0.TopKOutputFormat.BYPASSED
        and self.experts.supports_deferred_finalize
        and (
            self._deferred_finalize_max_tokens <= 0
            or hidden_states.shape[0] <= self._deferred_finalize_max_tokens
        )
    )
    # NOTE: if `return_moe_output`, always try to defer finalize
    deferred_finalize = deferred_finalize or (
        return_moe_output
        and isinstance(self.experts.quant_method, N0.Mxfp4FlashinferTrtllmMoEMethod)
    )
    if deferred_finalize:
        final_hidden_states = self.experts.forward_deferred_finalize(
            hidden_states, topk_output, pre_quant_input=routed_pre_quant_input
        )
    elif use_flashinfer_trtllm_bypass:
        final_hidden_states = self.experts.forward_impl(hidden_states, topk_output)
    elif routed_pre_quant_input is not None:
        final_hidden_states = self.experts(
            hidden_states, topk_output, pre_quant_input=routed_pre_quant_input
        )
    else:
        final_hidden_states = self.experts(hidden_states, topk_output)
    if (
        not N0._is_cuda
        and not N0._is_musa
        and not N0._use_aiter
        and not getattr(self.experts.quant_method, "scales_routed_output", False)
        or isinstance(self.experts.quant_method, N0.KTEPWrapperMethod)
    ):
        final_hidden_states *= self.routed_scaling_factor

    # Shared expert on alt stream, issued AFTER the main (routed) branch. See note above.
    # Only the quant-once fp8 pair is shared with it; the routed MXFP8 pre-quant is not.
    with N0.torch.cuda.stream(self.alt_stream):
        shared_output = self._forward_shared_experts(
            hidden_states,
            gemm_output_zero_allocator,
            pre_quant_input=pre_quant_input,
        )

    # The routed-input pre-quant was already joined inside the routed MoE apply.
    current_stream.wait_stream(self.alt_stream)

    if deferred_finalize and N0.get_forward().defer_moe_finalize:
        # deferred_finalize excludes _shared_expert_tp1, so the shared add folds in.
        assert shared_output is not None
        from sglang.srt.layers.layer_boundary.fusions.cutedsl import (
            MoeDeferredFinalize,
        )

        return MoeDeferredFinalize.from_flashinfer(
            final_hidden_states,
            gated_shared_output=shared_output,
            m=hidden_states.shape[0],
            reduce=N0.sum_post_experts_output,
        )

    if return_moe_output or leave_output_sum:
        return N0.MoEOutput(
            routed=final_hidden_states,
            shared=shared_output,
            experts=self.experts,
            routed_scaling_factor=self.routed_scaling_factor,
            shared_is_replicated=self._shared_expert_tp1,
        )

    if deferred_finalize:
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            finalize_flashinfer_trtllm_deferred_output,
        )

        final_hidden_states = finalize_flashinfer_trtllm_deferred_output(
            final_hidden_states,
            shared_output,
        )
    else:
        final_hidden_states = N0.maybe_fuse_routed_scale_and_shared_add(
            self.experts,
            final_hidden_states,
            None if self._shared_expert_tp1 else shared_output,
            self.routed_scaling_factor,
        )
    if self.reduce_results:
        final_hidden_states = N0.post_experts_all_reduce(final_hidden_states)
    # TP1 shared experts are replicated: add them after this block's
    # all-reduce, or on TP rank 0 only when the stage boundary sums.
    if self._shared_expert_tp1 and self._adds_replicated_output():
        final_hidden_states += shared_output
    return final_hidden_states


def _forward_shared_experts(
    self,
    hidden_states,
    gemm_output_zero_allocator: N0.BumpAllocator = None,
    pre_quant_input: N0.Optional[N0.Tuple[N0.torch.Tensor, N0.torch.Tensor]] = None,
):
    if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
        if pre_quant_input is None and N0._is_hip:
            # SGLANG_HIP_FFN_NORM_MXFP8: the FFN norm launch's fp8 + ue8m0 of these rows
            pre_quant_input = getattr(hidden_states, "_hip_mxfp8_operand", None)
        if pre_quant_input is not None:
            # SGLANG_OPT_MOE_QUANT_ONCE: (q, s) rows may be padded to a
            # multiple of 4; the padded rows flow through the MLP (all ops
            # are row-local) and are sliced off here.
            out = self.shared_experts(
                hidden_states, gateup_pre_quant=pre_quant_input
            )
            return out[: hidden_states.shape[0]]
        return self.shared_experts(
            hidden_states,
            gemm_output_zero_allocator=gemm_output_zero_allocator,
            input_quant_args=self._shared_expert_input_quant_args,
        )
    else:
        return None
