# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sglang.srt.arg_groups.speculative_hook as N

def _is_supported_dspark_pd_prefill_cp(server_args: ServerArgs) -> bool:
    model_arch = N.model_config_of(server_args).hf_config.architectures[0]
    attn_tp_size = (
        server_args.tp_size // server_args.dp_size // server_args.attn_cp_size
    )
    return (
        server_args.disaggregation_mode == "prefill"
        and server_args.disaggregation_transfer_backend == "mooncake"
        and server_args.attn_cp_size > 1
        and attn_tp_size == 1
        and server_args.enable_prefill_cp
        and server_args.cp_strategy == "interleave"
        and model_arch == "DeepseekV4ForCausalLM"
    )

def _handle_dspark(server_args: ServerArgs) -> None:
    # HCU uses the HIP PyTorch runtime, so the resolved device is not prefixed
    # with ``cuda``. Keep all other platforms rejected until they add a tested
    # backend, but allow the HCU implementation to reach its static-verify
    # compatibility path.
    cfg = N.resolving_view(server_args)
    _is_npu = cfg.device.startswith("npu")
    from hcu_sglang.utils.detection import is_hcu as is_hcu

    _is_hcu_device = is_hcu()
    if not cfg.device.startswith(("cuda", "npu", "xpu")):
        if not _is_hcu_device:
            raise ValueError(
                "DSpark speculative decoding only supports CUDA, NPU, XPU and HCU devices."
            )

    pd_prefill_cp = _is_supported_dspark_pd_prefill_cp(server_args)
    if cfg.attn_cp_size > 1 and not pd_prefill_cp:
        raise ValueError(
            "DSpark context parallel is only supported for DeepSeek-V4 PD prefill "
            "with Mooncake, attn_tp_size == 1, and interleave CP; "
            f"got disaggregation_mode={cfg.disaggregation_mode!r}, "
            f"pp_size={cfg.pp_size}, attn_cp_size={cfg.attn_cp_size}, "
            f"cp_strategy={cfg.cp_strategy!r}, "
            f"transfer_backend={cfg.disaggregation_transfer_backend!r}."
        )

    if cfg.attn_dp_size > 1 and not pd_prefill_cp:
        if not cfg.enable_dp_lm_head:
            raise ValueError("DSpark with dp attention requires --enable-dp-lm-head.")
        supports_target_moe = cfg.moe_a2a_backend in (
            "none",
            "megamoe",
            "mori",
            "mori-das",
        ) or (
            cfg.moe_a2a_backend == "deepep"
            # HCU serves DeepEP with its own runners (marlin/aiter/...).
            and (_is_hcu_device or cfg.moe_runner_backend == "deep_gemm")
        )
        if not _is_npu and not supports_target_moe:
            raise ValueError(
                "DSpark with dp attention supports MoE A2A 'none', 'megamoe', "
                "'mori', 'mori-das', or 'deepep' with runner 'deep_gemm'; got "
                f"a2a={cfg.moe_a2a_backend!r}, runner={cfg.moe_runner_backend!r}."
            )
        if not _is_npu and (
            cfg.moe_a2a_backend != "none"
            or N.envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        ):
            from sglang.srt.speculative.ragged_verify import (
                RaggedVerifyMode,
                read_ragged_verify_mode,
            )

            if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
                raise ValueError(
                    "DSpark DP MoE or prefill coordination requires "
                    "SGLANG_RAGGED_VERIFY_MODE=static."
                )
        draft_a2a = (
            cfg.speculative_moe_a2a_backend
            if cfg.speculative_moe_a2a_backend is not None
            else cfg.moe_a2a_backend
        )
        draft_runner = (
            cfg.speculative_moe_runner_backend
            if cfg.speculative_moe_runner_backend is not None
            else cfg.moe_runner_backend
        )
        supports_draft_moe = draft_a2a in (
            "none",
            "megamoe",
            "mori",
            "mori-das",
        ) or (
            draft_a2a == "deepep" and (_is_hcu_device or draft_runner == "deep_gemm")
        )
        if not supports_draft_moe:
            raise ValueError(
                "DSpark draft MoE A2A only supports 'none', 'megamoe', 'mori', "
                "'mori-das', "
                "or 'deepep' with runner 'deep_gemm'; got "
                f"a2a={draft_a2a!r}, runner={draft_runner!r}."
            )

    if cfg.pp_size != 1:
        if cfg.disaggregation_mode != "prefill":
            raise ValueError(
                "DSpark pipeline parallelism requires PD prefill; "
                "decode and non-disaggregated serving require pp_size == 1."
            )

    if cfg.speculative_draft_model_path is None:
        if N._target_checkpoint_bundles_dspark_draft(server_args):
            N.declare_resolution(
                server_args,
                "_handle_dspark",
                speculative_draft_model_path=cfg.model_path,
            )
            N.declare_resolution(
                server_args,
                "_handle_dspark",
                speculative_draft_model_revision=cfg.revision,
            )
            N.logger.info(
                "DSpark draft weights are bundled in the target checkpoint; "
                "defaulting --speculative-draft-model-path to --model-path (%s).",
                cfg.model_path,
            )
        else:
            raise ValueError(
                "DSpark dense speculative decoding requires setting "
                "--speculative-draft-model-path."
            )

    if cfg.speculative_num_steps is None:
        N.declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_num_steps=1,
        )
    elif int(cfg.speculative_num_steps) != 1:
        N.logger.warning(
            "DSpark only supports speculative_num_steps == 1; overriding speculative_num_steps=%s to 1.",
            cfg.speculative_num_steps,
        )
        N.declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_num_steps=1,
        )

    if cfg.speculative_eagle_topk is None:
        N.declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_eagle_topk=1,
        )
    elif int(cfg.speculative_eagle_topk) != 1:
        N.logger.warning(
            "DSpark only supports speculative_eagle_topk == 1; overriding speculative_eagle_topk=%s to 1.",
            cfg.speculative_eagle_topk,
        )
        N.declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_eagle_topk=1,
        )

    from sglang.srt.speculative.dspark_components.dspark_config import (
        DEFAULT_DSPARK_GAMMA,
        read_draft_checkpoint_config,
    )

    draft_config = None
    try:
        draft_config = read_draft_checkpoint_config(server_args=server_args)
    except Exception as e:
        N.logger.warning(
            "Failed to read DSpark draft config; preserving explicit/default "
            "gamma resolution. Error: %s",
            e,
        )

    gamma: N.Optional[int] = None
    if cfg.speculative_dspark_block_size is not None:
        if int(cfg.speculative_dspark_block_size) <= 0:
            raise ValueError(
                "DSpark requires --speculative-dspark-block-size to be positive, "
                f"got {cfg.speculative_dspark_block_size}."
            )
        gamma = int(cfg.speculative_dspark_block_size)
    else:
        if draft_config is not None:
            gamma = draft_config.resolve_gamma(default=None)
        if gamma is None and cfg.speculative_num_draft_tokens is None:
            gamma = DEFAULT_DSPARK_GAMMA
            N.logger.warning(
                "DSpark gamma is not set; defaulting to %d.",
                gamma,
            )

    if gamma is not None:
        verify_window = int(gamma) + 1
        if (
            cfg.speculative_num_draft_tokens is not None
            and int(cfg.speculative_num_draft_tokens) != verify_window
        ):
            raise ValueError(
                "DSpark speculative_num_draft_tokens must equal gamma + 1 "
                f"(= {verify_window} for gamma={gamma}), but got "
                f"speculative_num_draft_tokens={cfg.speculative_num_draft_tokens}."
            )
        N.declare_resolution(
            server_args,
            "_handle_dspark",
            speculative_num_draft_tokens=verify_window,
        )

    if cfg.speculative_num_draft_tokens is None:
        raise ValueError(
            "DSpark could not resolve speculative_num_draft_tokens; set "
            "--speculative-dspark-block-size (= gamma)."
        )
    if int(cfg.speculative_num_draft_tokens) < 2:
        raise ValueError(
            "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1), "
            f"got {cfg.speculative_num_draft_tokens}."
        )

    if cfg.max_running_requests is None:
        N.declare_resolution(
            server_args,
            "_handle_dspark",
            max_running_requests=48,
        )
        N.logger.warning(
            "Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests."
        )

    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    ragged_mode = read_ragged_verify_mode()
    if (
        cfg.speculative_dspark_align_verify_tokens_to_graph_tier
        and ragged_mode is not RaggedVerifyMode.COMPACT
    ):
        N.logger.warning(
            "--speculative-dspark-align-verify-tokens-to-graph-tier only takes "
            "effect with SGLANG_RAGGED_VERIFY_MODE=compact (got %r); it will be "
            "a no-op.",
            ragged_mode.value,
        )
    if cfg.speculative_dspark_sps_table_path and ragged_mode is RaggedVerifyMode.STATIC:
        N.logger.warning(
            "--speculative-dspark-sps-table-path feeds the ragged-verify budget "
            "scheduler, which is off under SGLANG_RAGGED_VERIFY_MODE=static; it "
            "will be a no-op."
        )


def register_hooks():
    from hcu_sglang import speculative_dspark_worker
    speculative_dspark_worker.register_hooks()
