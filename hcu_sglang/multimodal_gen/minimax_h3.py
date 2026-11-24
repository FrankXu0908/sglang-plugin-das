"""MiniMax-H3 HCU backend selection and QK-Norm compatibility."""

from __future__ import annotations

import logging
from typing import Any, Callable

from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)


def after_adjust_attention_backend(result: Any, server_args: Any) -> Any:
    if not is_hcu() or server_args.attention_backend != "fa":
        return result

    from sglang.multimodal_gen.runtime.server_args.server_args import Backend

    if (
        server_args.backend == Backend.DIFFUSERS
        or type(server_args.pipeline_config).__name__ != "MiniMaxH3PipelineConfig"
    ):
        return result

    defaults = {"text_encoder": "torch_sdpa", "transformer": "fa"}
    for component, backend in defaults.items():
        if component not in server_args.component_attention_backends:
            server_args.component_attention_backends[component] = backend
            logger.info(
                "Automatically set MiniMax-H3 HCU %s attention backend to %s",
                component,
                backend,
            )
    return result


def around_apply_qk_norm(
    original: Callable[..., Any], q, k, q_norm, k_norm, head_dim: int
):
    if not is_hcu():
        return original(q, k, q_norm, k_norm, head_dim)
    return q_norm(q), k_norm(k)


def after_post_load_weights(result: Any, model: Any) -> Any:
    if is_hcu():
        model._resolve_attention_backend_once()
    return result


def around_apply_rope_qk(original, q, k, cos_sin_cache, positions):
    if not is_hcu() or not q.is_cuda:
        return original(q, k, cos_sin_cache, positions)
    import torch
    rotary = getattr(torch.ops.sgl_kernel, "minimax_h3_rotary_embedding", None)
    if rotary is None:
        raise RuntimeError(
            "MiniMax-H3 on HCU requires its dedicated RoPE operator; "
            "rebuild sglang-kernel with setup_hip.py"
        )
    rotary(positions, q.view(q.shape[0], -1), k.view(k.shape[0], -1),
           q.shape[-1], cos_sin_cache, True)
    return q, k


def around_denoising_full_loop(original, stage, batch, server_args):
    if not is_hcu():
        return original(stage, batch, server_args)

    from sglang.multimodal_gen.runtime.layers.usp import _A2A_STAGING_BUFFERS
    from sglang.multimodal_gen.runtime.platforms import current_platform

    try:
        return original(stage, batch, server_args)
    finally:
        if _A2A_STAGING_BUFFERS:
            import torch

            device = current_platform.get_local_torch_device()
            if device.type == "cuda":
                index = device.index if device.index is not None else torch.cuda.current_device()
                for key in tuple(_A2A_STAGING_BUFFERS):
                    if key[2] == index:
                        _A2A_STAGING_BUFFERS.pop(key, None)


def register_hooks(registry=None) -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if registry is not None:
        HookRegistry = registry

    from hcu_sglang.multimodal_gen import minimax_h3_vae

    HookRegistry.register(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "minimax_h3.stages.denoising.MiniMaxH3DenoisingStage._run_full_loop",
        around_denoising_full_loop,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.processor."
        "VAEProcessor.convert_numpy_to_tensor",
        minimax_h3_vae.around_convert_numpy_to_tensor,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae."
        "AutoencoderKL.setup_forward",
        minimax_h3_vae.around_setup_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_rope_qk",
        around_apply_rope_qk,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.server_args.server_args."
        "ServerArgs._adjust_attention_backend",
        after_adjust_attention_backend,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_qk_norm",
        around_apply_qk_norm,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
        "MiniMaxH3DiTModel.post_load_weights",
        after_post_load_weights,
        HookType.AFTER,
    )
