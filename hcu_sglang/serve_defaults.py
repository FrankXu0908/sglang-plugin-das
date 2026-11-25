# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

_QWEN4_W4A8_DEFAULTS = {
    "AITER_AR_MAX_SIZE_MB": "64",
    "SGLANG_HCU_QSA_PREFILL_KEY_SPAN": "4096",
    "SGLANG_HCU_QSA_PREFILL_LOGITS_MIB": "512",
    "SGLANG_QWEN4_PREFILL_QSA_IMPL": "native-paged",
    "SGLANG_HCU_QWEN_SILU_INT8_FUSION": "1",
    "SGLANG_QWEN4_SPLIT_HC_MIX": "1",
    "SGLANG_QWEN4_FLAT_STATE_SCATTER": "1",
    "SGLANG_USE_LIGHTOP": "1",
    "SGLANG_USE_FUSED_TOPK_SOFTMAX": "1",
    "SGLANG_USE_CAUSAL_CONV1D": "1",
    "SGLANG_USE_AITER_LINEAR_ATTN": "1",
    "SGLANG_USE_AITER_AR": "1",
    "W8A8_SUPPORT_METHODS": "3",
    "SGLANG_W4A8_TPMOE_BACKEND": "aiter",
    "SGLANG_FORCE_AITER_MOE_C": "1",
    "SGLANG_ROCM_USE_AITER_MOE": "0",
    "SGLANG_DP_USE_GATHERV": "1",
    "SGLANG_MOE_C_NATIVE_SORT": "1",
    "SGLANG_QWEN4_COMPACT_PREFILL": "1",
    "SGLANG_QWEN4_LOCAL_ROUTING": "1",
    "SGLANG_QWEN4_ROUTED_INT8": "1",
    "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    "SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK": "1",
}


def _option(args, name, default=None):
    value = default
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def apply_qwen4_serve_defaults(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    model_path = _option(args, "--model-path")
    if not model_path and args[:1] == ["serve"] and len(args) > 1 and not args[1].startswith("-"):
        model_path = args[1]
    if not model_path:
        return {}
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(config, dict) or config.get("model_type") not in ("qwen4_exp", "qwen4_exp_text"):
        return {}
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        return {}
    compression = config.get("compression_config") or text.get("compression_config") or {}
    if not isinstance(compression, dict):
        return {}
    quant = _option(args, "--quantization", compression.get("quant_method"))
    dtype = _option(args, "--dtype", "auto")
    if dtype == "auto":
        dtype = text.get("dtype", text.get("torch_dtype"))
    if quant not in ("slimquant_w4a8", "slimquant_w4a8_marlin") or dtype != "bfloat16":
        return {}
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return {}
    defaults = dict(_QWEN4_W4A8_DEFAULTS)
    if _option(args, "--disaggregation-mode") == "prefill":
        defaults["SGLANG_HCU_PD_PREFILL_DRAIN_MODE"] = "wait"
        if "--enable-prefill-delayer" in args and _option(args, "--prefill-delayer-queue-min-ratio") is not None:
            if float(_option(args, "--prefill-delayer-queue-min-ratio")) == 0:
                defaults["SGLANG_HCU_PREFILL_MIXED_TIMEOUT"] = "1"
    applied = {}
    for name, value in defaults.items():
        if name not in os.environ:
            os.environ.setdefault(name, value)
            applied[name] = value
    logger.info(
        "HCU Qwen4 W4A8 serve settings: %s",
        {name: (os.environ[name], "default" if name in applied else "environment")
         for name in defaults},
    )
    return applied
