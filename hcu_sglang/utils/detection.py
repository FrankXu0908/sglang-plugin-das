"""HCU hardware detection (plugin-internal)."""

from __future__ import annotations

import logging
from functools import lru_cache

import torch


logger = logging.getLogger(__name__)

_SUPPORTED_ARCHS = ("gfx936", "gfx938", "gfx928")
_FP8_ARCHS = ("gfx938",)


@lru_cache(maxsize=1)
def is_hcu() -> bool:
    if getattr(torch.version, "hip", None) is None:
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        if not gcn_arch:
            raise RuntimeError("HIP returned an empty gcnArchName for device 0")
        return any(gfx in gcn_arch for gfx in _SUPPORTED_ARCHS)
    except Exception as e:
        if torch.cuda.is_available():
            raise RuntimeError("Cannot determine the architecture of visible HIP device 0") from e
        logger.warning("HCU detection failed: %s", e)
        return False


@lru_cache(maxsize=1)
def is_hcu_native_fp8_supported() -> bool:
    if not is_hcu() or not torch.version.hip:
        return False
    gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
    return any(gfx in gcn_arch for gfx in _FP8_ARCHS)


def get_gcn_arch(device_id: int = 0) -> str:
    try:
        props = torch.cuda.get_device_properties(device_id)
        return getattr(props, "gcnArchName", "")
    except Exception:
        return ""
