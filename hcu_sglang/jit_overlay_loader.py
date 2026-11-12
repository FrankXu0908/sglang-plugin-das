"""Redirect HCU JIT builds to plugin-owned fork-compatible sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from hcu_sglang.utils.detection import is_hcu


_OVERLAY_ROOT = Path(__file__).with_name("jit_overlay")


def _overlay_sources(files: Optional[Iterable[str]]) -> Optional[list[str]]:
    if files is None:
        return None
    rewritten = []
    for value in files:
        path = Path(value)
        overlay = _OVERLAY_ROOT / "csrc" / path
        rewritten.append(str(overlay) if not path.is_absolute() and overlay.is_file() else value)
    return rewritten


def around_load_jit(original: Callable[..., Any], *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)

    kwargs["cpp_files"] = _overlay_sources(kwargs.get("cpp_files"))
    kwargs["cuda_files"] = _overlay_sources(kwargs.get("cuda_files"))
    include_paths = list(kwargs.get("extra_include_paths") or [])
    overlay_include = str(_OVERLAY_ROOT / "include")
    if overlay_include not in include_paths:
        include_paths.insert(0, overlay_include)
    kwargs["extra_include_paths"] = include_paths
    files = (
        *(kwargs["cpp_files"] or ()),
        *(kwargs["cuda_files"] or ()),
    )
    if any(Path(file).is_relative_to(_OVERLAY_ROOT / "csrc") for file in files):
        flag = f"-I{overlay_include}"
        for key in ("extra_cflags", "extra_cuda_cflags"):
            kwargs[key] = [flag] + [
                value for value in (kwargs.get(key) or ()) if value != flag
            ]
    return original(*args, **kwargs)


def prefer_overlay_headers() -> None:
    """Put HCU headers ahead of the official root before BuildSpec is frozen.

    load_jit prepends DEFAULT_INCLUDE to extra_include_paths. Updating the shared
    default list at plugin registration keeps the real compiler order and build
    cache fingerprint consistent, without changing input files or toolchains.
    """
    if not is_hcu():
        return
    from sglang.kernels.jit.utils.compile.paths import DEFAULT_INCLUDE
    overlay_include = str(_OVERLAY_ROOT / "include")
    DEFAULT_INCLUDE[:] = [overlay_include] + [path for path in DEFAULT_INCLUDE if path != overlay_include]


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    prefer_overlay_headers()
    HookRegistry.register(
        "sglang.kernels.jit.utils.compile.loader.load_jit",
        around_load_jit,
        HookType.AROUND,
    )
