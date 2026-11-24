#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Replay the 2026-11-09 startup hook targets against the real HookRegistry.

Registers exactly the 15 targets that failed during that PD run (plus one
resolvable control) and compares unguarded vs guarded ``apply_hooks()``.
Which of the 15 a build actually lacks varies (a build with the multimodal
pipelines resolves 14 of them), so the outcome is read target by target:

* unguarded: one ERROR record (with traceback) per target the build lacks;
* guarded:   zero ERROR records, summary lines naming exactly those skipped
             targets, and the control target still patched.

Usage:
    python tests/runtime/hook_guard_replay.py --sglang-python <path/to/sglang/python>
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path

# The 15 targets that logged "Failed to apply hooks to ..." on 2026-11-09,
# each once per process. 14 are multimodal/video-generation surfaces.
FAILED_TARGETS = (
    "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.LTX2PipelineConfig.supports_dynamic_batching",
    "sglang.multimodal_gen.configs.pipeline_configs.wan.Wan2_2_T2V_A14B_Config.__post_init__",
    "sglang.multimodal_gen.configs.pipeline_configs.wan.Wan2_2_TI2V_5B_Config.__post_init__",
    "sglang.multimodal_gen.configs.pipeline_configs.wan.WanI2V480PConfig.__post_init__",
    "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerImpl.forward_varlen",
    "sglang.multimodal_gen.runtime.layers.attention.turbo_layer.MinimalA2AAttnOp.__init__",
    "sglang.multimodal_gen.runtime.layers.attention.turbo_layer.MinimalA2AAttnOp.forward",
    "sglang.multimodal_gen.runtime.models.dits.ltx_2.LTX2Attention.forward",
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTModel.post_load_weights",
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_qk_norm",
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_rope_qk",
    "sglang.multimodal_gen.runtime.models.dits.wanvideo.WanTransformer3DModel.__init__",
    "sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor.PipelineExecutor.profile_execution",
    "sglang.multimodal_gen.runtime.server_args.server_args.ServerArgs._adjust_attention_backend",
    "sglang.srt.speculative.eagle_utils._sync_eagle_verify_outputs",
)

CONTROL_TARGET = "json.dumps"


def load_real_registry(python_root: Path):
    source = python_root / "sglang" / "srt" / "plugins" / "hook_registry.py"
    if not source.is_file():
        raise SystemExit(f"hook_registry.py not found under {python_root}")
    for name in ("sglang", "sglang.srt", "sglang.srt.plugins"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules.setdefault(name, module)
    module = types.ModuleType("sglang.srt.plugins.hook_registry")
    module.__file__ = str(source)
    sys.modules["sglang.srt.plugins.hook_registry"] = module
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
    # hook_guard.install() reaches the registry with ``from sglang.srt.plugins
    # import hook_registry``, which reads the attribute off the parent package
    # before sys.modules. Where the installed sglang is already imported (an
    # image whose startup loads the plugins), that attribute still names the
    # installed registry and the guard would land on one this replay never drives.
    sys.modules["sglang.srt.plugins"].hook_registry = module
    return module


class Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def run(module, use_guard: bool):
    HookRegistry = module.HookRegistry
    HookType = module.HookType

    def around(original, *a, **kw):
        return original(*a, **kw)

    if use_guard:
        from hcu_sglang import hook_guard

        hook_guard._installed = False
        hook_guard._skipped.clear()
        hook_guard.install()
    else:
        from hcu_sglang import hook_guard

        hook_guard._installed = False
        hook_guard._skipped.clear()

    HookRegistry.reset()
    HookRegistry._patched = set()
    for target in FAILED_TARGETS:
        HookRegistry.register(target, around, HookType.AROUND)
    HookRegistry.register(CONTROL_TARGET, around, HookType.AROUND)

    collector = Collect()
    root = logging.getLogger()
    root.addHandler(collector)
    try:
        HookRegistry.apply_hooks()
    finally:
        root.removeHandler(collector)

    errors = [r for r in collector.records if r.levelno >= logging.ERROR]
    summaries = [
        r.getMessage()
        for r in collector.records
        if r.levelno in (logging.INFO, logging.WARNING)
        and "hcu hook guard: skipped" in r.getMessage()
    ]
    patched = CONTROL_TARGET in HookRegistry._patched
    # Every target the build lacks is also logged once as "Failed to apply hooks to".
    failed = [
        record.getMessage().partition("Failed to apply hooks to ")[2]
        for record in collector.records
        if "Failed to apply hooks to " in record.getMessage()
    ]
    # One summary line per failure kind, the targets follow the reason.
    skipped = [
        target
        for message in summaries
        for target in message.partition("): ")[2].split(", ")
    ]
    return {
        "errors": len(errors),
        "failed": failed,
        "summaries": summaries,
        "skipped": skipped,
        "patched": patched,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-python", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.CRITICAL)  # quiet console
    module = load_real_registry(args.sglang_python.resolve())

    # Root level must let ERROR through: basicConfig above sets the root
    # threshold, which would otherwise filter the very records under test
    # before any handler sees them.
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, root.handlers[:]
    root.setLevel(logging.DEBUG)
    root.handlers = []
    try:
        before = run(module, use_guard=False)
        after = run(module, use_guard=True)
    finally:
        root.setLevel(saved_level)
        root.handlers = saved_handlers

    print("\n=== unguarded (what the 2026-11-09 logs show) ===")
    print("  'Failed to apply hooks' lines :", len(before["failed"]))
    print("  ERROR records with traceback  :", before["errors"])
    print("  control target patched        :", before["patched"])
    for target in before["failed"]:
        print("    -", target)

    print("\n=== guarded ===")
    print("  'Failed to apply hooks' lines :", len(after["failed"]))
    print("  ERROR records with traceback  :", after["errors"])
    print("  summary lines                 :", len(after["summaries"]))
    print("  control target patched        :", after["patched"])
    for message in after["summaries"]:
        print("\n  summary:", message)

    # What this build lacks is what the replay guards: every target that fails
    # unguarded must be one of the 15 and be skipped guarded, and nothing else.
    failed = before["failed"]
    if not failed:
        print("\n  this build has all 15 targets; nothing to replay")
    ok = (
        bool(failed)
        and set(failed) <= set(FAILED_TARGETS)
        and len(set(failed)) == len(failed) == before["errors"]
        and after["errors"] == 0
        and not after["failed"]
        and sorted(after["skipped"]) == sorted(failed)
        and before["patched"]
        and after["patched"]
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
