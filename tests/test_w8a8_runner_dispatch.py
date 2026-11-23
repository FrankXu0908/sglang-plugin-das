# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def register_target_reprs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "HookRegistry"
            and node.args
        ):
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant):
            out.append(str(value.value))
        else:
            text = ast.unparse(value)
            if text.startswith(("f'", 'f"')):
                text = text[2:-1]
            out.append(text)
    return out


class W8A8RunnerDispatchTests(unittest.TestCase):
    def test_w8a8_arms_no_longer_register_w8a8_create_moe_runner(self):
        for name in ("w8a8_aiter", "w8a8_lightop"):
            path = ROOT / f"hcu_sglang/layers/quantization/{name}.py"
            targets = register_target_reprs(path)
            hits = [
                t
                for t in targets
                if t.endswith(".create_moe_runner")
                and ("{module}" in t or "{target}" in t)
            ]
            self.assertEqual(
                hits,
                [],
                f"{name} must not register w8a8 create_moe_runner (got {hits})",
            )

    def test_compressed_create_moe_runner_stays_with_lightop(self):
        path = ROOT / "hcu_sglang/layers/quantization/w8a8_lightop.py"
        targets = register_target_reprs(path)
        hits = [t for t in targets if t.endswith(".create_moe_runner")]
        self.assertEqual(hits, ["{compressed_target}.create_moe_runner"])
        self.assertIn("CompressedTensorsW8A8Fp8MoE", path.read_text())

    def test_dispatcher_registers_both_w8a8_targets_once(self):
        source = (
            ROOT / "hcu_sglang/layers/quantization/w8a8_runner_dispatch.py"
        ).read_text()
        self.assertIn('"W8A8Int8MoEMethod"', source)
        self.assertIn('"W8A8FP8MoEMethod"', source)
        targets = [
            t
            for t in register_target_reprs(
                ROOT / "hcu_sglang/layers/quantization/w8a8_runner_dispatch.py"
            )
            if t.endswith(".create_moe_runner")
        ]
        self.assertEqual(len(targets), 1)
        self.assertIn("{module}.{class_name}", targets[0])

    def test_lightop_arm_wins_when_requested(self):
        from hcu_sglang.layers.quantization import (
            w8a8_aiter,
            w8a8_lightop,
            w8a8_runner_dispatch,
        )

        original = object()
        with (
            patch.object(w8a8_lightop, "_requested_lightop", return_value=True),
            patch.object(w8a8_lightop, "around_create_moe_runner") as lightop_arm,
            patch.object(w8a8_aiter, "around_create_moe_runner") as aiter_arm,
        ):
            w8a8_runner_dispatch.around_create_moe_runner(
                original, "quant", "layer", "config"
            )
        lightop_arm.assert_called_once_with(original, "quant", "layer", "config")
        aiter_arm.assert_not_called()

    def test_aiter_arm_runs_when_lightop_not_requested(self):
        from hcu_sglang.layers.quantization import (
            w8a8_aiter,
            w8a8_lightop,
            w8a8_runner_dispatch,
        )

        original = object()
        with (
            patch.object(w8a8_lightop, "_requested_lightop", return_value=False),
            patch.object(w8a8_lightop, "around_create_moe_runner") as lightop_arm,
            patch.object(w8a8_aiter, "around_create_moe_runner") as aiter_arm,
        ):
            w8a8_runner_dispatch.around_create_moe_runner(
                original, "quant", "layer", "config"
            )
        aiter_arm.assert_called_once_with(original, "quant", "layer", "config")
        lightop_arm.assert_not_called()


    def test_apply_dispatches_by_runner_backend(self):
        from hcu_sglang.layers.quantization import (
            w8a8_aiter,
            w8a8_lightop,
            w8a8_runner_dispatch,
        )

        original = object()
        for lightop_runner in (True, False):
            with self.subTest(lightop_runner=lightop_runner):
                with (
                    patch.object(
                        w8a8_lightop,
                        "_uses_lightop_runner",
                        return_value=lightop_runner,
                    ),
                    patch.object(w8a8_lightop, "around_apply_int8") as lightop_arm,
                    patch.object(w8a8_aiter, "around_apply_int8") as aiter_arm,
                ):
                    w8a8_runner_dispatch.around_apply_int8(
                        original, "quant", "layer", "dispatch"
                    )
                if lightop_runner:
                    lightop_arm.assert_called_once_with(
                        original, "quant", "layer", "dispatch"
                    )
                    aiter_arm.assert_not_called()
                else:
                    aiter_arm.assert_called_once_with(
                        original, "quant", "layer", "dispatch"
                    )
                    lightop_arm.assert_not_called()

    def test_compressed_apply_dispatches_by_runner_backend(self):
        from hcu_sglang.layers.quantization import (
            w8a8_aiter,
            w8a8_lightop,
            w8a8_runner_dispatch,
        )

        original = object()
        with (
            patch.object(w8a8_lightop, "_uses_lightop_runner", return_value=True),
            patch.object(w8a8_lightop, "around_compressed_apply") as lightop_arm,
            patch.object(w8a8_aiter, "around_apply_compressed_fp8") as aiter_arm,
        ):
            w8a8_runner_dispatch.around_compressed_apply_weights(
                original, "quant", "layer", "dispatch"
            )
        lightop_arm.assert_called_once_with(original, "quant", "layer", "dispatch")
        aiter_arm.assert_not_called()

    def test_compressed_process_weights_chains_deepep_then_lightop(self):
        import types
        from hcu_sglang.layers import moe
        from hcu_sglang.layers.quantization import w8a8_lightop
        from hcu_sglang.layers.quantization import w8a8_runner_dispatch

        calls = []
        original = Mock(side_effect=lambda qm, layer: calls.append("original"))
        deepep_arm = Mock(
            side_effect=lambda orig, qm, layer: (
                calls.append("deepep"),
                orig(qm, layer),
            )[1]
        )
        lightop_arm = Mock(
            side_effect=lambda orig, qm, layer: (
                calls.append("lightop"),
                orig(qm, layer),
            )[1]
        )
        fake_deepep = types.ModuleType("hcu_sglang.layers.moe.deepep_fp8")
        fake_deepep.around_process_weights = deepep_arm
        with (
            patch.object(moe, "deepep_fp8", fake_deepep, create=True),
            patch.object(w8a8_lightop, "around_compressed_process_weights", lightop_arm),
        ):
            w8a8_runner_dispatch.around_compressed_process_weights(
                original, "quant", "layer"
            )
        # Registration order used to make lightop the outer hook: lightop's
        # original is the deepep chain, deepep's original is the loader.
        self.assertEqual(calls, ["lightop", "deepep", "original"])
        self.assertEqual(lightop_arm.call_args.args[0].__name__, "chained_original")

    def test_arms_no_longer_register_shared_apply_targets(self):
        for name in ("w8a8_aiter", "w8a8_lightop"):
            path = ROOT / f"hcu_sglang/layers/quantization/{name}.py"
            targets = register_target_reprs(path)
            shared = [
                t
                for t in targets
                if t.endswith((".apply", ".apply_weights"))
                or (
                    t.endswith(".process_weights_after_loading")
                    and "{compressed_target}" in t
                )
            ]
            self.assertEqual(shared, [], f"{name} must not register shared targets")


if __name__ == "__main__":
    unittest.main()
