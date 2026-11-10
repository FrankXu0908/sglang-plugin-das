# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

# The EAGLE verify-sync AROUND hook is retired (PR #30 boundary follow-up).
# The anchor `_sync_eagle_verify_outputs` never existed in upstream history:
# official v0.5.22@baa9bef and main both keep the sync inline in eagle_sample
# and already scope broadcasts to attn_tp_group under dp-attention (the
# self-healed form), so the hook's whole-TP-group fix has no consumer there.
# The helper only existed in the sglang-das fork tree, which the plugin and
# adapter do not need to support. With no supported host left, unconditional
# registration only produced "Failed to apply hooks" once per process on
# official trees (10x at startup, 2026-11-09 S1 run, main baseline).
RETIRED_TARGET = "sglang.srt.speculative.eagle_utils._sync_eagle_verify_outputs"
RETIRED_WRAPPER = "around_sync_eagle_verify_outputs"
KEPT_TARGETS = (
    "sglang.srt.distributed.parallel_state.initialize_model_parallel",
    "sglang.srt.distributed.parallel_state.destroy_model_parallel",
    "sglang.srt.distributed.parallel_state.GroupCoordinator.__init__",
    "sglang.srt.distributed.parallel_state.init_world_group",
    "sglang.srt.distributed.parallel_state.init_model_parallel_group",
    "sglang.srt.distributed.bootstrap._set_all_reduce_flags",
)


class EagleSyncHookRemovedTests(unittest.TestCase):
    def _declared(self):
        return json.loads(
            (ROOT / "tests/contracts/hcu_source_alignment_targets.json").read_text(
                encoding="utf-8"
            )
        )["expected_hook_targets"]

    def test_plugin_source_no_longer_mentions_the_hook(self):
        offenders = [
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "hcu_sglang").rglob("*.py"))
            if RETIRED_TARGET in path.read_text(encoding="utf-8")
            or RETIRED_WRAPPER in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def test_declared_target_contract_no_longer_lists_it(self):
        self.assertNotIn(RETIRED_TARGET, self._declared())

    def test_permanent_parallel_state_hooks_remain_declared(self):
        declared = self._declared()
        for kept in KEPT_TARGETS:
            self.assertIn(kept, declared)


if __name__ == "__main__":
    unittest.main()
