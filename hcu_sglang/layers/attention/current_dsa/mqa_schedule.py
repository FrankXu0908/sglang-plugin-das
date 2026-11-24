# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from sglang.srt.layers.attention.dsa.utils import (
    resolve_paged_mqa_logits_metadata_fn as native_resolver,
)
from hcu_sglang.utils import detection


def resolve_paged_mqa_logits_metadata_fn():
    if detection.is_hcu():
        from lightop.attention import get_paged_mqa_logits_metadata

        return get_paged_mqa_logits_metadata
    return native_resolver()
