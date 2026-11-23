# Copyright 2023-2026 SGLang Team
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import weakref


def around_allocator_init(original, self, size, page_size, dtype, device, kvcache, need_sort, host_to_device_ratio=2):
    from hcu_sglang.layers.attention.hisparse import enabled

    if not isinstance(host_to_device_ratio, float) or not enabled(w4a8_only=True):
        return original(self, size, page_size, dtype, device, kvcache, need_sort, host_to_device_ratio)

    import torch
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.platforms import current_platform

    self._kvcache = kvcache
    self._size_full = kvcache.index_buf_size
    self._size_hisparse = size
    self.compress_ratio = 1
    self.dtype = dtype
    self.device = device
    self.page_size = page_size
    self.need_sort = need_sort
    self._spec_scratch_capacity = 0
    self._spec_device_buffer_size = 0

    paged_allocator_cls = (
        current_platform.get_paged_allocator_cls() or PagedTokenToKVPoolAllocator
    )
    self.logical_attn_allocator = paged_allocator_cls(
        self._size_full,
        self.page_size,
        self.dtype,
        self.device,
        kvcache,
        need_sort,
    )
    self.hisparse_attn_allocator = paged_allocator_cls(
        self._size_hisparse,
        self.page_size,
        self.dtype,
        self.device,
        kvcache,
        need_sort,
    )
    self.full_to_hisparse_device_index_mapping = torch.cat(
        [
            torch.zeros(
                self._size_full + self.page_size,
                dtype=torch.int64,
                device=self.device,
            ),
            torch.tensor([-1], dtype=torch.int64, device=self.device),
        ]
    )

    self.free_pages = None
    self.release_pages = None
    self.free_group = None
    self.clear()
    self._kvcache.register_mapping(
        weakref.proxy(self.full_to_hisparse_device_index_mapping)
    )
