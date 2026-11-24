"""QSA prefill 调优与 HIP 平台守卫。

两部分：
1. row_chunk_size / around_row_chunk_size —— logits 预算切分（原有逻辑）。
2. around_mqa_prefill —— key 分段 + HIP 上的 TileLang 守卫（见下）。

关于 TileLang 守卫
------------------
上游 sglang 的 `qsa_mqa_prefill` 在不同版本里对 HIP 的处理不一致：

- 0.5.18（如 dtk2604 镜像）带守卫，注释写明原因：
    # HIP: TileLang QSA MQA has been observed to VMFault during warmup
    # Prefer the Torch reference until the HIP TileLang path is stable.
    if q.is_cuda and HAS_TILELANG and not is_hip():
  HIP 上走 torch_qsa_mqa_prefill。

- 0.5.22（如 dtk2610 镜像）守卫被去掉了，变成：
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill(...)
  而该 TileLang kernel 的 shared memory 需求是
    q_shared  = block_q * heads * head_dim * 2B
    k_shared  = block_n * head_dim * 2B   × num_stages(3)
  合计 80 KiB，超过设备 shared_memory_per_block = 64 KiB，
  hipModuleLaunchKernel 直接返回 invalid argument，prefill 首次请求即崩。

因此这里在 HIP 上主动改走 torch 参考实现，不依赖上游是否带守卫：
对已带守卫的版本是幂等无操作（它本来就不进 TileLang），
对缺失守卫的版本修复崩溃。
"""

from __future__ import annotations


def row_chunk_size(budget_mib, rows, keys, heads, original):
    if budget_mib == 0:
        return original(rows, keys, heads)
    if not 1 <= budget_mib <= 2048:
        raise ValueError('QSA prefill logits budget must be 0 or 1..2048 MiB')
    if rows <= 0 or keys <= 0:
        return original(rows, keys, heads)
    if heads <= 0:
        raise ValueError('QSA query head count must be positive')
    block_q = max(1, 128 // heads)
    padded_rows = max(block_q, budget_mib * 1024 * 1024 // (keys * 4))
    padded_rows = max(block_q, padded_rows // block_q * block_q)
    return min(rows, padded_rows)


def around_row_chunk_size(original, rows, keys, heads):
    from sglang.srt.environ import envs
    return row_chunk_size(envs.SGLANG_HCU_QSA_PREFILL_LOGITS_MIB.get(),
                          rows, keys, heads, original)


def _hip_torch_fallback(q, k, row_starts, row_ends, score_scale):
    """HIP 上绕过 TileLang，直接走上游 torch 参考实现。

    返回 None 表示不适用（非 HIP、或该 sglang 版本没有 torch 参考实现、
    或 torch 路径不可用），调用方应回落到 original。
    """
    try:
        from sglang.srt.utils.common import is_hip
    except Exception:
        return None
    if not is_hip():
        return None
    try:
        from sglang.srt.layers.attention.qsa.mqa import torch_qsa_mqa_prefill
    except Exception:
        return None
    try:
        return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    except Exception:
        # torch 路径也不可用时交回上游，让它自己报错，避免掩盖真实问题。
        return None


def around_mqa_prefill(original, q, k, row_starts, row_ends, score_scale=None):
    from sglang.srt.environ import envs
    key_span = envs.SGLANG_HCU_QSA_PREFILL_KEY_SPAN.get()
    if key_span not in (0, 1024, 2048, 4096, 8192):
        raise ValueError('QSA prefill key span must be 0, 1024, 2048, 4096 or 8192')

    # 1) key 分段实现（形状完全匹配时），优先于 HIP 守卫：
    #    它是为长 key 专门做的分块 kernel，smem 只有 48 KiB，HIP 上可安全使用。
    if key_span != 0:
        import torch
        if (q.is_cuda and k.device == q.device
                and q.dtype == k.dtype == torch.bfloat16
                and q.ndim == k.ndim == 3
                and q.shape[1:] == (4, 128) and k.shape[1:] == (1, 128)
                and k.shape[0] >= 8192
                and row_starts.device == row_ends.device == q.device
                and row_starts.ndim == row_ends.ndim == 1
                and row_starts.numel() == row_ends.numel() == q.shape[0]):
            from hcu_sglang.models.qwen.qsa_mqa_split_key import prefill
            return prefill(q, k, row_starts, row_ends, score_scale, key_span=key_span)

    # 2) HIP 守卫：key_span 未命中的形状会落到上游 tilelang 分支，
    #    而该分支在 HIP 上 smem 超限。改走 torch 参考实现。
    fallback = _hip_torch_fallback(q, k, row_starts, row_ends, score_scale)
    if fallback is not None:
        return fallback

    return original(q, k, row_starts, row_ends, score_scale)

