from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator as NativeKVCacheConfigurator
from sglang.srt.runtime_context import get_parallel

from hcu_sglang.utils.detection import is_hcu

_scratch_source = ContextVar("hcu_dsa_layer_split_scratch_source", default=None)


def before_draft_alloc(self, *args, **kwargs):
    if not is_hcu():
        return
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from hcu_sglang.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool

    target_pool = self.target_worker.model_runner.token_to_kv_pool
    self.draft_runner.dsa_layer_split_scratch_source = None
    if (
        self.server_args.disaggregation_mode == DisaggregationMode.PREFILL.value
        and get_parallel().pp_size == 1
        and self.speculative_algorithm.is_eagle()
        and not self.speculative_algorithm.is_eagle3()
        and self.draft_runner.layer_info.num_effective_layers == 1
        and self.draft_runner.model_config.num_nextn_predict_layers == 1
        and isinstance(target_pool, LayerSplitDSATokenToKVPool)
    ):
        self.draft_runner.dsa_layer_split_scratch_source = target_pool


def around_init_configurator(original, self):
    source = getattr(self, "dsa_layer_split_scratch_source", None)
    token = _scratch_source.set(source)
    try:
        return original(self)
    finally:
        _scratch_source.reset(token)


def after_default_pool_init(result, configurator, kvc):
    from hcu_sglang.layers.glm53_decode import model_matches
    from sglang.srt.arg_groups.overrides import resolved_view
    from sglang.srt.runtime_context import get_disagg, get_memory

    model_config = kvc.model_config
    hf_config = model_config.hf_config
    parallel = get_parallel()
    if (
        kvc.is_draft_worker
        or not model_matches(hf_config)
        or get_disagg().disaggregation_mode != "prefill"
        or parallel.pp_size != 1
        or parallel.attn_dcp_size != 1
        or not kvc.spec_algorithm.is_eagle()
        or kvc.spec_algorithm.is_eagle3()
        or resolved_view(kvc.server_args).enable_multi_layer_eagle
        or kvc.spec_aux_config.eagle_draft_num_layers != 1
        or model_config.num_nextn_predict_layers != 1
        or get_memory().enable_hisparse
    ):
        return result

    from sglang.srt.layers.cp.utils import (
        get_glm_dsa_cp_layer_shard_info,
        get_layer_shard_range,
    )
    from sglang.srt.utils.common import is_float4_e2m1fn_x2

    rank, shard_size = get_glm_dsa_cp_layer_shard_info(kvc)
    if (
        rank is None
        or shard_size <= 1
        or is_float4_e2m1fn_x2(kvc.kv_cache_dtype)
        or kvc.fused_entry_bytes("full") is not None
    ):
        return result

    from sglang.srt.configs.model_config import (
        dsa_layer_skips_topk,
        get_dsa_index_head_dim,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import (
        _should_elide_dsa_index_k,
        calculate_mla_kv_cache_dim,
    )
    from hcu_sglang.mem_cache.dsa_index_cache_mode import (
        index_k_cache_bytes_per_token,
        index_k_workspace_bytes_per_token,
        resolve_index_k_cache_mode,
    )

    main_kv_bytes = calculate_mla_kv_cache_dim(
        model_config=model_config, kv_cache_dtype=kvc.kv_cache_dtype,
    ) * kvc.kv_cache_dtype.itemsize
    cache_mode = resolve_index_k_cache_mode(
        kvc.kv_cache_dtype, kvc.pool_page_size, get_dsa_index_head_dim(hf_config),
    )
    index_bytes = index_k_cache_bytes_per_token(cache_mode)
    workspace_bytes = 2 * index_k_workspace_bytes_per_token(cache_mode)
    num_layers = kvc.layer_info.num_effective_layers
    first_layer = kvc.layer_info.start_layer
    elide_index_k = _should_elide_dsa_index_k(is_draft_worker=False)
    rank_costs = []
    for cp_rank in range(shard_size):
        start, end = get_layer_shard_range(cp_rank, shard_size, num_layers)
        draft_owned = int(cp_rank == shard_size - 1)
        owned = end - start + draft_owned
        owned_indexers = draft_owned + sum(
            not elide_index_k or not dsa_layer_skips_topk(hf_config, first_layer + i)
            for i in range(start, end)
        )
        # Main-KV shares one target scratch; Index-K has two independent scratches.
        rank_costs.append(
            (owned + 1) * main_kv_bytes
            + (owned_indexers + 2) * index_bytes
            + workspace_bytes
        )
    configurator._cell_size = math.ceil(max(rank_costs))
    return result


@dataclass(slots=True, kw_only=True)
class HCUKVCacheConfigurator(NativeKVCacheConfigurator):
    dsa_layer_split_scratch_source: Any = field(default_factory=_scratch_source.get)

    _hisparse_pd_capacity: int | None = field(default=None, init=False)

    def configure(self, *, pre_model_load_memory):
        from hcu_sglang.layers.attention.hisparse import enabled

        result = super(HCUKVCacheConfigurator, self).configure(
            pre_model_load_memory=pre_model_load_memory
        )
        if not self.is_draft_worker and enabled(pd_decode_only=True):
            self._hisparse_pd_capacity = result.token_to_kv_pool_allocator.size_full
        return result

    def logical_token_capacity(self, *, max_total_num_tokens):
        if self._hisparse_pd_capacity is not None:
            return self._hisparse_pd_capacity
        return super(HCUKVCacheConfigurator, self).logical_token_capacity(
            max_total_num_tokens=max_total_num_tokens
        )

    def _derive_pool_sizes(self, *, config):
        if isinstance(config.max_total_num_tokens, float):
            from hcu_sglang.layers.attention.hisparse import enabled

            if enabled():
                config = replace(config, max_total_num_tokens=int(config.max_total_num_tokens))
        return super(HCUKVCacheConfigurator, self)._derive_pool_sizes(config=config)

    def _build_dsa_kv_pool(
        self, *, max_total_num_tokens, max_running_requests, dsa_pool_class,
    ):
        source = self.dsa_layer_split_scratch_source
        if source is None:
            return super(HCUKVCacheConfigurator, self)._build_dsa_kv_pool(
                max_total_num_tokens=max_total_num_tokens,
                max_running_requests=max_running_requests,
                dsa_pool_class=dsa_pool_class,
            )
        from sglang.srt.configs.model_config import (
            dsa_layer_skips_topk, get_dsa_index_head_dim,
            get_dsa_index_kpool, get_dsa_index_kpool_compress,
        )
        from sglang.srt.layers.cp.utils import get_glm_dsa_cp_layer_shard_info
        from sglang.srt.mem_cache.kv_cache_configurator import (
            _should_elide_dsa_index_k, calculate_mla_kv_cache_dim,
        )
        from sglang.srt.runtime_context import get_exec, max_speculative_num_draft_tokens
        from hcu_sglang.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
        from hcu_sglang.layers.glm53_decode import model_matches

        if model_matches(self.model_config.hf_config):
            rank, size = source.layer_shard_rank, source.layer_shard_size
        else:
            rank, size = get_glm_dsa_cp_layer_shard_info(self)
        if not self.is_draft_worker or rank is None or size <= 1:
            raise ValueError("Shared DSA scratch requires a layer-split draft pool")
        pool_kwargs = {}
        if self.loc_space_scale > 1:
            pool_kwargs["dcp_replicated"] = True
        if _should_elide_dsa_index_k(is_draft_worker=self.is_draft_worker):
            pool_kwargs["skip_topk_layers"] = [
                dsa_layer_skips_topk(self.model_config.hf_config, layer_id)
                for layer_id in range(self.layer_info.start_layer, self.layer_info.end_layer)
            ]
        return LayerSplitDSATokenToKVPool(
            max_total_num_tokens,
            page_size=self.pool_page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.model_config.kv_lora_rank,
            qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            layer_num=self.layer_info.num_effective_layers,
            device=self.device,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=self.model_config, kv_cache_dtype=self.kv_cache_dtype,
            ),
            enable_memory_saver=get_exec().features.enable_memory_saver,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
            index_kpool=get_dsa_index_kpool(self.model_config.hf_config),
            index_kpool_compress=get_dsa_index_kpool_compress(self.model_config.hf_config),
            tail_extra_slots=(max_speculative_num_draft_tokens() or 0),
            max_running_requests=max_running_requests,
            layer_shard_rank=rank,
            layer_shard_size=size,
            layer_shard_rank_offset=size - 1,
            layer_split_scratch_source=source,
            **pool_kwargs,
        )


class _PackedDraftMLAHostView:
    def __init__(self, host_pool, draft_pool):
        self._host_pool = host_pool
        self.owns_draft_layer = draft_pool._is_layer_owned(draft_pool.start_layer)

    def __getattr__(self, name):
        return getattr(self._host_pool, name)

    def _is_device_layer_owned(self, device_pool, layer_id):
        return self.owns_draft_layer

    def _host_layer_index(self, layer_id, device_pool=None):
        return self.target_layer_num


def _glm53_packed_draft_mla_host(host_pool, layer_id):
    if layer_id != host_pool.device_pool.layer_num:
        return None
    drafts = host_pool.mtp_draft_device_pools
    if len(drafts) != 1:
        return None
    draft = drafts[0]
    if getattr(draft, "layer_split_scratch_source", None) is not host_pool.device_pool:
        return None

    from hcu_sglang.layers.glm53_decode import model_matches
    from sglang.srt.runtime_context import get_memory, process_model_config

    if (
        get_memory().hicache_storage_backend is None
        and model_matches(process_model_config().hf_config)
    ):
        return _PackedDraftMLAHostView(host_pool, draft)
    return None


def around_mla_load_layer(
    original, host_pool, device_pool, host_indices, device_indices, layer_id,
    io_backend, *, is_draft=False,
):
    if is_draft:
        view = _glm53_packed_draft_mla_host(host_pool, layer_id)
        if view is not None:
            if view.owns_draft_layer:
                return original(
                    view, device_pool, host_indices, device_indices, layer_id, io_backend,
                )
            return None
    return original(
        host_pool, device_pool, host_indices, device_indices, layer_id, io_backend,
        is_draft=is_draft,
    )


def around_mla_backup_layer(
    original, host_pool, host_indices, device_indices, layer_id, io_backend,
    *, is_draft=False,
):
    if is_draft:
        view = _glm53_packed_draft_mla_host(host_pool, layer_id)
        if view is not None:
            if view.owns_draft_layer:
                return original(view, host_indices, device_indices, layer_id, io_backend)
            return None
    return original(
        host_pool, host_indices, device_indices, layer_id, io_backend, is_draft=is_draft,
    )


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.pool_host.mla.MLATokenToKVPoolHost.load_to_device_per_layer",
        around_mla_load_layer, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.pool_host.mla.MLATokenToKVPoolHost._backup_from_device_per_layer",
        around_mla_backup_layer, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.alloc_memory_pool",
        before_draft_alloc, HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.init_kv_cache_configurator",
        around_init_configurator, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator.__init__",
        after_default_pool_init, HookType.AFTER,
    )
    for target in (
        "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator",
        "sglang.srt.model_executor.model_runner.KVCacheConfigurator",
    ):
        HookRegistry.register(target, HCUKVCacheConfigurator, HookType.REPLACE)
