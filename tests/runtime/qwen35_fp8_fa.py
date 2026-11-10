# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from hcu_sglang import server_args_hooks as args_hooks
from hcu_sglang.layers.attention import flashattention_backend as subject
from hcu_sglang.layers.attention import flashattention_interface as interface
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


@pytest.mark.parametrize("dtype,expected", [(torch.bfloat16, True), (torch.float16, False), (None, False)])
def test_auto_dtype_uses_resolved_model_dtype(dtype, expected):
    config = SimpleNamespace(dtype=dtype)
    view = SimpleNamespace(dtype="auto", get_model_config=lambda: config)
    assert args_hooks._uses_bfloat16(view) is expected


@pytest.mark.parametrize("scale", [None, 0.75])
def test_registered_prepare_keeps_bf16_query_and_rejects_scales(monkeypatch, scale):
    monkeypatch.setattr(subject, "is_hcu", lambda: True)
    target = "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.prepare_paged_mha_query"
    registered = []
    monkeypatch.setattr(HookRegistry, "register", lambda *args: registered.append(args))
    subject.register()
    selected = [(kind, hook, None) for name, hook, kind in registered if name == target]
    assert selected == [(HookType.AROUND, subject.around_prepare_paged_mha_query, None)]
    original = FlashAttentionBackend.prepare_paged_mha_query
    monkeypatch.setattr(FlashAttentionBackend, "prepare_paged_mha_query", original)
    backend = SimpleNamespace(_hcu_bf16_fp8_kv_query=True, kv_cache_dtype_str="fp8_e5m2",
        kv_cache_dtype=torch.float8_e5m2, kv_cache_is_mxfp8=False, fa_impl_ver=3)
    value = torch.tensor(scale) if scale is not None else None
    layer = SimpleNamespace(head_dim=256, k_scale=value, v_scale=value)
    q = torch.randn(3, 12, 256, dtype=torch.bfloat16)
    old = original(backend, q, None, None, layer, logical_batch_size=2, kv_head_num=2, is_prefill=False)
    assert old[0].dtype == torch.float8_e5m2
    try:
        HookRegistry._apply_target(target, selected)
        if scale is not None:
            with pytest.raises(RuntimeError, match="requires unscaled KV cache"):
                FlashAttentionBackend.prepare_paged_mha_query(backend, q, None, None, layer,
                    logical_batch_size=2, kv_head_num=2, is_prefill=False)
        else:
            result = FlashAttentionBackend.prepare_paged_mha_query(backend, q, None, None, layer,
                logical_batch_size=2, kv_head_num=2, is_prefill=False)
            assert result[0] is q and result[0].dtype == torch.bfloat16
            assert result[1:] == (None, None, None, None)
    finally:
        FlashAttentionBackend.prepare_paged_mha_query = original


@pytest.mark.parametrize("case", ["non_hcu", "disabled", "float16", "large_head", "rope"])
def test_prepare_preserves_effective_fallback(monkeypatch, case):
    monkeypatch.setattr(subject, "is_hcu", lambda: case != "non_hcu")
    backend = SimpleNamespace(_hcu_bf16_fp8_kv_query=case != "disabled")
    q = torch.ones(1, 2, 256, dtype=torch.float16 if case == "float16" else torch.bfloat16)
    layer = SimpleNamespace(head_dim=384 if case == "large_head" else 256)
    rope = q if case == "rope" else None
    original = mock.Mock(return_value=object())
    result = subject.around_prepare_paged_mha_query(original, backend, q, rope, None, layer,
        logical_batch_size=1, kv_head_num=2, is_prefill=False)
    assert result is original.return_value
    original.assert_called_once_with(backend, q, rope, None, layer,
        logical_batch_size=1, kv_head_num=2, is_prefill=False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_ragged_prefill_reaches_varlen_without_rectangular_query_view(monkeypatch, dtype):
    q = torch.randn(7, 12, 256, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 2, 7], dtype=torch.int32)
    k = torch.zeros(4, 2, 64, 256, dtype=dtype)
    v = torch.zeros_like(k)
    expected = torch.ones_like(q)
    native = ModuleType("flash_attn")
    native.flash_attn_varlen_func = mock.Mock(return_value=expected)
    native.flash_attn_with_kvcache = mock.Mock(side_effect=AssertionError("unexpected fallback"))
    native.vllm_flash_attn_with_kvcache = mock.Mock(side_effect=AssertionError("unexpected decode"))
    monkeypatch.setitem(sys.modules, "flash_attn", native)
    monkeypatch.setattr(interface, "_layout_enabled", lambda: True)
    out = torch.empty_like(q)
    result = interface.flash_attn_with_kvcache(q, k.view(4, 64, 2, 256), v.view(4, 64, 2, 256),
        cache_seqlens=torch.tensor([20, 30], dtype=torch.int32), page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        cu_seqlens_q=cu_q, max_seqlen_q=5, max_seqlen_k=30, causal=True, out=out)
    assert result is out
    torch.testing.assert_close(out, expected)
    kwargs = native.flash_attn_varlen_func.call_args.kwargs
    assert kwargs["q"] is q and kwargs["cu_seqlens_q"] is cu_q
    assert kwargs["k"].shape == kwargs["v"].shape == (4, 2, 64, 256)
    assert kwargs["layout"] == "bhsd" and kwargs["max_seqlen_q"] == 5
    native.flash_attn_varlen_func.assert_called_once()


@pytest.mark.parametrize("query_len", [1, 4])
def test_uniform_decode_and_verify_keep_native_query_contract(monkeypatch, query_len):
    q = torch.randn(2 * query_len, 12, 256, dtype=torch.bfloat16)
    k = torch.zeros(4, 2, 64, 256, dtype=torch.float8_e5m2)
    native = ModuleType("flash_attn")
    native.flash_attn_varlen_func = mock.Mock(return_value=q)
    native.vllm_flash_attn_with_kvcache = mock.Mock(return_value=q)
    native.flash_attn_with_kvcache = mock.Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setitem(sys.modules, "flash_attn", native)
    monkeypatch.setattr(interface, "_layout_enabled", lambda: True)
    interface.flash_attn_with_kvcache(q, k.view(4, 64, 2, 256), k.view(4, 64, 2, 256),
        cache_seqlens=torch.tensor([20, 30], dtype=torch.int32), page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32), max_seqlen_q=query_len, causal=True)
    if query_len == 1:
        native.vllm_flash_attn_with_kvcache.assert_called_once()
        assert native.vllm_flash_attn_with_kvcache.call_args.kwargs["q"].shape == (2, 1, 12, 256)
        native.flash_attn_varlen_func.assert_not_called()
    else:
        native.flash_attn_varlen_func.assert_called_once()
        assert native.flash_attn_varlen_func.call_args.kwargs["q"] is q
        native.vllm_flash_attn_with_kvcache.assert_not_called()


@pytest.fixture
def initialized_backend(monkeypatch):
    import sglang.srt.layers.attention.flashattention_backend as official
    from sglang.srt.speculative import spec_utils
    from hcu_sglang.mem_cache import hcu_mha_pool
    from hcu_sglang.utils import detection

    monkeypatch.setattr(subject, "is_hcu", lambda: True)
    monkeypatch.setattr(detection, "get_gcn_arch", lambda: "gfx936:sramecc+")
    monkeypatch.setattr(hcu_mha_pool, "_layout_enabled", lambda: True)
    monkeypatch.setattr(official, "get_parallel", lambda: SimpleNamespace(attn_cp_size=1, attn_tp_size=1))
    monkeypatch.setattr(official, "get_spec", lambda: SimpleNamespace(
        speculative_eagle_topk=1, speculative_num_draft_tokens=4, speculative_algorithm="NEXTN"))
    monkeypatch.setattr(spec_utils, "get_spec", official.get_spec)
    monkeypatch.setattr(official, "get_exec", lambda: SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False)))
    monkeypatch.setattr(official, "get_model", lambda: SimpleNamespace(is_embedding=False))
    monkeypatch.setattr(official, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(official, "get_kv_shard_pool", lambda pool: None)
    monkeypatch.setattr(official, "_should_disable_scheduler_metadata_precompute", lambda: False)
    monkeypatch.setattr(FlashAttentionBackend, "_maybe_init_local_attn_builder", lambda *args: None)
    original = FlashAttentionBackend.prepare_paged_mha_query
    fallback = mock.Mock(wraps=original)

    def effective(*args, **kwargs):
        return fallback(*args, **kwargs)

    monkeypatch.setattr(FlashAttentionBackend, "prepare_paged_mha_query", effective)
    monkeypatch.setattr(FlashAttentionBackend, "__init__", FlashAttentionBackend.__init__)
    registered = []
    monkeypatch.setattr(HookRegistry, "register", lambda *args: registered.append(args))
    subject.register()
    for target, hook, kind in registered:
        if target.endswith((".__init__", ".prepare_paged_mha_query")):
            HookRegistry._apply_target(target, [(kind, hook, None)])

    def create(architecture, page_size=64, keyword=False):
        config = SimpleNamespace(
            is_encoder_decoder=False, context_len=1024, head_dim=256,
            attention_arch=official.AttentionArch.MHA,
            hf_config=SimpleNamespace(architectures=[architecture] if architecture else []),
            hf_text_config=SimpleNamespace(num_attention_heads=12),
            get_num_kv_heads=lambda size: 2)
        runner = SimpleNamespace(
            model_config=config, sliding_window_size=None, device="cpu",
            req_to_token_pool=SimpleNamespace(size=2, req_to_token=torch.zeros(2, 1024, dtype=torch.int32)),
            token_to_kv_pool=SimpleNamespace(), kv_index_translator=SimpleNamespace(is_translating=False),
            kv_cache_dtype=torch.float8_e5m2, kv_cache_dtype_str="fp8_e5m2", page_size=page_size,
            is_draft_worker=bool(architecture and architecture.endswith("MTP")))
        if keyword:
            return FlashAttentionBackend(model_runner=runner)
        return FlashAttentionBackend(runner)

    return create, original, fallback


@pytest.mark.parametrize("architecture", [
    "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5ForCausalLMMTP"])
@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("scale", [None, 1.0, 0.75])
def test_initialized_qwen35_page64_query_contract(initialized_backend, architecture, is_prefill, scale):
    create, original, fallback = initialized_backend
    backend = create(architecture, keyword=is_prefill)
    assert backend._hcu_bf16_fp8_kv_query
    value = torch.tensor(scale) if scale is not None else None
    layer = SimpleNamespace(head_dim=256, k_scale=value, v_scale=value)
    q = torch.randn(6 if is_prefill else 2, 12, 256, dtype=torch.bfloat16)
    kwargs = dict(logical_batch_size=2, kv_head_num=2, is_prefill=is_prefill)
    assert original(backend, q, None, None, layer, **kwargs)[0].dtype == torch.float8_e5m2
    if scale is not None:
        with pytest.raises(RuntimeError, match="requires unscaled KV cache"):
            backend.prepare_paged_mha_query(q, None, None, layer, **kwargs)
    else:
        result = backend.prepare_paged_mha_query(q, None, None, layer, **kwargs)
        assert result[0] is q
        assert result[1:] == (None, None, None, None)
    fallback.assert_not_called()


@pytest.mark.parametrize("architecture,page_size", [
    ("Qwen4ExpForConditionalGeneration", 64),
    ("Qwen4ExpForCausalLMMTP", 64),
    ("Qwen3_5ForConditionalGeneration", 16),
    ("Qwen3_5ForConditionalGeneration", 128),
    ("Qwen3_5ForCausalLMMTP", 16),
    ("Qwen3_5MoeForConditionalGeneration", 64),
    ("UnknownForCausalLM", 64),
    (None, 64),
])
@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("scale", [None, 1.0, 0.75])
def test_initialized_out_of_scope_query_matches_official(
        initialized_backend, architecture, page_size, is_prefill, scale):
    create, original, fallback = initialized_backend
    backend = create(architecture, page_size, keyword=is_prefill)
    k_scale = torch.tensor(scale) if scale is not None else None
    v_scale = torch.tensor(scale * 2 if scale != 1.0 else scale) if scale is not None else None
    layer = SimpleNamespace(head_dim=256, k_scale=k_scale, v_scale=v_scale)
    q = torch.randn(6 if is_prefill else 2, 12, 256, dtype=torch.bfloat16)
    kwargs = dict(logical_batch_size=2, kv_head_num=2, is_prefill=is_prefill)
    expected = original(backend, q, None, None, layer, **kwargs)
    actual = backend.prepare_paged_mha_query(q, None, None, layer, **kwargs)
    fallback.assert_called_once_with(backend, q, None, None, layer, **kwargs)
    assert not backend._hcu_bf16_fp8_kv_query
    assert actual[0].dtype == expected[0].dtype == torch.float8_e5m2
    assert actual[1:3] == expected[1:3] == (None, None)
    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=0, atol=0)
    for got, want in zip(actual[3:], expected[3:]):
        if want is None:
            assert got is None
        else:
            assert got.shape == want.shape == (2, 2)
            assert got.untyped_storage().data_ptr() == want.untyped_storage().data_ptr()
            torch.testing.assert_close(got, want, rtol=0, atol=0)
