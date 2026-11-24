from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


@unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_LAYER_FIX_TESTS") == "1", "requires pinned container runtime")
class LayerMoeFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, M, Q, C
        import torch
        from hcu_sglang.compat.aiter import bootstrap_aiter_compat
        bootstrap_aiter_compat()
        from hcu_sglang.environ import register_envs
        register_envs()
        from hcu_sglang.layers.moe import mega_moe as M
        from hcu_sglang.layers.quantization import slimquant_w4a8 as Q
        from hcu_sglang.layers.quantization import compressed_w8a8_int8_moe as C

    def test_deepgemm_prepared_weights_keep_graph_dispatch_and_capacity_rejection(self):
        experts = SimpleNamespace(_mega_moe_weights_built=True, _mega_moe_hcu_runtime="deep_gemm")
        moe = SimpleNamespace(experts=experts)
        with patch.object(M, "get_moe_a2a_backend", return_value=SimpleNamespace(is_megamoe=lambda: True)), patch.object(M, "get_hcu_mega_moe_runtime", return_value="deep_gemm"), patch.object(M, "_get_is_capture_mode" if hasattr(M, "_get_is_capture_mode") else "get_is_capture_mode", return_value=True):
            self.assertTrue(M.should_use_mega_moe(moe, torch.empty(1, 8)))
        with patch.object(M, "get_moe_a2a_backend", return_value=SimpleNamespace(is_megamoe=lambda: True)), patch.object(M, "get_hcu_mega_moe_runtime", return_value="deep_gemm"), patch.object(M, "_get_is_capture_mode" if hasattr(M, "_get_is_capture_mode") else "get_is_capture_mode", return_value=False), patch.object(M, "get_dp_global_num_tokens", return_value=[10**8]), patch.object(M, "is_dsa_enable_prefill_cp", return_value=False):
            with self.assertRaisesRegex(ValueError, "exceeds cap"):
                M.should_use_mega_moe(moe, torch.empty(1, 8))

    def test_standalone_quantizer_receives_view_of_flat_symmetric_scale_storage(self):
        x = torch.ones(3, 8)
        buf = SimpleNamespace(x=torch.empty(3, 8, dtype=torch.float8_e4m3fn), x_sf=torch.zeros(3), topk_idx=torch.empty(3, 1, dtype=torch.int64), topk_weights=torch.empty(3, 1))
        def quant(value, *, dtype, out_q, out_scale):
            self.assertEqual(out_scale.shape, (3, 1))
            self.assertEqual(out_scale.data_ptr(), buf.x_sf.data_ptr())
            out_scale.fill_(2)
        with patch.object(M, "_get_hcu_w8a8_pre_dispatch_quant", return_value=quant):
            M._prepare_standalone_megamoe_inputs(x, torch.zeros(3, 1, dtype=torch.int64), torch.ones(3, 1), buf, 3)
        torch.testing.assert_close(buf.x_sf, torch.full((3,), 2.))

    def test_ll_capability_gate_preserves_explicit_failure_and_auto_normal(self):
        provider = types.ModuleType("megamoe.dcu_megamoe_opt.v3_config")
        provider.staged_v3_capability_supported = Mock(return_value=False)
        buf = SimpleNamespace(group=SimpleNamespace(size=lambda: 8), num_experts=384, num_topk=6, hidden=4096, intermediate_hidden=2304)
        with patch.dict(sys.modules, {provider.__name__: provider}), patch.object(M, "is_dsa_enable_prefill_cp", return_value=False), patch.object(M, "_is_pd_prefill_instance", return_value=False):
            with patch.dict(os.environ, {"MEGAMOE_HCU_BACKEND": "auto"}):
                self.assertEqual(M._select_hcu_megamoe_backend(1, buf), "normal")
            with patch.dict(os.environ, {"MEGAMOE_HCU_BACKEND": "ll"}):
                with self.assertRaisesRegex(ValueError, "does not support"):
                    M._select_hcu_megamoe_backend(1, buf)
        self.assertEqual(provider.staged_v3_capability_supported.call_count, 2)

    def test_slimquant_expert_only_config_retains_bf16_dense_method(self):
        from sglang.srt.layers import linear as L
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        with patch.object(L, "get_parallel", return_value=SimpleNamespace(tp_group=None, tp_size=1, tp_rank=0)):
            layer = L.ReplicatedLinear(8, 4, bias=False, params_dtype=torch.float32)
        config = Q.SlimQuantW4A8Int8Config()
        self.assertIsInstance(config.get_quant_method(layer, "mlp"), UnquantizedLinearMethod)

    def test_compressed_int8_constructor_rejects_shuffled_aiter_without_matching_table(self):
        from compressed_tensors.quantization import QuantizationStrategy as S
        config = SimpleNamespace(strategy=S.CHANNEL)
        input_config = SimpleNamespace(strategy=S.TOKEN, dynamic=True)
        scheme = C.CompressedTensorsW8A8Int8MoE(config, input_config)
        layer = torch.nn.Module()
        scheme.create_weights(layer, 2, 64, 128, torch.bfloat16)
        layer.top_k = 2
        with patch.object(scheme, "_probe_hcu_aiter_int8_noshuffle", return_value=False), patch.object(C, "MoeRunner", side_effect=lambda backend, cfg: SimpleNamespace(runner_backend=backend)), patch.object(C, "get_moe_runner_backend", return_value=C.MoeRunnerBackend.AITER):
            scheme.create_moe_runner(layer, SimpleNamespace())
        self.assertTrue(scheme.runner.runner_backend.is_triton())
        original = layer.w13_weight
        scheme.process_weights_after_loading(layer)
        self.assertIs(layer.w13_weight, original)

    def test_int8_deepgemm_postload_produces_both_layouts_for_real_dispatch_consumer(self):
        from hcu_sglang.layers.moe import deepep_int8 as I, deepep_fp8 as F, deepep_current as D
        from hcu_sglang.layers.quantization import hcu_deepgemm_w8a8_utils as W
        from compressed_tensors.quantization import QuantizationStrategy as S
        scheme = C.CompressedTensorsW8A8Int8MoE(SimpleNamespace(strategy=S.CHANNEL), SimpleNamespace(strategy=S.TOKEN, dynamic=True))
        layer = torch.nn.Module()
        scheme.create_weights(layer, 2, 64, 128, torch.bfloat16)
        layer.w13_weight.data.fill_(1)
        layer.w2_weight.data.fill_(2)
        layer.scheme = scheme
        layer.quant_method = scheme
        layer.moe_runner_config = SimpleNamespace(activation="silu", swiglu_limit=None)
        layer.deepep_mode = SimpleNamespace(is_normal=lambda: False)
        with patch.dict(os.environ, {"SGLANG_USE_DEEPGEMM_MOE": "1"}), patch.object(I, "use_int8_deepgemm_asm", return_value=True), patch.object(I, "is_hcu_pair32_int8", return_value=False):
            I.configure_layer(layer, None)
        self.assertTrue(layer.use_int8_w8a8_deepgemm)
        mode = SimpleNamespace(is_normal=lambda: False, enable_normal=lambda: True, enable_low_latency=lambda: True)
        with patch.object(W, "get_hcu_gfx_arch", return_value="gfx938"), patch.object(W, "use_int8_deepgemm_asm", return_value=True), patch("sglang.srt.layers.moe.utils.get_deepep_mode", return_value=mode), patch("deepgemm.marlin_i8_contiguous_weight", side_effect=lambda w, **kw: w.clone()):
            scheme.process_weights_after_loading(layer)
        self.assertFalse(hasattr(layer, "w13_weight"))
        self.assertEqual(layer._w8a8_int8_w13_weight_shape, (2, 256, 64))
        self.assertTrue(layer._w8a8_int8_deepgemm_has_contiguous)
        self.assertTrue(layer._w8a8_int8_deepgemm_has_masked)
        seen = []
        def gemm(a, weights, output, counts, expected):
            seen.append(weights[0])
            output.fill_(len(seen))
        x = torch.ones(2, 3, 64, dtype=torch.int8)
        ids = torch.tensor([[0], [1]], dtype=torch.int64)
        output = D.DeepEPLLDispatchOutput(x, torch.ones(2, 3, 1), ids, torch.ones(2, 1), torch.tensor([3, 3], dtype=torch.int32), 3)
        with patch("deepgemm.m_grouped_w8a8_gemm_nt_masked", side_effect=gemm), patch("torch.ops.sglang.fuse_silu_mul_quant_ep", side_effect=I.fuse_silu_mul_quant_ep_wrapper), patch.object(I, "fuse_silu_mul_quant_ep", side_effect=lambda x, *args, **kw: (x.to(torch.int8), torch.ones(2, 3, 1))):
            combined = F.around_run_moe_core(Mock(side_effect=AssertionError("generic runner")), layer, output)
        self.assertIs(seen[0], layer.w13_weight_deepgemm_masked)
        self.assertIs(seen[1], layer.w2_weight_deepgemm_masked)
        torch.testing.assert_close(combined.hidden_states, torch.full((2, 3, 64), 2., dtype=torch.bfloat16))
        self.assertIs(combined.topk_ids, ids)
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            fake_input = torch.empty(2, 3, 256)
            fake_q, fake_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(fake_input)
            self.assertEqual(fake_q.shape, (2, 3, 128))
            self.assertEqual(fake_q.dtype, torch.int8)
            self.assertEqual(fake_scale.shape, (2, 3, 1))

    def test_lightop_runner_preserves_explicit_zero_route_scale(self):
        from hcu_sglang.layers.moe import lightop_runner as L
        L.register_lightop_backend()
        config = SimpleNamespace(routed_scaling_factor=0., inplace=False, activation="silu", apply_router_weight_on_input=False, num_experts=1)
        runner = L.LightOpRunnerCore(config)
        value = torch.ones(2, 4)
        inputs = L.LightOpRunnerInput(value, torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64))
        quant = L.LightOpMoeQuantInfo(torch.ones(1, 8, 4), torch.ones(1, 4, 4), torch.ones(1, 8, 1), torch.ones(1, 4, 1))
        def gemm(x, *args, **kwargs):
            self.assertEqual(kwargs["routed_scaling_factor"], 0.)
            return x * kwargs["routed_scaling_factor"]
        with patch("torch.ops.sglang.fused_experts_impl_int8_marlin", side_effect=gemm):
            output = runner.run(inputs, quant, {})
        torch.testing.assert_close(output.hidden_states, torch.zeros_like(value))

    def test_compressed_marlin_registry_constructs_source_config_lazily(self):
        from hcu_sglang.layers.quantization import registry as R
        from hcu_sglang.layers.quantization.compressed_marlin import SlimQuantCompressedTensorsMarlinConfig
        from sglang.srt.layers.quantization import get_quantization_config
        R.install_quantization_method()
        config_type = get_quantization_config("slimquant_marlin")
        config = config_type(target_scheme_map={}, ignore=[], quant_format="int-quantized", sparsity_scheme_map={}, sparsity_ignore_list=[])
        self.assertIsInstance(config, SlimQuantCompressedTensorsMarlinConfig)
        self.assertEqual(config.get_name(), "slimquant_marlin")
        self.assertEqual(config_type.override_quantization_method({"quant_method": "compressed-tensors"}, "slimquant_marlin"), "slimquant_marlin")

    @unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1", "requires owned GPU lane")
    def test_int8_custom_op_matches_provider_and_replays_changed_input(self):
        from hcu_sglang.layers.moe import deepep_int8 as I
        x = torch.randn(2, 4, 256, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
        def run():
            return torch.ops.sglang.fuse_silu_mul_quant_ep(x, counts)
        expected = I.fuse_silu_mul_quant_ep(x, counts)
        actual = run()
        def check(a, b):
            for expert, rows in enumerate([2, 3]):
                torch.testing.assert_close(a[0][expert, :rows], b[0][expert, :rows], rtol=0, atol=0)
                torch.testing.assert_close(a[1][expert, :rows], b[1][expert, :rows], rtol=0, atol=0)
        check(actual, expected)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        check(output, I.fuse_silu_mul_quant_ep(x, counts))

    def test_dp_public_padding_class_preserves_type_and_hcu_graph_gate(self):
        from hcu_sglang.layers import dp_attention as D
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset()
        D.register_hooks()
        HookRegistry.apply_hooks()
        self.assertIs(D.DpPaddingMode, D.N.DpPaddingMode)
        with patch.object(D, "_dp_use_max_len", return_value=False):
            self.assertIs(D.N.DpPaddingMode.get_default_mode_in_cuda_graph(), D.N.DpPaddingMode.SUM_LEN)
        with patch.object(D, "_dp_use_max_len", return_value=True):
            self.assertIs(D.N.DpPaddingMode.get_default_mode_in_cuda_graph(), D.N.DpPaddingMode.MAX_LEN)
        backend = SimpleNamespace(is_pplx=lambda: False, is_deepep_v2=lambda: False)
        with patch("sglang.srt.layers.moe.utils.get_moe_a2a_backend", return_value=backend), patch.object(D, "dp_gather_width", return_value=4), patch.object(D, "get_flags", return_value=SimpleNamespace(dp=SimpleNamespace(max_len_with_idle=True))):
            self.assertIs(D.N.DpPaddingMode.get_dp_padding_mode(False, [0, 0, 0, 3]), D.N.DpPaddingMode.MAX_LEN)

    def test_dp_local_index_and_cp_combine_use_source_ownership(self):
        from hcu_sglang.layers import dp_attention as D
        batch = SimpleNamespace(dp_local_start_pos=None, dp_local_num_tokens=None, dp_local_token_index=1, global_num_tokens_gpu=torch.tensor([2, 3, 1]))
        start, count = D.get_dp_local_info(batch)
        self.assertEqual(start.item(), 2)
        self.assertEqual(count.item(), 3)
        data = torch.arange(24).reshape(6, 4).float()
        out = torch.empty(2, 4)
        group = SimpleNamespace(attn_cp_size=2, attn_dp_rank=1)
        with patch.object(D, "get_parallel", return_value=group), patch.object(D, "_note_dp_gather_in_prefill_graph"), patch.object(D, "tensor_model_parallel_all_reduce", side_effect=lambda x: x * 2):
            D.dp_reduce_scatter_tensor(out, data)
        torch.testing.assert_close(out, data[2:4] * 2)

    def test_dp_aiter_equal_chunk_uses_unregistered_output(self):
        from hcu_sglang.layers import dp_attention as D
        x = torch.arange(16).reshape(4, 4).float()
        out = torch.empty(2, 4)
        ca = SimpleNamespace(disabled=False, should_custom_ar=lambda x: True, reduce_scatter=Mock(side_effect=lambda x, y, **kw: y.copy_(x[:2])))
        parallel = SimpleNamespace(tp_group=SimpleNamespace(ca_comm=ca), tp_size=2)
        with patch.object(D, "_dp_use_max_len", return_value=True), patch.object(D, "get_parallel", return_value=parallel):
            self.assertTrue(D._aiter_reduce_scatter_tensor(out, x))
            self.assertFalse(D._aiter_reduce_scatter_tensor(out[:1], x))
        self.assertEqual(ca.reduce_scatter.call_args.kwargs, {"registered": False})
        torch.testing.assert_close(out, x[:2])

    def test_non_hcu_megamoe_preserves_previous_wrappers(self):
        original = Mock(return_value=object())
        marker = object()
        with patch.object(M, "_IS_HCU", False):
            self.assertIs(M.around_forward_mega_moe(original, marker), original.return_value)
        original.assert_called_once_with(marker)


if __name__ == "__main__":
    unittest.main()
