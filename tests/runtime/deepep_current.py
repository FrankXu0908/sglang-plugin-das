# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.moe import deepep_current as D

class DeepEPCurrentTests(unittest.TestCase):
    def test_vendor_abi_keeps_weights_quant_type_and_scale_contract(self):
        x=torch.ones(2,8);ids=torch.tensor([[0,1],[1,0]]);weights=torch.full((2,2),.5);seen=[]
        def dispatch(**kw):seen.append(kw);return kw
        b=SimpleNamespace(low_latency_dispatch=dispatch)
        with patch.object(D,'_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE',True):
            for fp8,ue8m0,qt,group in [(False,False,0,0),(True,False,2,0),(True,True,3,128)]:
                out=D._low_latency_dispatch_compat(b,x,ids,weights,128,256,use_fp8=fp8,use_nvfp4=False,input_global_scale=None,round_scale=False,use_ue8m0=ue8m0,async_finish=True,return_recv_hook=False)
                self.assertIs(out['x'],x);self.assertIs(out['topk_weight'],weights);self.assertEqual(out['quant_type'],qt);self.assertEqual(out['quant_group_size'],group)
            with self.assertRaises(RuntimeError):D._low_latency_dispatch_compat(b,x,ids,weights,128,256,use_fp8=False,use_nvfp4=True,input_global_scale=None,round_scale=False,use_ue8m0=False,async_finish=True,return_recv_hook=False)

    def test_channel_fp8_dispatch_requests_per_token_scales_through_bridge(self):
        calls = []
        packet = (object(), object(), object(), object(), object())
        buffer = SimpleNamespace(low_latency_dispatch=lambda *args, **kwargs: calls.append(kwargs) or packet)
        worker = SimpleNamespace(
            quant_config={"hcu_legacy_groupgemm": True}, use_fp8=True,
            use_nvfp4=False, return_recv_hook=False,
            num_max_dispatch_tokens_per_rank=128, num_experts=192,
            _get_buffer=lambda: buffer,
            _get_npu_mxfp_quantization_kwargs=lambda _: {},
        )
        with patch.object(D, "_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE", False), patch.object(
            D, "_deepep_precompile_tp_barrier"
        ), patch.object(D, "_is_hcu", True):
            D._DeepEPDispatcherImplLowLatency._dispatch_core(
                worker, torch.ones(1, 4096), torch.zeros(1, 8), torch.ones(1, 8)
            )
        self.assertEqual(calls[0]["quant_group_size"], 0)
        self.assertTrue(calls[0]["use_fp8"])
        self.assertFalse(calls[0]["use_ue8m0"])

    def test_hcu_deepgemm_requires_selected_backend_and_non_slimquant(self):
        for available,selected,quant,expected in [(True,True,'compressed-tensors',True),(True,False,'compressed-tensors',False),(False,True,'compressed-tensors',False),(True,True,'slimquant_w4a8_marlin',False)]:
            with patch.object(D,'_hcu_deepgemm_available',available),patch.object(D,'get_moe_runner_backend',return_value=SimpleNamespace(is_deep_gemm=lambda:selected)),patch.object(D,'get_model',return_value=SimpleNamespace(quantization=quant)):
                self.assertEqual(D._hcu_deepgemm_is_selected(),expected)

    def test_public_dispatcher_selects_owned_normal_and_low_latency(self):
        from sglang.srt.plugins.hook_registry import HookRegistry
        import sglang.srt.layers.moe.token_dispatcher.deepep as native
        HookRegistry.reset();D.register_hooks();HookRegistry.apply_hooks()
        layer=object.__new__(native.DeepEPDispatcher);layer.deepep_mode=D.DeepEPMode.AUTO;layer._normal_dispatcher=object();layer._low_latency_dispatcher=object()
        with patch.object(D,'get_is_extend_in_batch',return_value=True):self.assertIs(layer._get_impl(),layer._normal_dispatcher)
        with patch.object(D,'get_is_extend_in_batch',return_value=False):self.assertIs(layer._get_impl(),layer._low_latency_dispatcher)
        self.assertIs(D.DeepEPLLDispatchOutput,native.DeepEPLLDispatchOutput)

    def test_target_and_draft_reserve_one_buffer_for_max_layout(self):
        resources=SimpleNamespace(buffers={});group=SimpleNamespace(size=lambda:4)
        def hint(tokens,hidden,ranks,experts,num_topk=8):return tokens*hidden*experts*num_topk
        with patch.object(D,'get_resources',return_value=resources),patch.object(D.Buffer,'get_low_latency_rdma_size_hint',side_effect=hint):
            D.DeepEPBuffer.register_low_latency_layout(group,128,64,256,8);state=D.DeepEPBuffer._state();size=state.low_latency_rdma_bytes
            D.DeepEPBuffer.register_low_latency_layout(group,128,64,64,4)
            self.assertIs(D.DeepEPBuffer._state(),state);self.assertEqual(state.low_latency_rdma_bytes,size);self.assertEqual(state.low_latency_num_qps,64)

if __name__=='__main__':unittest.main()
