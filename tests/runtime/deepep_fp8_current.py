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
from hcu_sglang.layers.moe import deepep_fp8 as H
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import CompressedTensorsW8A8Fp8MoE

class DeepEPFp8Tests(unittest.TestCase):
    def test_packed_aliases_keep_shape_and_do_not_repeat_packing(self):
        H.register_hooks();scheme=object.__new__(CompressedTensorsW8A8Fp8MoE);layer=torch.nn.Module();layer.w13_weight=torch.nn.Parameter(torch.ones(2,16,128),requires_grad=False);layer.w2_weight=torch.nn.Parameter(torch.ones(2,128,8),requires_grad=False)
        with patch.object(H,'get_deepgemm_marlin_weight_pack_fn',return_value=lambda w:w.clone()*2) as packer:
            H._prepare_dsv4_channel_fp8_deepgemm_weights(scheme,layer);H._prepare_dsv4_channel_fp8_deepgemm_weights(scheme,layer)
        self.assertEqual(packer.call_count,1);self.assertFalse(hasattr(layer,'w13_weight'));self.assertFalse(hasattr(layer,'w2_weight'));self.assertEqual(layer._dsv4_w13_weight_shape,(2,16,128));self.assertEqual(layer._dsv4_w2_weight_shape,(2,128,8));self.assertIn('w13_weight_deepgemm',layer._non_persistent_buffers_set);self.assertEqual(layer.w13_weight_deepgemm.max().item(),2.)

    def test_masked_two_gemm_quant_dataflow_keeps_scales_and_clamp(self):
        import deepgemm
        q=torch.ones(2,3,128);s=torch.ones(2,3,1);ids=torch.zeros(1,2,dtype=torch.int64);weights=torch.full((1,2),.5);masked=torch.tensor([2,1],dtype=torch.int32)
        out=H.D.DeepEPLLDispatchOutput(q,s,ids,weights,masked,2);layer=SimpleNamespace(moe_runner_config=SimpleNamespace(activation='silu',swiglu_limit=3.),w13_weight_deepgemm=torch.ones(2,16,128),w2_weight_deepgemm=torch.ones(2,128,8),w13_weight_scale=torch.ones(2,16,1),w2_weight_scale=torch.ones(2,128,1));calls=[];a2=torch.ones(2,3,8);s2=torch.ones(2,3,1)
        def gemm(a,b,result,m,expected,**kw):
            self.assertIs(m,masked);self.assertEqual(expected,2);calls.append((a,b));result.fill_(len(calls))
        def quant(*,input,fp8type,tokens_per_expert,limit):
            self.assertEqual(input.shape,(2,3,16));self.assertIs(tokens_per_expert,masked);self.assertEqual(limit,3.);return a2,s2
        with patch.object(deepgemm,'m_grouped_fp8_gemm_nt_masked',side_effect=gemm),patch.object(H,'fuse_silu_mul_fp8_quant_ep',side_effect=quant):result=H.forward_groupgemm_w8a8_fp8_masked(layer,out)
        self.assertEqual(len(calls),2);self.assertIs(calls[0][0][0],q);self.assertIs(calls[0][0][1],s);self.assertIs(calls[1][0][0],a2);self.assertIs(calls[1][0][1],s2);self.assertEqual(result.shape,(2,3,128));self.assertEqual(result.max().item(),2.)

    def test_masked_compacts_strided_scales_at_both_gemm_boundaries(self):
        import deepgemm
        q = torch.ones(2, 3, 128)
        scale = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)[..., :1]
        scale2 = scale + 1
        scale2 = torch.stack((scale2, scale2), dim=-1)[..., 0]
        self.assertFalse(scale.is_contiguous())
        self.assertFalse(scale2.is_contiguous())
        expected_scales = [scale.clone(), scale2.clone()]
        masked = torch.tensor([2, 1], dtype=torch.int32)
        packet = H.D.DeepEPLLDispatchOutput(
            q, scale, torch.zeros(1, 2, dtype=torch.int64),
            torch.ones(1, 2), masked, 2,
        )
        layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(activation="silu", swiglu_limit=None),
            w13_weight_deepgemm=torch.ones(2, 16, 128),
            w2_weight_deepgemm=torch.ones(2, 128, 8),
            w13_weight_scale=torch.ones(2, 16, 1),
            w2_weight_scale=torch.ones(2, 128, 1),
        )
        calls = []

        def gemm(a, b, result, counts, expected, **kwargs):
            self.assertTrue(a[1].is_contiguous(), "a_scale must be contiguous")
            torch.testing.assert_close(a[1], expected_scales[len(calls)], rtol=0, atol=0)
            self.assertIs(counts, masked)
            self.assertEqual(expected, 2)
            calls.append(a)
            result.fill_(len(calls))

        with patch.object(deepgemm, "m_grouped_fp8_gemm_nt_masked", side_effect=gemm), patch.object(
            H, "fuse_silu_mul_fp8_quant_ep",
            return_value=(torch.ones(2, 3, 8), scale2),
        ):
            result = H.forward_groupgemm_w8a8_fp8_masked(layer, packet)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.shape, (2, 3, 128))
        self.assertEqual(result.max().item(), 2)
        torch.testing.assert_close(scale, expected_scales[0], rtol=0, atol=0)
        torch.testing.assert_close(scale2, expected_scales[1], rtol=0, atol=0)

    def test_combine_packet_retains_router_ids_and_weights(self):
        q=torch.ones(2,3,8);s=torch.ones(2,3,1);ids=torch.zeros(1,2,dtype=torch.int64);weights=torch.full((1,2),.5);out=H.D.DeepEPLLDispatchOutput(q,s,ids,weights,torch.ones(2,dtype=torch.int32),2);value=torch.ones_like(q,dtype=torch.bfloat16)
        with patch.object(H,'_use_hcu_legacy_channel_fp8_deepep',return_value=True),patch.object(H,'forward_groupgemm_w8a8_fp8_masked',return_value=value):r=H.around_run_moe_core(lambda *a:None,object(),out)
        self.assertIsInstance(r,H.D.DeepEPLLCombineInput);self.assertIs(r.hidden_states,value);self.assertIs(r.topk_ids,ids);self.assertIs(r.topk_weights,weights)

    def test_normal_two_gemm_quant_scatter_and_gather(self):
        x=torch.ones(2,128,dtype=torch.int8);scale=torch.ones(2,1);ids=torch.zeros(2,2,dtype=torch.int64);weights=torch.full((2,2),.5);counts=[256,256];packet=H.D.DeepEPNormalDispatchOutput(x,scale,ids,weights,counts)
        layer=SimpleNamespace(quant_method=object(),moe_runner_config=SimpleNamespace(activation='silu',swiglu_limit=3.),_dsv4_w13_weight_shape=(2,16,128),w13_weight_deepgemm=torch.ones(2,16,128),w2_weight_deepgemm=torch.ones(2,128,8),w13_weight_scale=torch.ones(2,16,1),w2_weight_scale=torch.ones(2,128,1));midx=torch.zeros(512,dtype=torch.int32);index=torch.zeros_like(ids,dtype=torch.int32);calls=[]
        def scatter(*args,**kwargs):self.assertIs(args[0],x);self.assertIs(args[1],scale);self.assertTrue(kwargs['counts_are_aligned']);return midx,index
        def gemm(a,b,out,mi):self.assertIs(mi,midx);calls.append((a,b));out.fill_(len(calls))
        def quant(t,*,fp8type,limit):self.assertEqual(t.shape,(512,16));self.assertEqual(limit,3.);return torch.ones(512,8),torch.ones(512,1)
        def gather(down,topids,topweights,idx,out):self.assertIs(topids,ids);self.assertIs(topweights,weights);self.assertIs(idx,index);out.copy_(down[:2])
        with patch.object(H,'get_offloader',return_value=SimpleNamespace(forbid_copy_engine_usage=True)),patch.object(H,'copy_list_to_gpu_no_ce',return_value=torch.tensor(counts,dtype=torch.int32)) as copier,patch.object(H,'_ep_scatter_with_optional_lightop',side_effect=scatter),patch.object(H,'m_grouped_fp8_gemm_nt_contiguous',side_effect=gemm),patch.object(H,'fuse_silu_mul_fp8_quant',side_effect=quant),patch.object(H,'_ep_gather_with_optional_lightop',side_effect=gather):result=H.forward_groupgemm_w8a8_fp8_contiguous(layer,packet)
        copier.assert_called_once_with(counts);self.assertEqual(len(calls),2);self.assertEqual(result.shape,(2,128));self.assertEqual(result.max().item(),2.)

    def test_owned_functions_have_real_global_providers(self):
        import builtins,symtable
        from pathlib import Path
        from hcu_sglang.layers.moe import deepep_ep_kernels as K
        missing=set()
        module=H
        def inspect_scope(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in vars(module) and symbol.get_name() not in vars(builtins):missing.add(symbol.get_name())
            for child in scope.get_children():inspect_scope(child)
        for module in (H,K):
            tree=symtable.symtable(Path(module.__file__).read_text(),module.__file__,'exec')
            for child in tree.get_children():inspect_scope(child)
        self.assertEqual(missing,set())

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_real_hcu_scatter_gather_and_indices(self):
        from hcu_sglang.layers.moe import deepep_ep_kernels as K
        x=(torch.arange(256,device='cuda').reshape(2,128)%127).to(torch.int8);scale=torch.tensor([[.25],[.5]],device='cuda');ids=torch.tensor([[0,1],[1,-1]],device='cuda',dtype=torch.int64);weights=torch.tensor([[.25,.75],[1.,0.]],device='cuda');counts=torch.tensor([256,256],device='cuda',dtype=torch.int32);actual=torch.tensor([1,2],device='cuda',dtype=torch.int32);loc=torch.zeros_like(counts);qx=torch.empty(512,128,device='cuda',dtype=x.dtype);qs=torch.empty(512,1,device='cuda');mi=torch.full((512,),-1,device='cuda',dtype=torch.int32);index=torch.full(ids.shape,-1,device='cuda',dtype=torch.int32)
        K.ep_scatter(x,scale,ids,counts,actual,loc,qx,qs,mi,index,expert_alignment=256)
        values=(qx.float()*qs).bfloat16();out=torch.empty(2,128,device='cuda',dtype=torch.bfloat16);K.ep_gather(values,ids,weights,index,out);torch.cuda.synchronize();ref=(x.float()*scale).bfloat16();torch.testing.assert_close(out,ref,rtol=0,atol=0)
        indices=K.build_m_indices_triton(ids,torch.device('cuda'),2);expected=torch.full((512,),-1,device='cuda',dtype=torch.int32);expected[0]=0;expected[256:258]=1;torch.testing.assert_close(indices,expected,rtol=0,atol=0);print('HCU EP scatter/gather max abs error:',(out.float()-ref.float()).abs().max().item())

if __name__=='__main__':unittest.main()
