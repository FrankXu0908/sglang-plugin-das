# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import sys
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.models.qwen import qwen2 as Q2, qwen3 as Q3, qwen3_source as S3
from hcu_sglang.models.deepseek.deepseek_common.methods import weight_loader as WL

class AttentionModelParityTests(unittest.TestCase):
    def test_hy3_int8_rope_matches_legacy_cache_precision(self):
        from hcu_sglang.models import hunyuan_v3 as H
        from sgl_kernel.rotary import rotary_embedding

        cache = torch.tensor([[0.12345, 0.98765, 0.45678, 0.76543],
                              [0.87654, 0.23456, 0.65432, 0.34567]])
        values = torch.arange(24, dtype=torch.bfloat16).reshape(2, 12) / 7
        positions = torch.tensor([0, 1])
        class Rotary:
            is_neox_style = True
            def __init__(self):
                self.cos_sin_cache = cache.clone()
            def __call__(self, positions, q, k):
                rotary_embedding(positions, q, k, 4, self.cos_sin_cache, True)
                return q, k
        rotary = Rotary()
        original_cache = rotary.cos_sin_cache
        for hcu, dtype in ((True, torch.int8), (True, torch.bfloat16),
                           (True, torch.float8_e4m3fn), (False, torch.int8)):
            with self.subTest(hcu=hcu, weight_dtype=dtype):
                x = values.clone()
                qkv = Mock(return_value=(x, None))
                qkv.weight = torch.empty(1, dtype=dtype)
                model = NS(hy3_sp=False, qkv_proj=qkv, use_hpc_ops_fp8_attn=False,
                           q_size=4, kv_size=4, use_qk_norm=False, head_dim=4,
                           rotary_emb=rotary, attn=Mock(side_effect=lambda q,k,v,b,**kw:q),
                           o_proj=lambda value:(value, None))
                trig = cache.to(torch.bfloat16).float() if hcu and dtype == torch.int8 else cache
                a, b = values[:, :2].float(), values[:, 2:4].float()
                cos, sin = trig[:, :2], trig[:, 2:]
                expected = torch.cat((a*cos-b*sin, a*sin+b*cos), dim=-1).to(torch.bfloat16)
                with patch.object(H, '_is_hcu', hcu), patch.object(H, '_use_fused_hunyuan_rotary', False):
                    actual = H.HYV3Attention.forward(model, positions, x, NS())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertIs(rotary.cos_sin_cache, original_cache)
                torch.testing.assert_close(rotary.cos_sin_cache, cache, rtol=0, atol=0)

    def test_hy3_int8_rope_reuses_private_cache_and_refreshes_replaced_source(self):
        from hcu_sglang.models import hunyuan_v3 as H

        from sgl_kernel.rotary import rotary_embedding

        class Rotary:
            is_neox_style = True
            cos_sin_cache = torch.tensor([[0.12345, 0.98765, 0.45678, 0.76543]])

        rotary = Rotary()
        qkv = Mock()
        qkv.weight = torch.empty(1, dtype=torch.int8)
        model = NS(hy3_sp=False, qkv_proj=qkv, use_hpc_ops_fp8_attn=False,
                   q_size=4, kv_size=4, use_qk_norm=False, head_dim=4,
                   rotary_emb=rotary, attn=Mock(side_effect=lambda q,k,v,b,**kw:q),
                   o_proj=lambda value:(value, None))
        previous = None
        for dtype, replace_source in ((torch.bfloat16, False), (torch.bfloat16, False),
                                      (torch.float16, False), (torch.float16, True)):
            if replace_source:
                rotary.cos_sin_cache = rotary.cos_sin_cache + 0.01
            source = rotary.cos_sin_cache
            x = torch.arange(12, dtype=dtype).reshape(1, 12) / 7
            qkv.return_value = (x.clone(), None)
            expected_q, expected_k = x[:, :4].clone(), x[:, 4:8].clone()
            rotary_embedding(torch.tensor([0]), expected_q, expected_k, 4,
                             source.to(dtype=dtype), True)
            with patch.object(H, '_is_hcu', True), patch.object(H, '_use_fused_hunyuan_rotary', False):
                actual = H.HYV3Attention.forward(model, torch.tensor([0]), x, NS())
            torch.testing.assert_close(actual, expected_q, rtol=0, atol=0)
            torch.testing.assert_close(model.attn.call_args.args[1], expected_k, rtol=0, atol=0)
            self.assertIs(rotary.cos_sin_cache, source)
            cached = rotary._hy3_int8_cos_sin_cache
            self.assertIs(cached[0], source)
            self.assertEqual(cached[1].dtype, dtype)
            if previous is not None:
                if previous[0] is source and previous[1].dtype == dtype:
                    self.assertIs(cached, previous)
                else:
                    self.assertIsNot(cached, previous)
            previous = cached

    def test_qwen2_fusion_leaves_cache_write_to_attention(self):
        calls=[]
        x=torch.zeros(2,8);kv=torch.zeros(4,1,2)
        def fused(*a, **kw):
            calls.append(kw)
            return torch.ones(2,4),torch.ones(2,2),torch.ones(2,2)
        attn=NS(layer_id=2,head_dim=2,q_size=4,kv_size=2,page_size=64,kv_cache_dtype=torch.bfloat16,
                qkv_proj=lambda x:(x,None),rotary_emb=NS(cos_sin_cache=x),o_proj=lambda x:(x,None),
                attn=lambda q,k,v,b,**kw:(calls.append(kw) or q))
        batch=NS(out_cache_loc=torch.tensor([1,2]))
        with patch.object(Q2,'is_hcu',return_value=True),patch.object(Q2,'get_bool_env_var',return_value=True),patch.object(Q2,'get_token_to_kv_pool',return_value=NS(get_kv_buffer=lambda layer:(kv,kv))),patch.dict(sys.modules,{'lightop':NS(split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant=fused)}):
            out=Q2.around_forward(lambda *a: self.fail('unexpected fallback'),attn,None,x,batch)
        self.assertIsNone(calls[0]['kv_cache_loc']);self.assertEqual(calls[1],{})
        self.assertEqual(tuple(out.shape),(2,2,2))
        self.assertFalse(hasattr(batch,'_hcu_fused_rms_rotary_kv_layers'))

    def test_qwen3_constructor_installs_1d_path_and_keeps_cpu_scales(self):
        def linear(*args,**kwargs):return NS()
        class Norm:
            def __init__(self,size,eps,**kw):self.weight=torch.ones(size);self.variance_epsilon=eps
        class Rotary:
            def __init__(self):self.rotary_dim=64
        attn=torch.nn.Module()
        with patch('hcu_sglang.utils.detection.is_hcu',return_value=True):
            with patch.object(S3,'get_parallel',return_value=NS(attn_tp_size=2)),patch.object(S3,'get_exec',return_value=NS(deterministic=NS(rl_on_policy_target=None))),patch.object(S3,'get_server_args',return_value=NS(kv_cache_dtype='bf16')),patch.object(S3,'RMSNorm',Norm),patch.object(S3,'QKVParallelLinear',linear),patch.object(S3,'RowParallelLinear',linear),patch.object(S3,'RadixAttention',linear),patch.object(S3,'RotaryEmbedding',Rotary),patch.object(S3,'get_rope',return_value=Rotary()),patch.object(S3,'_has_fused_qk_norm_rope',True):
                Q3.around_init(lambda *a: self.fail('HCU constructor delegated'),attn,256,4,2,rms_norm_eps=1e-5)
        self.assertTrue(attn.use_fused_qk_norm_rope)
        self.assertEqual(attn.num_heads,2);self.assertEqual(attn.num_kv_heads,1)
        self.assertEqual(attn._fused_k_scale.device.type,'cpu')
        self.assertIsNot(attn._fused_k_scale,attn._fused_v_scale)
        self.assertIs(attn.forward_prepare_native.__func__,S3.Qwen3Attention.forward_prepare_native)

    def test_qwen3_forward_selects_1d_decode_only_for_nhd_matching_dtype(self):
        calls=[]
        x=torch.zeros(2,128)
        pool=NS(get_kv_buffer=lambda layer:(torch.empty(4,1,64),torch.empty(4,1,64)))
        model=NS(use_fused_qk_norm_mrope=False,use_fused_qk_norm_rope=True,attn=lambda *a,**kw:(calls.append(kw) or x),o_proj=lambda y:(y,None))
        model.attn=type('Attention',(),{'layer_id':1,'__call__':lambda self,*a,**kw:(calls.append(kw) or x)})()
        model.forward_prepare_aiter_fused=lambda *a:(calls.append('aiter') or (x,None,None))
        model.forward_prepare_native=lambda **kw:(calls.append('native') or (x,x,x))
        batch=NS(forward_mode=NS(is_decode=lambda:True))
        with patch.object(S3,'get_exec',return_value=NS(deterministic=NS(rl_on_policy_target=None))),patch.object(S3,'get_token_to_kv_pool',return_value=pool):
            S3.Qwen3Attention.forward(model,None,x,batch)
        self.assertEqual(calls,['aiter',{'save_kv_cache':False}])
        calls.clear();pool.get_kv_buffer=lambda layer:(torch.empty(4,1,64,dtype=torch.bfloat16),None)
        with patch.object(S3,'get_exec',return_value=NS(deterministic=NS(rl_on_policy_target=None))),patch.object(S3,'get_token_to_kv_pool',return_value=pool),patch.object(S3,'_is_npu',False):
            S3.Qwen3Attention.forward(model,None,x,batch)
        self.assertEqual(calls,['native',{'save_kv_cache':True}])

    def test_mla_fp8_padding_and_dense_scheduler_object(self):
        from hcu_sglang.layers.attention import hcu_mla_backend as MLA, hcu_mla_utils as MU
        with patch.object(MU,'is_hcu',return_value=True):
            self.assertEqual(MU.get_hcu_mla_fp8_kv_cache_dim(512,0,torch.float8_e4m3fn,True),576)
            for d,r,t,h in [(512,64,torch.float8_e4m3fn,True),(512,0,torch.bfloat16,True),(512,0,torch.float8_e4m3fn,False)]:
                self.assertEqual(MU.get_hcu_mla_fp8_kv_cache_dim(d,r,t,h),512)
        class Meta: pass
        with patch.object(MLA,'FlashMLASchedMeta',Meta):
            meta,splits=MLA._dense_flashmla_sched_meta(torch.ones(2),torch.ones(1))
            self.assertIsInstance(meta,Meta);self.assertIsNone(splits)
            self.assertIs(MLA._dense_flashmla_sched_meta(meta,None)[0],meta)

    def test_weight_postload_uses_local_pp_range_and_skips_non_mla_layers(self):
        w=torch.arange(12,dtype=torch.float32).reshape(6,2)
        sa=NS(kv_b_proj=NS(weight=w),qk_nope_head_dim=2,v_head_dim=1,w_kc=None,w_vc=None)
        model=NS(model=NS(start_layer=2,end_layer=4,layers={2:NS(self_attn=sa),3:NS(self_attn=NS())}),config=NS(num_hidden_layers=8,architectures=[]),quant_config=None)
        with patch.object(WL,'_use_aiter_gfx95',False),patch.object(WL,'_is_cuda',False),patch.object(WL,'_is_hip',False),patch.object(WL,'_is_musa',False),patch.object(WL,'_is_npu',False),patch.object(WL,'_is_xpu',False):
            WL.post_load_weights(model,weight_names=['model.layers.0.self_attn.kv_b_proj.weight','model.layers.2.self_attn.kv_b_proj.weight','model.layers.3.self_attn.kv_b_proj.weight','model.layers.7.self_attn.kv_b_proj.weight'])
        self.assertTrue(torch.equal(sa.w_kc,w.unflatten(0,(-1,3))[:,:2]))


class GdnParityTests(unittest.TestCase):
    def test_prefill_only_uses_hcu_conv_without_block_table(self):
        from hcu_sglang.layers.attention.linear import gdn_source as G
        calls=[]
        def conv(*args,**kwargs):
            calls.append(kwargs)
            return args[0]
        backend=NS(forward_metadata=NS(conv_block_table=None,query_start_loc=torch.tensor([0,2])))
        layer=NS(conv_weights=torch.ones(4,2),bias=None,activation='silu')
        batch=NS(extend_prefix_lens=torch.tensor([0]),extend_seq_lens_cpu=[2])
        with patch.object(G,'_use_hcu_causal_conv1d',True),patch.object(G,'causal_conv1d_fn_hcu',conv,create=True),patch.object(G,'causal_conv1d_fn',conv):
            G._convolve_prefill(backend,layer,batch,torch.ones(2,4),torch.ones(1,4,2),torch.tensor([0]))
            self.assertIn('initial_states',calls[-1]);self.assertNotIn('conv_states',calls[-1])
            backend.forward_metadata.conv_block_table=torch.tensor([[0]])
            G._convolve_prefill(backend,layer,batch,torch.ones(2,4),torch.ones(1,4,2),torch.tensor([0]))
            self.assertIn('conv_states',calls[-1]);self.assertIn('block_table',calls[-1]);self.assertNotIn('initial_states',calls[-1])


class TritonAndMultimodalParityTests(unittest.TestCase):
    def test_aiter_extend_uses_current_named_provider_contract(self):
        from hcu_sglang.layers.attention import triton_hcu as T
        from sglang.srt.layers.radix_attention import AttentionType
        calls=[]
        x=torch.ones(2,4)
        md=NS(window_kv_offsets=None,kv_indptr=torch.tensor([0,2]),kv_indices=torch.tensor([0,1]),qo_indptr=torch.tensor([0,2]),custom_mask=None,mask_indptr=None,max_extend_len=2)
        backend=NS(_can_run_dense_fp8_chunked_mha=lambda *a:False,token_to_kv_pool=NS(get_key_buffer=lambda layer:x,get_value_buffer=lambda layer:x),forward_metadata=md,dcp_size=1,enable_deterministic=False,use_aiter_triton_extend_fwd=True,extend_attention_fwd=lambda **kw:(calls.append(kw) or kw['o_extend'].fill_(3)))
        layer=NS(layer_id=0,qk_head_dim=2,v_head_dim=2,tp_q_head_num=2,k_scale=None,logit_capping_method='tanh',logit_cap=0,is_cross_attention=False,attn_type=AttentionType.DECODER,sliding_window_size=-1,scaling=.5,xai_temperature_len=-1)
        batch=NS(mha_one_shot=False,attn_attend_prefix_cache=True)
        out=T.forward_extend(backend,x,x,x,layer,batch,save_kv_cache=False)
        self.assertEqual(out.sum().item(),24)
        self.assertIn('q_extend',calls[0]);self.assertIn('sm_scale',calls[0]);self.assertNotIn('page_size',calls[0])

    def test_multimodal_sla_admission_and_fa2_selection(self):
        from hcu_sglang.multimodal_gen import attention as A
        with patch.object(A,'is_hcu',return_value=True),patch.object(A.importlib.util,'find_spec',return_value=object()):
            self.assertIn('FlashAttention2Backend',A.around_rocm_backend_selection(lambda *a:'fallback',None,NS(name='FA2'),128,torch.bfloat16))
            with self.assertRaises(ValueError):A.around_rocm_backend_selection(lambda *a:'fallback',None,NS(name='SLA_ATTN'),256,torch.bfloat16)
            with self.assertRaises(ValueError):A.around_rocm_backend_selection(lambda *a:'fallback',None,NS(name='SLA_ATTN'),128,torch.float32)


class VisionRoutingParityTests(unittest.TestCase):
    def test_cpu_vision_bias_and_normalization(self):
        from hcu_sglang.multimodal.dsv41 import vl_routing as V
        cfg=NS(num_fused_shared_experts=0,top_k=2,renormalize=True,routed_scaling_factor=1.5,apply_routed_scaling_factor_on_output=True,fused_shared_experts_scaling_factor=None)
        bias=torch.tensor([0.,0.,0.,2.]);vl=torch.tensor([3.,0.,0.,0.])
        moe=NS(topk=NS(topk_config=cfg),gate=NS(e_score_correction_bias=bias,e_score_correction_bias_vl=vl),config=NS(image_token_id=99))
        logits=torch.tensor([[1.,2.,3.,4.],[1.,2.,3.,4.]])
        with patch.object(V,'get_moe_a2a_backend',return_value=NS(is_megamoe=lambda:False)),patch.object(V,'is_cuda',return_value=False),patch.object(V,'_use_lightop_sqrtsoftplus_v41_gate',False):
            out=V.vision_topk(moe,logits,torch.tensor([99,1]))
        score=torch.nn.functional.softplus(logits).sqrt()
        expected_idx=(score+torch.stack([vl,bias])).topk(2,dim=-1).indices
        self.assertTrue(torch.equal(out.topk_ids.long(),expected_idx))
        torch.testing.assert_close(out.topk_weights.sum(-1),torch.tensor([1.5,1.5]))


class GlmModelParityTests(unittest.TestCase):
    def test_glm_expert_consumer_keeps_original_routed_scaling_field(self):
        from hcu_sglang.models import glm4_moe as G, glm4_moe_source as S
        seen=[];x=torch.ones(2,4)
        moe=NS(routed_scaling_factor=2.5,_forward_shared_experts=lambda x:None,gate=lambda x:x,topk=lambda x,l:l)
        moe.experts=lambda x,k:(seen.append(moe.routed_scaling_factor) or (x*2.5))
        with patch.object(G,'is_hcu',return_value=True),patch.object(S,'_is_hcu',True):
            out=G.around_forward_normal(lambda *a:self.fail('HCU fallback'),moe,x)
        self.assertEqual(seen,[2.5]);self.assertEqual(moe.routed_scaling_factor,2.5)
        torch.testing.assert_close(out,x*2.5)
        mlp=NS(gate_up_proj=NS(weight=torch.ones(2,2).to(torch.float8_e4m3fn),quant_method=NS()))
        G.after_mlp_init(None,mlp)
        self.assertFalse(hasattr(mlp.gate_up_proj.quant_method,'quant_config'))


class BoundaryParityTests(unittest.TestCase):
    def test_dsa_mixed_positional_optional_kwargs_preserve_scale_storage(self):
        from hcu_sglang.layers.attention import dsa_kernels as D
        k=torch.ones(2,3,4);scale=torch.ones(2,3,1)
        with patch.object(D,'is_hcu',return_value=True):
            args,kw=D.before_set_k_and_s_triton(None,None,k,scale,extra=True)
            self.assertEqual(tuple(args[2].shape),(6,4));self.assertEqual(tuple(args[3].shape),(6,1));self.assertEqual(kw,{'extra':True})
            self.assertEqual(args[2].data_ptr(),k.data_ptr())
            args,kw=D.before_set_k_and_s_triton(None,None,k,index_k_scale=scale,extra=True)
            self.assertEqual(tuple(args[2].shape),(6,4));self.assertEqual(tuple(kw['index_k_scale'].shape),(6,1))

    def test_bailing_fused_projection_requires_opt_in_and_keeps_batch(self):
        from hcu_sglang.models import bailing_moe as B
        calls=[];x=torch.ones(2,4);batch=object()
        model=NS(is_replicated=True,gate_up_proj=lambda x:(x,None),down_proj=lambda x,**kw:(calls.append(kw) or (x,None)))
        with patch.object(B,'is_hcu',return_value=True),patch.object(B,'get_bool_env_var',return_value=False):
            self.assertEqual(B.replacement_mlp_forward(lambda *a:'baseline',model,x,batch),'baseline')
        with patch.object(B,'is_hcu',return_value=True),patch.object(B,'get_bool_env_var',return_value=True):
            B.replacement_mlp_forward(lambda *a:self.fail('enabled fusion delegated'),model,x,batch)
        self.assertIs(calls[0]['forward_batch'],batch);self.assertTrue(calls[0]['use_fused_silu_mul_fp8_quant'])

class SpeculativeDisaggregationParityTests(unittest.TestCase):
    def test_hidden_streaming_plan_validates_pp_ownership_and_capacity(self):
        from hcu_sglang import speculative_disaggregation as D
        req=NS(origin_input_ids=list(range(8)),rid='request')
        model_config=NS(hidden_size=4)
        runner=NS(spec_aux_config=NS(dflash_target_layer_ids=[1,3]))
        pool=NS(size=3)
        metadata={'hidden_start':2,'hidden_len':6,'decode_radix_cache_enabled':True,'streaming_hidden':True,'target_layer_ids':[1,3],'pp_slices':{'0':{'layer_ids':[1],'slice_len':4,'dst_indices':[7,8,9]}}}
        kwargs=dict(req=req,metadata=metadata,decode_prefix_len=2,pp_rank=0,model_config=model_config,model_runner=runner,metadata_buffers=NS(pd_hidden_pool=pool),prefill_radix_enabled=True)
        plan,error=D.resolve_hidden_bootstrap_plan(**kwargs)
        self.assertIsNone(error);self.assertIs(plan.pool,pool);self.assertEqual(plan.source_window_rows,3);self.assertEqual(plan.local_layer_ids,[1])
        pool.size=2
        plan,error=D.resolve_hidden_bootstrap_plan(**kwargs)
        self.assertIsNone(plan);self.assertIn('capacity',error)
        kwargs['pp_rank']=1
        plan,error=D.resolve_hidden_bootstrap_plan(**kwargs)
        self.assertIsNone(plan);self.assertIn('missing PP slice',error)
        kwargs['pp_rank']=0;kwargs['prefill_radix_enabled']=False
        plan,error=D.resolve_hidden_bootstrap_plan(**kwargs)
        self.assertIsNone(plan);self.assertIn('matching prefill/decode radix',error)


class AttentionBaseContractTests(unittest.TestCase):
    def test_source_base_defaults_are_visible_to_existing_derived_classes(self):
        from hcu_sglang.layers.attention import base_backend as B
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
        from sglang.srt.plugins.hook_registry import HookRegistry
        class Existing(AttentionBackend):
            pass
        with patch.object(B,'is_hcu',return_value=True),patch.object(HookRegistry,'register'):
            B.register_hooks()
        self.assertIsNone(Existing().late_layer_num_tokens(NS()))
        self.assertIsNone(Existing().get_fused_kv_write_buffers(NS(),NS()))
        class Specialized(Existing):
            def late_layer_num_tokens(self, batch):return 7
        self.assertEqual(Specialized().late_layer_num_tokens(NS()),7)

    def test_glm5_next_hcu_disables_generic_dsa_graph_variants(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import graph_variants as G
        from hcu_sglang.models.deepseek.deepseek_common import model_types as M
        with patch('hcu_sglang.utils.detection.is_hcu',return_value=True),patch.object(M,'is_glm5_next_hcu',return_value=True):
            self.assertIsNone(G.create_attention_graph_variants(NS()))

if __name__=='__main__':unittest.main()
