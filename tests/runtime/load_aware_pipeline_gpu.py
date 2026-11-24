from __future__ import annotations
from contextlib import ExitStack
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class LoadAwarePipelineGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def fixture(self):
        from hcu_sglang.eplb import integration as I,lplb_solver as L,expert_location_dispatch as O
        from sglang.srt.eplb import lplb_solver as N,expert_location_dispatch as D
        from sglang.srt import runtime_context as R
        from sglang.srt.model_executor.model_runner_components import moe_ep_setup
        from hcu_sglang.layers.moe import topk_source as T,hash_topk_source as H
        from sglang.srt.layers.moe import utils as U
        stack=ExitStack();mapping=torch.tensor([[[0,4],[1,5],[2,6],[3,7]]],device='cuda')
        physical=torch.tensor([[0,1,2,3,0,1,2,3]],device='cuda')
        metadata=S(physical_to_logical_map=physical,logical_to_all_physical_map=mapping,logical_to_all_physical_map_num_valid=torch.full((1,4),2,device='cuda'),num_layers=1,ep_size=2,logical_to_rank_dispatch_physical_map=None,num_physical_experts=8)
        config=S(moe=S(moe_a2a_backend='deepep',ep_dispatch_algorithm='load_aware',enable_waterfill=False,ep_num_redundant_experts=4,enable_eplb=True,init_expert_location='trivial'))
        parallel=S(moe_ep_size=2,moe_ep_rank=0,moe_ep_group=None,tp_group=None)
        resources=S(lplb_solvers={})
        for module,name,value in [(I,'get_global_expert_location_metadata',metadata),(O,'get_global_expert_location_metadata',metadata),(O,'get_exec',config),(O,'get_parallel',parallel),(R,'get_exec',config),(R,'get_parallel',parallel),(L,'get_resources',resources),(N,'get_resources',resources),(T,'get_exec',config),(H,'get_exec',config),(T,'get_parallel',parallel),(H,'get_parallel',parallel),(T,'get_server_args',object()),(T,'get_moe_a2a_backend',U.MoeA2ABackend.DEEPEP),(U,'get_moe_a2a_backend',U.MoeA2ABackend.DEEPEP),(T,'get_moe_runner_backend',U.MoeRunnerBackend.DEEP_GEMM),(T,'get_deepep_mode',U.DeepEPMode.LOW_LATENCY),(T,'is_allocation_symmetric',False)]:
            stack.enter_context(patch.object(module,name,return_value=value))
        recorder=S(on_select_experts=lambda **kw:None)
        for module in [T,H]:stack.enter_context(patch.object(module,'get_global_expert_distribution_recorder',return_value=recorder))
        moe_ep_setup.init_lplb_solvers(model_config=S(hf_config=S(architectures=['MiniMaxM2ForCausalLM'])))
        info=D.ExpertLocationDispatchInfo.init_new(0)
        self.assertIs(type(info),O.ExpertLocationDispatchInfo)
        self.assertEqual(info.source_rank,0)
        return stack,resources.lplb_solvers[0],info,physical[0]

    def route(self,hidden_states,gating_output,topk,renormalize):
        scores=gating_output.softmax(1);weights,ids=scores.topk(topk,dim=1)
        if renormalize:weights=weights/weights.sum(1,keepdim=True)
        return weights,ids.int()

    def test_native_topk_factory_solver_dispatch_pipeline_and_padding(self):
        from sglang.srt.layers.moe import topk as N
        stack,solver,info,physical=self.fixture()
        with stack:
            x=torch.ones(3,8,device='cuda');logits=torch.tensor([[1.,4.,2.,3.],[4.,3.,2.,1.],[2.,3.,4.,1.]],device='cuda')
            count=torch.tensor(2,device='cuda',dtype=torch.int32)
            cfg=N.TopKConfig(top_k=2,custom_routing_function=self.route,allow_routed_experts_capture=False)
            result=N.select_experts(x,logits,cfg,layer_id=0,num_token_non_padded=count,expert_location_dispatch_info=info);torch.cuda.synchronize()
            _,expected=self.route(x,logits,2,True)
            torch.testing.assert_close(physical[result.topk_ids[:2].long()],expected[:2].long(),rtol=0,atol=0)
            torch.testing.assert_close(result.topk_ids[2:],torch.full_like(result.topk_ids[2:],-1),rtol=0,atol=0)
            torch.testing.assert_close(solver._local_counts,torch.bincount(expected[:2].flatten().long(),minlength=4).float(),rtol=0,atol=0)

    def test_native_hash_solver_dispatch_pipeline_reads_real_valid_rows(self):
        from sglang.srt.layers.moe import hash_topk as N
        from sglang.srt.utils import is_hip
        stack,solver,info,physical=self.fixture()
        with stack,torch.device('cuda'):
            obj=N.HashTopK(2,4,0,3,layer_id=0)
            obj.tid2eid.copy_(torch.tensor([[0,1],[2,3],[0,0]],device='cuda',dtype=torch.int32))
            logits=torch.randn(3,4,device='cuda');inputs=torch.arange(3,device='cuda',dtype=torch.int64)
            count=torch.tensor(2,device='cuda',dtype=torch.int32)
            result=obj(torch.ones(3,8,device='cuda'),logits,inputs,num_token_non_padded=count,expert_location_dispatch_info=info);torch.cuda.synchronize()
            torch.testing.assert_close(physical[result.topk_ids[:2].long()],obj.tid2eid[:2].long(),rtol=0,atol=0)
            torch.testing.assert_close(solver._local_counts,torch.ones(4,device='cuda'),rtol=0,atol=0)
            # padded rows take the source's own branch: zero weights on HIP, masked ids otherwise
            # (an HCU host answers CUDA by default; adapter governance §0.5)
            if is_hip():torch.testing.assert_close(result.topk_weights[2:],torch.zeros_like(result.topk_weights[2:]),rtol=0,atol=0)
            else:torch.testing.assert_close(result.topk_ids[2:],torch.full_like(result.topk_ids[2:],-1),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
