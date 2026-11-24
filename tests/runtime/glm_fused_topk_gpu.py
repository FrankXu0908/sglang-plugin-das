import os
from types import SimpleNamespace
import pytest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.attention.current_dsa.dsa_topk_backend import DSATopKBackend, TopkTransformMethod


@pytest.mark.skipif(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1", reason="explicit GPU lane")
def test_existing_lightop_fused_topk_and_changed_graph():
    width = 4096
    score = torch.arange(width, device="cuda", dtype=torch.float32).repeat(2, 1)
    lens = torch.tensor([129, width], device="cuda", dtype=torch.int32)
    table = torch.arange(width, device="cuda", dtype=torch.int32).repeat(2, 1)
    table[1].add_(8192)
    meta = SimpleNamespace(page_table_1=table, token_to_batch_idx=torch.arange(2, device="cuda", dtype=torch.int32))
    cu = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)

    def run():
        return DSATopKBackend.SGL_KERNEL.topk_transform(score, lens, 2048, TopkTransformMethod.PAGED, meta, cu_seqlens_q_topk=cu)

    def check(out):
        values = score.cpu()
        lengths = lens.cpu().tolist()
        result = out.cpu()
        for i, n in enumerate(lengths):
            got = sorted(x for x in result[i].tolist() if x >= 0)
            want = sorted((torch.topk(values[i, :n], min(n, 2048)).indices + (8192 if i else 0)).tolist()) if n else []
            assert got == want

    for _ in range(3):
        check(run())
    stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    for lengths in ([0, 65], [2049, 4095], [7, 127]):
        lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        score.neg_()
        graph.replay();torch.cuda.synchronize();check(output)
