import os
import pytest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.layers.attention.current_dsa import indexer as I

pytestmark = pytest.mark.skipif(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1", reason="explicit GPU lane")


@pytest.mark.parametrize("heads", [8, 32, 64])
def test_lightop_mqa_schedule_eager_and_changed_input_graph(heads):
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    device = "cuda"
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    q = torch.randn(2, heads, 128, device=device).clamp(-3, 3).to(torch.float8_e4m3fn)
    kval = torch.randn(4, 64, 1, 128, device=device).clamp(-3, 3).to(torch.float8_e4m3fn)
    scales = torch.rand(4, 64, 1, 1, device=device) * 0.04 + 0.01
    cache = torch.empty(4, 64, 1, 132, dtype=torch.uint8, device=device)
    packed = cache.view(4, -1)
    packed[:, :64 * 128].copy_(kval.view(torch.uint8).reshape(4, -1))
    packed[:, 64 * 128:].copy_(scales.view(torch.uint8).reshape(4, -1))
    weights = torch.rand(2, heads, device=device) / heads
    blocks = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]], dtype=torch.int32, device=device)
    lens = torch.tensor([63, 129], dtype=torch.int32, device=device)
    planner = I.resolve_paged_mqa_logits_metadata_fn()
    schedule = planner(lens.view(-1, 1), 64, cus)
    assert schedule.ndim == 2 and schedule.shape[1] == 2
    assert schedule.dtype == torch.int32

    def reference():
        k = kval.float() * scales
        ordered = k[blocks.long()].reshape(2, 256, 128)
        return torch.einsum("bht,bh->bt", torch.einsum("bhd,btd->bht", q.float(), ordered).relu(), weights)

    def verify(out):
        ref = reference()
        valid = torch.arange(256, device=device)[None, :] < lens[:, None]
        torch.testing.assert_close(out[valid].float(), ref[valid], rtol=0.03, atol=0.002)
        cos = torch.nn.functional.cosine_similarity(out[valid].float().view(1, -1), ref[valid].view(1, -1))
        assert cos.item() >= 0.9999

    for lengths in ([1, 65], [64, 127], [128, 255]):
        lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        schedule.copy_(planner(lens.view(-1, 1), 64, cus))
        verify(I._hcu_paged_mqa_logits(q, cache, weights, lens, blocks, schedule, 256))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            schedule.copy_(planner(lens.view(-1, 1), 64, cus))
            I._hcu_paged_mqa_logits(q, cache, weights, lens, blocks, schedule, 256)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        schedule.copy_(planner(lens.view(-1, 1), 64, cus))
        output = I._hcu_paged_mqa_logits(q, cache, weights, lens, blocks, schedule, 256)
    graph.replay()
    torch.cuda.synchronize()
    verify(output)
    for lengths, gain in (([7, 193], 0.7), ([65, 128], 1.2), ([255, 31], 0.8)):
        lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        q.copy_((q.float() * gain).to(q.dtype))
        weights.mul_(gain)
        graph.replay()
        torch.cuda.synchronize()
        verify(output)
