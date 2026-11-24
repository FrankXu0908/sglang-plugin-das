import os
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import rotary as R


@pytest.mark.skipif(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1", reason="explicit GPU lane")
@pytest.mark.parametrize("neox", [False, True])
def test_existing_lightop_rope_selection_eager_and_changed_graph(neox):
    torch.cuda.set_device(0)
    torch.manual_seed(31)
    t = torch.arange(512, device="cuda", dtype=torch.float32)
    inv = 1 / (10000 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64))
    freq = t[:, None] * inv[None, :]
    cache = torch.cat((freq.cos(), freq.sin()), dim=-1).to(torch.bfloat16)
    layer = SimpleNamespace(dtype=torch.bfloat16, head_size=64, rotary_dim=64, cos_sin_cache=cache, is_neox_style=neox)
    with patch.object(R, "is_hcu", return_value=True), patch.object(R.envs.SGLANG_HCU_USE_LIGHTOP_ROPE, "get", return_value=True):
        R.after_rotary_init(None, layer)
    assert isinstance(layer._hcu_lightop_rope, R.HcuRotaryFallback)
    original = Mock(side_effect=AssertionError("LightOP opt-in must bypass FP32-only JIT"))
    pos = torch.tensor([1, 7, 65], dtype=torch.int64, device="cuda")
    q = torch.empty(3, 8, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(3, 1, 64, dtype=torch.bfloat16, device="cuda")

    def reference(x):
        cos, sin = cache[pos].float().chunk(2, dim=-1)
        cos, sin = cos[:, None, :], sin[:, None, :]
        xf = x.float()
        a, b = xf.chunk(2, dim=-1) if neox else (xf[..., 0::2], xf[..., 1::2])
        u, v = a * cos - b * sin, b * cos + a * sin
        return (torch.cat((u, v), dim=-1) if neox else torch.stack((u, v), dim=-1).flatten(-2)).to(x.dtype)

    def load():
        q0 = torch.randn_like(q);k0 = torch.randn_like(k)
        q.copy_(q0);k.copy_(k0)
        return reference(q0), reference(k0)

    def check(ref):
        torch.testing.assert_close(q, ref[0], rtol=0.02, atol=0.02)
        torch.testing.assert_close(k, ref[1], rtol=0.02, atol=0.02)

    for _ in range(3):
        ref = load()
        R.forward_cuda(original, layer, pos, q, k)
        check(ref)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        R.forward_cuda(original, layer, pos, q, k)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        R.forward_cuda(original, layer, pos, q, k)
    for positions in ([2, 19, 127], [31, 64, 511], [3, 9, 255]):
        pos.copy_(torch.tensor(positions, dtype=pos.dtype, device=pos.device))
        ref = load();graph.replay();torch.cuda.synchronize();check(ref)
    original.assert_not_called()
    assert layer.cos_sin_cache is cache
