"""Ulysses replicated-KV-prefix attention without the post-all-to-all K/V cat.

Original (USPAttention._forward_with_replicated_kv_prefix_split, B=1, Ring=1):
    send_x = rearrange(x, "b l (w h) d -> w b l h d").contiguous()   # x in q, k, v
    recv_x = empty_like(send_x); all_to_all_single(recv_x, send_x)   # -> [1, S, h, D]
    k_rep = k_rep[..., my heads, :].contiguous()                     # prefix slice copy
    k = cat([k_rep, recv_k], 1); v = cat([v_rep, recv_v], 1)         # full K/V copy
    out = attn_impl(q, k, v); return output_all_to_all(out)

Here K/V are received straight into ``final[:, P:]`` of a preallocated
``[1, P+S, h, D]`` buffer and the prefix head slice is copied once into
``final[:, :P]``, so attention reads the same bytes as ``cat`` produced.
Steady-state collective count, order and sizes are unchanged.
"""

import torch
from einops import rearrange

from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.usp import _usp_output_all_to_all
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)

logger = init_logger(__name__)
_DIRECT_RECV_GATE = BitExactFusionGate(
    "Ulysses replicated-KV-prefix direct receive into final K/V buffer",
    per_signature=True,
)


def _supported(attn, q, k_rep, v_rep, k_shard, v_shard):
    if (
        torch.version.hip is None
        or not q.is_cuda
        or q.ndim != 4
        or attn.backend.name != "TORCH_SDPA"
        or torch.compiler.is_compiling()
        or torch.cuda.is_current_stream_capturing()
        or is_in_breakable_cuda_graph()
        or get_ring_parallel_world_size() != 1
    ):
        return False
    world = get_ulysses_parallel_world_size()
    if world <= 1:
        return False
    b, _, heads, dim = k_shard.shape
    return (
        b == 1
        and q.shape == k_shard.shape == v_shard.shape
        and k_rep.shape == v_rep.shape
        and k_rep.ndim == 4
        and k_rep.shape[0] == 1
        and k_rep.shape[2:] == (heads, dim)
        and heads % world == 0
        and len({t.dtype for t in (q, k_rep, v_rep, k_shard, v_shard)}) == 1
        and len({t.device for t in (q, k_rep, v_rep, k_shard, v_shard)}) == 1
    )


def _receive(attn, q, k_rep, v_rep, k_shard, v_shard):
    """Same issue order as async_a2a_communicate: pack i+1 overlaps a2a i."""
    world = get_ulysses_parallel_world_size()
    rank = get_ulysses_parallel_rank()
    group = get_sp_group().ulysses_group
    stream = attn._get_usp_a2a_stream()
    _, seq, heads, dim = k_shard.shape
    local_heads = heads // world
    prefix = k_rep.shape[1]
    total = prefix + world * seq

    k_final = k_shard.new_empty(1, total, local_heads, dim)
    v_final = v_shard.new_empty(1, total, local_heads, dim)
    q_recv = q.new_empty(world, 1, seq, local_heads, dim)
    outputs = (
        q_recv,
        k_final[:, prefix:].view(world, 1, seq, local_heads, dim),
        v_final[:, prefix:].view(world, 1, seq, local_heads, dim),
    )
    inputs = [q, k_shard, v_shard]
    requests = [None] * 3
    inputs[0] = rearrange(inputs[0], "b l (w h) d -> w b l h d", w=world).contiguous()
    for i in range(3):
        requests[i] = torch.distributed.all_to_all_single(
            outputs[i], inputs[i], group=group, async_op=True
        )
        if i + 1 < 3:
            inputs[i + 1] = rearrange(
                inputs[i + 1], "b l (w h) d -> w b l h d", w=world
            ).contiguous()
    heads_slice = slice(rank * local_heads, (rank + 1) * local_heads)
    # Disjoint from the a2a destinations ([:, P:]), so it overlaps the transfer.
    k_final[:, :prefix].copy_(k_rep[:, :, heads_slice])
    v_final[:, :prefix].copy_(v_rep[:, :, heads_slice])
    with torch.get_device_module().stream(stream):
        for request in requests:
            request.wait()
    torch.get_device_module().current_stream().wait_stream(stream)
    return q_recv.view(1, world * seq, local_heads, dim), k_final, v_final


def try_direct_recv(original_fn, attn, q, k_rep, v_rep, k_shard, v_shard, ctx_attn_metadata):
    if _DIRECT_RECV_GATE.disabled or not _supported(
        attn, q, k_rep, v_rep, k_shard, v_shard
    ):
        return None
    signature = (
        q.shape,
        k_rep.shape,
        q.dtype,
        get_ulysses_parallel_world_size(),
    )
    verified = _DIRECT_RECV_GATE.is_verified(signature)
    q_full, k_final, v_final = _receive(attn, q, k_rep, v_rep, k_shard, v_shard)
    out = attn.attn_impl.forward(q_full, k_final, v_final, ctx_attn_metadata)
    out = _usp_output_all_to_all(out, head_dim=2)
    if verified:
        return out
    # First sight of this signature: every rank sees the same signature at
    # the same call, so all ranks run the reference collectives together.
    reference = original_fn(attn, q, k_rep, v_rep, k_shard, v_shard, ctx_attn_metadata)
    # Agree on acceptance across the Ulysses group before any rank can fall back.
    agreement = torch.tensor(int(torch.equal(out, reference)), device=q.device)
    torch.distributed.all_reduce(
        agreement, op=torch.distributed.ReduceOp.MIN, group=get_sp_group().ulysses_group
    )
    if agreement.item():
        _DIRECT_RECV_GATE.mark_verified(signature)
        if get_ulysses_parallel_rank() == 0:
            logger.info("SP direct receive exact verification passed on all ranks: %s", signature)
        return out
    _DIRECT_RECV_GATE.disable()
    if get_ulysses_parallel_rank() == 0:
        logger.warning("SP direct receive mismatch on a rank; all ranks use original route")
    return reference
