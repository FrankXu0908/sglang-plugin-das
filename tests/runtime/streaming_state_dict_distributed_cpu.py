from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from hcu_sglang.multimodal_gen.loader.fsdp_load import (
    load_model_from_streaming_state_dict,
)


def load_rank(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        checkpoint = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        for offload in (False, True):
            model = nn.Module()
            model.weight = nn.Parameter(
                distribute_tensor(torch.zeros(4, 2), mesh, [Shard(0)]),
                requires_grad=False,
            )
            result = load_model_from_streaming_state_dict(
                model,
                iter((("weight", checkpoint),)),
                torch.device("cpu"),
                torch.float32,
                strict=True,
                cpu_offload=offload,
            )
            assert not result.missing_keys and not result.unexpected_keys
            assert isinstance(model.weight, DTensor)
            assert model.weight.placements == (Shard(0),)
            assert model.weight.to_local().shape == (2, 2)
            torch.testing.assert_close(model.weight.to_local(), checkpoint.chunk(2)[rank])

        model = nn.Module()
        model.weight = nn.Parameter(
            distribute_tensor(torch.zeros(4, 2), mesh, [Shard(0)]),
            requires_grad=False,
        )
        try:
            load_model_from_streaming_state_dict(
                model,
                iter((("unknown", torch.ones(1)),)),
                torch.device("cpu"),
                torch.float32,
                strict=False,
            )
        except ValueError as error:
            assert "not found in custom model state dict" in str(error)
        else:
            raise AssertionError("Distributed unknown weights must fail even in nonstrict mode")
        torch.testing.assert_close(model.weight.to_local(), torch.zeros(2, 2))
    finally:
        dist.destroy_process_group()


class DistributedStreamingStateDictTests(unittest.TestCase):
    def test_two_real_cpu_ranks_load_their_local_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            init_method = (Path(directory) / "gloo-store").as_uri()
            mp.start_processes(
                load_rank,
                args=(init_method,),
                nprocs=2,
                join=True,
                start_method="fork",
            )


if __name__ == "__main__":
    unittest.main()
