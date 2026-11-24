# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.environ import register_envs

register_envs()  # topk_source reads envs.SGLANG_SIMULATED_EXPERT_BALANCE at import

from hcu_sglang.layers.moe import mori_das, topk_source
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput
from sglang.srt.layers.moe.utils import MoeA2ABackend


class FakeGroup:
    world_size = 16
    rank_in_group = 0


class FakeOp:
    def __init__(self):
        kernel = SimpleNamespace(name="InterNodeV1LLChunkedSync")
        self.config = SimpleNamespace(kernel_type=kernel)
        self.combine_buffer = torch.empty(2, 3, 4, dtype=torch.bfloat16)
        self.dispatch_calls = 0
        self.combine_calls = 0

    def dispatch_standard_moe(self, *, input, scales, weights, indices):
        self.dispatch_calls += 1
        packed = torch.ones(2, 3, 4, dtype=torch.int8)
        scale = torch.ones(2, 3, 1, dtype=torch.float32)
        counts = torch.tensor([2, 1], dtype=torch.int32)
        return (packed, scale), counts, None, None

    def combine_standard_moe(self, *, input, indices, weights):
        self.combine_calls += 1
        return torch.ones(indices.shape[0], 4, dtype=torch.bfloat16)

    def get_standard_moe_combine_input_buffer(self, *, dtype, hidden_dim):
        self.combine_buffer = self.combine_buffer.to(dtype)
        return self.combine_buffer


class MoriDasTest(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get(mori_das._BACKEND_ENV)

    def tearDown(self):
        if self.previous is None:
            os.environ.pop(mori_das._BACKEND_ENV, None)
        else:
            os.environ[mori_das._BACKEND_ENV] = self.previous

    def test_masked_gemm_declares_output_buffer_write(self):
        schema = torch.ops.sglang.mori_grouped_i8_gemm_nt_masked.default._schema
        written_args = [
            arg.name
            for arg in schema.arguments
            if arg.alias_info is not None and arg.alias_info.is_write
        ]
        self.assertEqual(written_args, ["output"])
        self.assertEqual(len(schema.returns), 0)

    def test_moe_core_routes_by_dispatcher_with_both_backends_present(self):
        from hcu_sglang.layers.moe import ct_w4a8, deepep_fp8

        mori_das._enable()
        dispatcher = mori_das.MoriDasDispatcher(
            group=FakeGroup(),
            router_topk=2,
            num_experts=32,
            num_local_experts=2,
            hidden_size=4,
            mori_op=FakeOp(),
        )
        quant_method = SimpleNamespace(hcu_w4a8_hipc=True)
        mori_layer = SimpleNamespace(dispatcher=dispatcher, quant_method=quant_method)
        deepep_layer = SimpleNamespace(dispatcher=object(), quant_method=quant_method)
        output = object()
        original = Mock()
        with (
            patch.object(deepep_fp8, "_is_hcu", True),
            patch.object(mori_das, "run_moe_core") as mori_run,
            patch.object(ct_w4a8, "run_moe_core") as deepep_run,
        ):
            self.assertIs(
                deepep_fp8.around_run_moe_core(original, mori_layer, output),
                mori_run.return_value,
            )
            self.assertIs(
                deepep_fp8.around_run_moe_core(original, deepep_layer, output),
                deepep_run.return_value,
            )
        mori_run.assert_called_once_with(mori_layer, output)
        deepep_run.assert_called_once_with(deepep_layer, output)
        original.assert_not_called()

    def test_dispatch_uses_native_deepep_ll_protocol(self):
        dispatcher = mori_das.MoriDasDispatcher(
            group=FakeGroup(),
            router_topk=2,
            num_experts=32,
            num_local_experts=2,
            hidden_size=4,
            mori_op=FakeOp(),
        )
        topk = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1], [2, -1]], dtype=torch.int64),
            topk_weights=torch.ones(2, 2),
        )
        output = dispatcher.dispatch(torch.ones(2, 4, dtype=torch.bfloat16), topk)
        self.assertEqual(output.topk_ids.dtype, torch.int32)
        self.assertEqual(output.hidden_states.dtype, torch.int8)
        self.assertEqual(output.expected_m, 3)
        combined = dispatcher.combine(
            DeepEPLLCombineInput(
                hidden_states=torch.ones(2, 3, 4, dtype=torch.bfloat16),
                topk_ids=output.topk_ids,
                topk_weights=output.topk_weights,
            )
        )
        self.assertEqual(tuple(combined.shape), (2, 4))

    def test_split_phase_keeps_mori_ops_in_b_phases(self):
        op = FakeOp()
        dispatcher = mori_das.MoriDasDispatcher(
            group=FakeGroup(),
            router_topk=2,
            num_experts=32,
            num_local_experts=2,
            hidden_size=4,
            mori_op=op,
        )
        topk = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1], [2, -1]], dtype=torch.int64),
            topk_weights=torch.ones(2, 2),
        )

        dispatcher.dispatch_a(torch.ones(2, 4, dtype=torch.bfloat16), topk)
        self.assertEqual(dispatcher._stage, mori_das._Stage.AFTER_DISPATCH_A)
        self.assertEqual(op.dispatch_calls, 0)

        output = dispatcher.dispatch_b()
        self.assertEqual(dispatcher._stage, mori_das._Stage.AFTER_DISPATCH_B)
        self.assertEqual(op.dispatch_calls, 1)

        dispatcher.combine_a(
            DeepEPLLCombineInput(
                hidden_states=torch.ones(2, 3, 4, dtype=torch.bfloat16),
                topk_ids=output.topk_ids,
                topk_weights=output.topk_weights,
            )
        )
        self.assertEqual(dispatcher._stage, mori_das._Stage.AFTER_COMBINE_A)
        self.assertEqual(op.combine_calls, 0)

        combined = dispatcher.combine_b()
        self.assertEqual(dispatcher._stage, mori_das._Stage.INITIAL)
        self.assertEqual(op.combine_calls, 1)
        self.assertEqual(tuple(combined.shape), (2, 4))

    def test_torch_compile_keeps_communication_eager(self):
        class CompileProbeOp(FakeOp):
            def dispatch_standard_moe(self, **kwargs):
                self.dispatch_compiling = torch.compiler.is_compiling()
                return super().dispatch_standard_moe(**kwargs)

            def combine_standard_moe(self, *, input, indices, weights):
                self.combine_compiling = torch.compiler.is_compiling()
                self.combine_input = input
                return super().combine_standard_moe(
                    input=input, indices=indices, weights=weights
                )

        op = CompileProbeOp()
        dispatcher = mori_das.MoriDasDispatcher(
            group=FakeGroup(),
            router_topk=2,
            num_experts=32,
            num_local_experts=2,
            hidden_size=4,
            mori_op=op,
        )
        topk = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1], [2, -1]], dtype=torch.int64),
            topk_weights=torch.ones(2, 2),
        )

        @torch.compile(backend="eager")
        def forward(hidden_states):
            dispatched = dispatcher.dispatch(hidden_states, topk)
            op.combine_buffer.copy_(dispatched.hidden_states)
            return dispatcher.combine(
                DeepEPLLCombineInput(
                    hidden_states=op.combine_buffer,
                    topk_ids=dispatched.topk_ids,
                    topk_weights=dispatched.topk_weights,
                )
            )

        combined = forward(torch.ones(2, 4, dtype=torch.bfloat16))
        self.assertFalse(op.dispatch_compiling)
        self.assertFalse(op.combine_compiling)
        self.assertIs(op.combine_input, op.combine_buffer)
        torch.testing.assert_close(op.combine_buffer, torch.ones_like(op.combine_buffer))
        torch.testing.assert_close(combined, torch.ones_like(combined))
        self.assertEqual(dispatcher._stage, mori_das._Stage.INITIAL)

    def test_split_phase_rejects_invalid_order(self):
        dispatcher = mori_das.MoriDasDispatcher(
            group=FakeGroup(),
            router_topk=2,
            num_experts=32,
            num_local_experts=2,
            hidden_size=4,
            mori_op=FakeOp(),
        )
        with self.assertRaisesRegex(RuntimeError, "expected after_dispatch_a"):
            dispatcher.dispatch_b()

    def test_empty_input_uses_one_negative_sentinel(self):
        dispatcher = mori_das.MoriDasDispatcher(
            group=FakeGroup(),
            router_topk=2,
            num_experts=32,
            num_local_experts=2,
            hidden_size=4,
            mori_op=FakeOp(),
        )
        topk = SimpleNamespace(
            topk_ids=torch.empty(0, 2, dtype=torch.int64),
            topk_weights=torch.empty(0, 2),
        )
        output = dispatcher.dispatch(torch.empty(0, 4, dtype=torch.bfloat16), topk)
        self.assertEqual(output.topk_ids.tolist(), [[-1, -1]])
        combined = dispatcher.combine(
            DeepEPLLCombineInput(
                hidden_states=torch.ones(2, 3, 4, dtype=torch.bfloat16),
                topk_ids=output.topk_ids,
                topk_weights=output.topk_weights,
            )
        )
        self.assertEqual(tuple(combined.shape), (0, 4))

    def test_registered_combine_buffer_is_exposed(self):
        dispatcher = mori_das.MoriDasDispatcher(
            group=FakeGroup(),
            router_topk=2,
            num_experts=32,
            num_local_experts=2,
            hidden_size=4,
            mori_op=FakeOp(),
        )
        self.assertTrue(dispatcher.requires_registered_combine_input_buffer())
        self.assertEqual(
            tuple(
                dispatcher.get_standard_moe_combine_input_buffer(
                    torch.bfloat16, 4
                ).shape
            ),
            (2, 3, 4),
        )

    def test_default_chunked_sync_matches_explicit_selection(self):
        kernels = SimpleNamespace(
            InterNodeV1LLChunkedSync=object(),
            InterNodeV1LLChunked=object(),
            InterNodeV1LLNative=object(),
        )
        mori = SimpleNamespace(
            ops=SimpleNamespace(
                EpDispatchCombineKernelType=kernels,
                EpDispatchCombineConfig=SimpleNamespace,
                EpDispatchCombineOp=lambda config: SimpleNamespace(config=config),
            )
        )
        env = {"SGLANG_MORI_DAS_CHUNKED_SYNC_IMPL": "0"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"mori": mori}),
            patch.dict(mori_das._OPS, clear=True),
            patch.object(mori_das, "_initialize_shmem", return_value="test"),
        ):
            default_op = mori_das._build_op(FakeGroup(), 2, 2, 4, torch.bfloat16)
            self.assertIs(default_op.config.kernel_type, kernels.InterNodeV1LLChunkedSync)
            self.assertEqual(default_op.config.chunk_size, 32)
            self.assertEqual(default_op.config.warp_num_per_block, 8)
            self.assertFalse(default_op.config.combine_zero_copy)
            self.assertEqual(os.environ["MORI_CHUNKED_SYNC_IMPL"], "0")

            os.environ["SGLANG_MORI_DAS_KERNEL_TYPE"] = "InterNodeV1LLChunkedSync"
            explicit_op = mori_das._build_op(FakeGroup(), 2, 2, 4, torch.bfloat16)
            self.assertIs(default_op, explicit_op)

            for name in ("InterNodeV1LLChunked", "InterNodeV1LLNative"):
                with self.subTest(kernel=name):
                    os.environ["SGLANG_MORI_DAS_KERNEL_TYPE"] = name
                    other_op = mori_das._build_op(FakeGroup(), 2, 2, 4, torch.bfloat16)
                    self.assertIs(other_op.config.kernel_type, getattr(kernels, name))

    def test_gemm2_output_is_passed_directly_to_combine(self):
        layer = SimpleNamespace(
            w13_weight=torch.empty(0),
            w13_weight_scale=torch.ones(2, 8),
            w2_weight=torch.empty(0),
            w2_weight_scale=torch.ones(2, 4),
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
        )
        topk = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1], [2, -1]], dtype=torch.int64),
            topk_weights=torch.ones(2, 2),
        )
        for name, registered in (
            ("InterNodeV1LLChunkedSync", True),
            ("InterNodeV1LLChunked", True),
            ("InterNodeV1LLNative", False),
        ):
            with self.subTest(kernel=name):
                op = FakeOp()
                op.config.kernel_type.name = name
                layer.dispatcher = mori_das.MoriDasDispatcher(
                    group=FakeGroup(),
                    router_topk=2,
                    num_experts=32,
                    num_local_experts=2,
                    hidden_size=4,
                    mori_op=op,
                )
                dispatch = layer.dispatcher.dispatch(
                    torch.ones(2, 4, dtype=torch.bfloat16), topk
                )
                with (
                    patch.object(
                        op,
                        "get_standard_moe_combine_input_buffer",
                        wraps=op.get_standard_moe_combine_input_buffer,
                    ) as get_buffer,
                    patch.object(
                        op, "combine_standard_moe", wraps=op.combine_standard_moe
                    ) as combine,
                    patch.object(
                        torch.ops.sglang,
                        "mori_grouped_i8_gemm_nt_masked",
                        side_effect=lambda *args: args[4].fill_(1),
                    ) as gemm,
                    patch.object(
                        torch.ops.sglang,
                        "fuse_silu_mul_quant_ep",
                        return_value=(dispatch.hidden_states, dispatch.hidden_states_scale),
                        create=True,
                    ),
                ):
                    result = mori_das.run_moe_core(layer, dispatch)
                    layer.dispatcher.combine(result)
                gemm2_output = gemm.call_args.args[4]
                self.assertEqual(gemm.call_count, 2)
                self.assertIs(result.hidden_states, gemm2_output)
                self.assertIs(combine.call_args.kwargs["input"], gemm2_output)
                self.assertEqual(gemm2_output.dtype, torch.bfloat16)
                if registered:
                    get_buffer.assert_called_once_with(dtype=torch.bfloat16, hidden_dim=4)
                    self.assertIs(gemm2_output, op.combine_buffer)
                else:
                    get_buffer.assert_not_called()
                    self.assertNotEqual(gemm2_output.data_ptr(), op.combine_buffer.data_ptr())

    def test_moe_core_selects_swiglu_quantization(self):
        unclamped = torch.tensor([[[-16.0, 18.0, -14.0, 16.0]]], dtype=torch.bfloat16)
        clamped = torch.tensor([[[-16.0, 10.0, -10.0, 10.0]]], dtype=torch.bfloat16)
        masked_m = torch.tensor([1], dtype=torch.int32)
        dispatch_output = mori_das.DeepEPLLDispatchOutput(
            hidden_states=torch.zeros(1, 1, 2, dtype=torch.int8),
            hidden_states_scale=torch.ones(1, 1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
            topk_weights=torch.ones(1, 1),
            masked_m=masked_m,
            expected_m=1,
        )
        layer = SimpleNamespace(
            w13_weight=torch.empty(0),
            w13_weight_scale=torch.ones(1, 4),
            w2_weight=torch.empty(0),
            w2_weight_scale=torch.ones(1, 2),
            dispatcher=SimpleNamespace(
                requires_registered_combine_input_buffer=lambda: False
            ),
        )

        def gemm(hidden, _scale, _weight, _weight_scale, output, _mask, _expected):
            if output.size(-1) == 4:
                output.copy_(unclamped)
            else:
                output.copy_(hidden)

        register_envs()
        for limit, flag in ((10.0, "0"), (10.0, "1"), (None, "0"), (None, "1")):
            with self.subTest(limit=limit, fused=flag):
                layer.moe_runner_config = SimpleNamespace(swiglu_limit=limit)
                expected = clamped if limit is not None else unclamped

                def quant(gateup, mask):
                    torch.testing.assert_close(gateup, expected)
                    self.assertIs(mask, masked_m)
                    return gateup[..., :2].to(torch.int8), torch.ones(1, 1, 1)

                def fused_quant(*, input, limit, mask_m, expect_m):
                    torch.testing.assert_close(input, unclamped)
                    self.assertEqual(limit, 10.0)
                    self.assertIs(mask_m, masked_m)
                    self.assertEqual(expect_m, 1)
                    return clamped[..., :2].to(torch.int8), torch.ones(1, 1, 1)

                quant_op = Mock(side_effect=quant)
                fused_op = Mock(side_effect=fused_quant)
                with (
                    patch.dict(
                        os.environ, {"SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT": flag}
                    ),
                    patch.dict(
                        sys.modules,
                        {
                            "lightop.activation": SimpleNamespace(
                                fuse_silu_mul_clamp_quant=fused_op
                            )
                        },
                    ),
                    patch.object(
                        torch.ops.sglang, "mori_grouped_i8_gemm_nt_masked", gemm
                    ),
                    patch.object(
                        torch.ops.sglang, "fuse_silu_mul_quant_ep", quant_op, create=True
                    ),
                ):
                    result = mori_das.run_moe_core(layer, dispatch_output)
                if limit is not None and flag == "1":
                    fused_op.assert_called_once()
                    quant_op.assert_not_called()
                else:
                    fused_op.assert_not_called()
                    quant_op.assert_called_once()
                torch.testing.assert_close(result.hidden_states, expected[..., :2])

    def test_backend_marker_is_isolated_to_mori_das(self):
        os.environ.pop(mori_das._BACKEND_ENV, None)
        self.assertFalse(
            mori_das.around_is_deepep(lambda _: False, MoeA2ABackend.CUSTOMIZED)
        )
        backend = mori_das.around_missing_backend(
            lambda cls, value: cls.NONE, MoeA2ABackend, "mori-das"
        )
        self.assertEqual(backend, MoeA2ABackend.CUSTOMIZED)
        self.assertTrue(mori_das.around_is_deepep(lambda _: False, backend))

    def test_topk_padding_uses_int32_negative_ids(self):
        class DeepGemmRunner:
            @staticmethod
            def is_deep_gemm():
                return True

        mori_das._enable()
        ids = torch.tensor([[1, 2], [0, 0]], dtype=torch.int32)
        weights = torch.ones(2, 2)
        config = SimpleNamespace(
            num_fused_shared_experts=0,
            fused_shared_experts_scaling_factor=None,
            top_k=2,
            allow_routed_experts_capture=False,
        )

        def postprocess(tensor, remap_info, count, output_int64=False):
            self.assertIsNone(remap_info)
            self.assertFalse(output_int64)
            result = tensor.clone()
            result[int(count.item()) :] = -1
            return result

        with (
            patch.object(topk_source, "_is_hcu", True),
            patch.object(topk_source, "_is_hip", True),
            patch.object(topk_source, "_is_cuda", False),
            patch.object(topk_source, "_use_aiter", False),
            patch.object(topk_source, "_use_deepgemm_moe", False),
            patch.object(
                topk_source,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.CUSTOMIZED,
            ),
            patch.object(
                topk_source,
                "get_moe_runner_backend",
                return_value=DeepGemmRunner(),
            ),
            patch.object(
                topk_source, "has_per_rank_fused_shared_slots", return_value=False
            ),
            patch.object(
                topk_source, "is_expert_remapping_enabled", return_value=False
            ),
            patch.object(
                topk_source,
                "_biased_grouped_topk_postprocess",
                side_effect=postprocess,
            ),
            patch.object(topk_source, "_mask_topk_ids_padded_region") as mask_ids,
            patch.object(
                topk_source, "_zero_topk_weights_padded_region"
            ) as zero_weights,
        ):
            result = topk_source._post_process_topk_ids(
                topk_ids=ids,
                topk_weights=weights,
                topk_config=config,
                router_logits=torch.empty(2, 8),
                layer_id=0,
                num_token_non_padded=torch.tensor(1),
            )
        self.assertEqual(result[0].dtype, torch.int32)
        self.assertEqual(result[0].tolist(), [[1, 2], [-1, -1]])
        self.assertTrue(torch.equal(result[1], weights))
        mask_ids.assert_not_called()
        zero_weights.assert_not_called()

    def test_deepep_padding_keeps_int64_and_skips_weight_zero(self):
        class DeepGemmRunner:
            @staticmethod
            def is_deep_gemm():
                return True

        os.environ.pop(mori_das._BACKEND_ENV, None)
        ids = torch.tensor([[1, 2], [0, 0]], dtype=torch.int32)
        weights = torch.ones(2, 2)
        config = SimpleNamespace(
            num_fused_shared_experts=0,
            fused_shared_experts_scaling_factor=None,
            top_k=2,
            allow_routed_experts_capture=False,
        )

        def postprocess(tensor, remap_info, count, output_int64=False):
            self.assertTrue(output_int64)
            result = tensor.to(torch.int64)
            result[int(count.item()) :] = -1
            return result

        with (
            patch.object(topk_source, "_is_hcu", True),
            patch.object(topk_source, "_is_hip", True),
            patch.object(topk_source, "_is_cuda", False),
            patch.object(topk_source, "_use_aiter", False),
            patch.object(topk_source, "_use_deepgemm_moe", False),
            patch.object(
                topk_source,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.DEEPEP,
            ),
            patch.object(
                topk_source,
                "get_moe_runner_backend",
                return_value=DeepGemmRunner(),
            ),
            patch.object(
                topk_source, "has_per_rank_fused_shared_slots", return_value=False
            ),
            patch.object(
                topk_source, "is_expert_remapping_enabled", return_value=False
            ),
            patch.object(
                topk_source,
                "_biased_grouped_topk_postprocess",
                side_effect=postprocess,
            ),
            patch.object(topk_source, "_mask_topk_ids_padded_region") as mask_ids,
            patch.object(
                topk_source, "_zero_topk_weights_padded_region"
            ) as zero_weights,
        ):
            result = topk_source._post_process_topk_ids(
                topk_ids=ids,
                topk_weights=weights,
                topk_config=config,
                router_logits=torch.empty(2, 8),
                layer_id=0,
                num_token_non_padded=torch.tensor(1),
            )
        self.assertEqual(result[0].dtype, torch.int64)
        self.assertEqual(result[0].tolist(), [[1, 2], [-1, -1]])
        mask_ids.assert_not_called()
        zero_weights.assert_not_called()

    def test_generic_hip_padding_still_zeros_weights(self):
        class DeepGemmRunner:
            @staticmethod
            def is_deep_gemm():
                return True

        os.environ.pop(mori_das._BACKEND_ENV, None)
        ids = torch.tensor([[1, 2], [7, 6]], dtype=torch.int32)
        weights = torch.ones(2, 2)
        config = SimpleNamespace(
            num_fused_shared_experts=0,
            fused_shared_experts_scaling_factor=None,
            top_k=2,
            allow_routed_experts_capture=False,
        )

        def mask_ids(tensor, count, fill_value=-1):
            tensor[int(count.item()) :] = fill_value

        def zero_weights(tensor, count):
            tensor[int(count.item()) :] = 0

        with (
            patch.object(topk_source, "_is_hcu", True),
            patch.object(topk_source, "_is_hip", True),
            patch.object(topk_source, "_is_cuda", False),
            patch.object(topk_source, "_use_aiter", False),
            patch.object(topk_source, "_use_deepgemm_moe", False),
            patch.object(
                topk_source,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            patch.object(
                topk_source,
                "get_moe_runner_backend",
                return_value=DeepGemmRunner(),
            ),
            patch.object(
                topk_source, "has_per_rank_fused_shared_slots", return_value=False
            ),
            patch.object(
                topk_source, "is_expert_remapping_enabled", return_value=False
            ),
            patch.object(
                topk_source,
                "_mask_topk_ids_padded_region",
                side_effect=mask_ids,
            ) as mask_mock,
            patch.object(
                topk_source,
                "_zero_topk_weights_padded_region",
                side_effect=zero_weights,
            ) as zero_mock,
        ):
            result = topk_source._post_process_topk_ids(
                topk_ids=ids,
                topk_weights=weights,
                topk_config=config,
                router_logits=torch.empty(2, 8),
                layer_id=0,
                num_token_non_padded=torch.tensor(1),
            )
        self.assertEqual(result[0].tolist(), [[1, 2], [0, 0]])
        self.assertEqual(result[1].tolist(), [[1.0, 1.0], [0.0, 0.0]])
        mask_mock.assert_called_once()
        zero_mock.assert_called_once()

if __name__ == "__main__":
    unittest.main()
