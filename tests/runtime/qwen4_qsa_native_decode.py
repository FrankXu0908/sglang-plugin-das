import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

GPU = os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1" and torch.cuda.is_available()


@unittest.skipUnless(GPU, "explicit HCU GPU lane")
class NativePagedDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang.environ import register_envs
        register_envs()

    def setUp(self):
        from sglang.srt.runtime_context import get_parallel
        self.enterContext(get_parallel().override(attn_cp_size=1))

    def make_backend(self, dtype, hcu_layout=True):
        from hcu_sglang.mem_cache.hcu_mha_pool import HCUMHATokenToKVPool
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
        from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        pool = QSATokenToKVPool(
            size=1024, page_size=64, dtype=dtype, head_num=2, head_dim=256,
            full_attention_layer_ids=[0], device="cuda", mamba_pool=None,
            qsa_index_kv_heads=2, qsa_index_head_dim=128, qsa_compress_ratio=4,
            qsa_token_topk=128, num_request_slots=4,
            full_kv_pool_class=HCUMHATokenToKVPool if hcu_layout else MHATokenToKVPool,
        )
        torch.manual_seed(32)
        for buffer in (pool.get_key_buffer(0), pool.get_value_buffer(0)):
            buffer.copy_((torch.randn(buffer.shape, device="cuda") * .15).to(dtype))
        pages = torch.tensor([[0, 0, 0, 0], [6, 2, 4, 8], [1, 7, 3, 5], [9, 10, 11, 12]], device="cuda")
        table = (pages[:, :, None] * 64 + torch.arange(64, device="cuda")).reshape(4, 256).to(torch.int32)
        runner = SimpleNamespace(token_to_kv_pool=pool, device="cuda", is_draft_worker=False,
                                 model_config=SimpleNamespace(context_len=256, hf_config=None),
                                 req_to_token_pool=SimpleNamespace(req_to_token=table))
        backend = QwenSparseAttnBackend(runner)
        layer = SimpleNamespace(layer_id=0, scaling=256**-.5, tp_q_head_num=24,
                                head_dim=256, k_scale_float=.5, v_scale_float=1.5)
        return backend, layer

    def batch(self, speculative=False):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        requests = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
        positions = torch.tensor([126, 130], device="cuda", dtype=torch.int64)
        if speculative:
            positions = (positions[:, None] + torch.arange(4, device="cuda")).flatten()
        return SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY if speculative else ForwardMode.DECODE,
            batch_size=2, input_ids=torch.zeros(positions.numel(), dtype=torch.int64, device="cuda"),
            req_pool_indices=requests, seq_lens=torch.tensor([127, 131], dtype=torch.int32, device="cuda"),
            seq_lens_cpu=torch.tensor([127, 131], dtype=torch.int32), positions=positions,
            out_cache_loc=torch.zeros_like(positions), mrope_positions=None,
            spec_info=SimpleNamespace(topk=1, draft_token_num=4) if speculative else None,
            num_padding=0,
        )

    def reference(self, backend, layer, q, indices):
        meta = backend.forward_metadata
        k, v = backend.token_to_kv_pool.get_key_buffer(0), backend.token_to_kv_pool.get_value_buffer(0)
        table = backend.req_to_token_pool.req_to_token
        outputs = []
        for row in range(q.shape[0]):
            valid = (indices[row] >= 0) & (indices[row] < meta.sequence_lengths[row])
            slots = table[meta.row_req_pool_indices[row].long(), indices[row, valid].long()].long()
            key = (k[slots] if k.ndim == 3 else k[slots // 64, :, slots % 64, :]).float()
            value = (v[slots] if v.ndim == 3 else v[slots // 64, :, slots % 64, :]).float()
            if k.dtype != torch.bfloat16:
                key *= layer.k_scale_float
                value *= layer.v_scale_float
            scores = torch.einsum("hgd,nhd->hgn", q[row].float().reshape(2, 12, 256), key) * layer.scaling
            outputs.append(torch.einsum("hgn,nhd->hgd", scores.softmax(-1), value).reshape(24, 256))
        return torch.stack(outputs).flatten(1)

    def check_case(self, dtype, speculative, graph, hcu_layout=True):
        from hcu_sglang.models.qwen import qsa_kv as K
        from sglang.srt.layers.attention import qwen_sparse_attn_backend as N

        backend, layer = self.make_backend(dtype, hcu_layout)
        batch = self.batch(speculative)
        backend.init_forward_metadata(batch)
        rows = batch.positions.numel()
        indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(rows, 1)
        indices[:, 100:] = -1
        indices[:, 95:100] = 200
        q = torch.randn(rows, 24, 256, device="cuda", dtype=torch.bfloat16) * .15
        original = N.QwenSparseAttnBackend._forward_paged_attention
        def run():
            return K.around_paged_attention(original, backend, q, layer, batch, indices)
        with patch.dict(os.environ, {"SGLANG_HCU_QSA_DECODE_IMPL": "native-paged"}):
            for _ in range(3):
                output = run()
            torch.testing.assert_close(output.float(), self.reference(backend, layer, q, indices), atol=2e-3, rtol=.015)
            self.assertTrue(backend._hcu_native_decode_logged)
            if not graph:
                indices.fill_(-1)
                self.assertTrue(torch.isnan(run()).all())
                return
            with patch.object(N, "get_spec", return_value=SimpleNamespace(speculative_num_draft_tokens=4)):
                backend.init_cuda_graph_state(2, rows)
            backend.init_forward_metadata_out_graph(batch, in_capture=True)
            self.assertEqual(backend.forward_metadata.token_slot_table.shape[-1], 1)
            for _ in range(3):
                run()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                output = run()
            old = output.clone()
            q.copy_(torch.randn_like(q) * .2)
            indices[:, :95] = torch.arange(95, device="cuda", dtype=torch.int32).flip(0) + 4
            batch.req_pool_indices.copy_(torch.tensor([2, 3], device="cuda", dtype=torch.int32))
            batch.seq_lens.add_(8)
            batch.seq_lens_cpu.add_(8)
            batch.positions.add_(8)
            backend.init_forward_metadata_out_graph(batch, in_capture=False)
            backend.req_to_token_pool.req_to_token[3, :64] = torch.arange(64, device="cuda", dtype=torch.int32) + 13 * 64
            captured.replay()
            torch.cuda.synchronize()
            self.assertFalse(torch.equal(output, old))
            torch.testing.assert_close(output.float(), self.reference(backend, layer, q, indices), atol=2e-3, rtol=.015)

    def test_eager_dtypes_scales_and_empty_rows(self):
        for dtype in (torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn):
            for speculative in (False, True):
                for hcu_layout in (False, True):
                    with self.subTest(dtype=dtype, speculative=speculative, hcu_layout=hcu_layout):
                        self.check_case(dtype, speculative, False, hcu_layout)

    def test_graph_request_page_query_and_index_changes(self):
        for speculative in (False, True):
            for hcu_layout in (False, True):
                with self.subTest(speculative=speculative, hcu_layout=hcu_layout):
                    self.check_case(torch.float8_e5m2, speculative, True, hcu_layout)

    def test_cpu_uses_current_original_without_loading_fa(self):
        from hcu_sglang.models.qwen import qsa_kv as K
        sentinel = object()
        with patch.dict(os.environ, {"SGLANG_HCU_QSA_DECODE_IMPL": "native-paged"}):
            output = K.around_paged_attention(lambda *args: sentinel, None, torch.zeros(1, 24, 256), None, None, None)
        self.assertIs(output, sentinel)

    def test_fallback_override_and_provider_failure(self):
        from hcu_sglang.models.qwen import qsa_kv as K
        from sglang.srt.layers.attention import qwen_sparse_attn_backend as N

        backend, layer = self.make_backend(torch.float8_e5m2)
        batch = self.batch()
        backend.init_forward_metadata(batch)
        indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(2, 1)
        q = torch.randn(2, 24, 256, device="cuda", dtype=torch.bfloat16)
        original = N.QwenSparseAttnBackend._forward_paged_attention
        for mode, query in (("triton", q), ("native-paged", q[:, :12])):
            with patch.dict(os.environ, {"SGLANG_HCU_QSA_DECODE_IMPL": mode}), patch.object(
                N, "sparse_gqa_packed_decode_triton", return_value=torch.zeros_like(query)
            ) as packed:
                actual = K.around_paged_attention(original, backend, query, layer, batch, indices)
                self.assertEqual(actual.shape, (2, query.shape[1] * 256))
                packed.assert_called_once()
        from sglang.srt.runtime_context import get_parallel
        with get_parallel().override(attn_cp_size=2), patch.dict(
            os.environ, {"SGLANG_HCU_QSA_DECODE_IMPL": "native-paged"}
        ), patch.object(N, "sparse_gqa_packed_decode_triton", return_value=torch.zeros_like(q)) as packed:
            K.around_paged_attention(original, backend, q, layer, batch, indices)
            packed.assert_called_once()
        with patch("hcu_sglang.utils.detection.get_gcn_arch", return_value="gfx938"), patch.dict(
            os.environ, {"SGLANG_HCU_QSA_DECODE_IMPL": "native-paged"}
        ), patch.object(N, "sparse_gqa_packed_decode_triton", return_value=torch.zeros_like(q)) as packed:
            K.around_paged_attention(original, backend, q, layer, batch, indices)
            packed.assert_called_once()
        with patch.dict(os.environ, {"SGLANG_HCU_QSA_DECODE_IMPL": "native-paged"}), patch(
            "flash_attn.flash_attn_interface.sparse_gqa_paged_attn_fp8_func",
            side_effect=RuntimeError("provider sentinel"),
        ):
            with self.assertRaisesRegex(RuntimeError, "provider sentinel"):
                K.around_paged_attention(original, backend, q, layer, batch, indices)

    def test_empty_decode_batch(self):
        from hcu_sglang.models.qwen import qsa_kv as K

        backend, layer = self.make_backend(torch.bfloat16)
        batch = self.batch()
        backend.init_forward_metadata(batch)
        q = torch.empty(0, 24, 256, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"SGLANG_HCU_QSA_DECODE_IMPL": "native-paged"}):
            output = K.around_paged_attention(None, backend, q, layer, batch,
                                             torch.empty(0, 128, device="cuda", dtype=torch.int32))
        self.assertEqual(output.shape, (0, 6144))


if __name__ == "__main__":
    unittest.main()
