"""QSA prefill HIP TileLang guard: branch coverage without touching devices.

上游 0.5.22 的 `qsa_mqa_prefill` 在 HIP 上直接进 TileLang 分支，而该 kernel
的 shared memory 需求（80 KiB）超过设备上限（64 KiB），
`hipModuleLaunchKernel` 返回 invalid argument，prefill 首次请求即崩。
0.5.18 曾有 `not is_hip()` 守卫，0.5.22 丢失。

`hcu_sglang.models.qwen.qsa_tuning.around_mqa_prefill` 现在在 HIP 上主动改走
`sglang.srt.layers.attention.qsa.mqa.torch_qsa_mqa_prefill`。本文件用 fake 对象
钉住该 hook 的分支选择，不需要真的 GPU：

  1. key_span 形状完全匹配 -> 走 qsa_mqa_split_key.prefill（最高优先级）
  2. 形状不匹配 + HIP     -> 走 torch 参考实现
  3. 形状不匹配 + 非 HIP  -> 交回 original
  4. torch 参考实现不可用 -> 交回 original（不掩盖上游错误）
  5. key_span 非法值      -> 抛 ValueError
"""

import sys
import types
import unittest
from unittest import mock

FAKE_BF16 = "bf16"
FAKE_F32 = "f32"


class _FakeTensor:
    """足够让形状/设备/类型判断走通的最小张量替身。"""

    def __init__(self, shape, dtype=FAKE_BF16, device="cuda:0", numel=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = str(device).startswith("cuda")
        self.ndim = len(self.shape)
        self._numel = numel if numel is not None else _prod(self.shape)

    def numel(self):
        return self._numel


def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


class _FakeEnvs:
    """模拟 sglang.srt.environ.envs 的 SGLANG_HCU_QSA_PREFILL_KEY_SPAN。"""

    def __init__(self, key_span):
        self.SGLANG_HCU_QSA_PREFILL_KEY_SPAN = mock.Mock()
        self.SGLANG_HCU_QSA_PREFILL_KEY_SPAN.get.return_value = key_span
        self.SGLANG_HCU_QSA_PREFILL_LOGITS_MIB = mock.Mock()
        self.SGLANG_HCU_QSA_PREFILL_LOGITS_MIB.get.return_value = 0


class _Harness:
    """构造 around_mqa_prefill 运行所需的 sglang 模块替身。"""

    def __init__(self, *, key_span, is_hip, has_torch_impl, torch_raises=False):
        self.key_span = key_span
        self.is_hip = is_hip
        self.has_torch_impl = has_torch_impl
        self.torch_raises = torch_raises
        self.calls = []

    def __enter__(self):
        self._saved = {}

        def put(name, mod):
            self._saved[name] = sys.modules.get(name)
            sys.modules[name] = mod

        # sglang.srt.environ
        environ = types.ModuleType("sglang.srt.environ")
        environ.envs = _FakeEnvs(self.key_span)
        put("sglang.srt.environ", environ)

        # sglang.srt.utils.common.is_hip
        utils = types.ModuleType("sglang.srt.utils")
        common = types.ModuleType("sglang.srt.utils.common")
        common.is_hip = lambda: self.is_hip
        utils.common = common
        put("sglang.srt.utils", utils)
        put("sglang.srt.utils.common", common)

        # sglang.srt.layers.attention.qsa.mqa.torch_qsa_mqa_prefill
        layers = types.ModuleType("sglang.srt.layers")
        attn = types.ModuleType("sglang.srt.layers.attention")
        qsa = types.ModuleType("sglang.srt.layers.attention.qsa")
        mqa = types.ModuleType("sglang.srt.layers.attention.qsa.mqa")
        if self.has_torch_impl:
            def _torch_impl(q, k, starts, ends, scale):
                if self.torch_raises:
                    raise RuntimeError("torch impl unavailable")
                self.calls.append("torch")
                return "TORCH_RESULT"
            mqa.torch_qsa_mqa_prefill = _torch_impl
        layers.attention = attn
        attn.qsa = qsa
        qsa.mqa = mqa
        put("sglang.srt.layers", layers)
        put("sglang.srt.layers.attention", attn)
        put("sglang.srt.layers.attention.qsa", qsa)
        put("sglang.srt.layers.attention.qsa.mqa", mqa)

        # torch 替身：只需要 bfloat16 比较
        torch_mod = types.ModuleType("torch")
        torch_mod.bfloat16 = FAKE_BF16
        put("torch", torch_mod)

        # hcu_sglang.models.qwen.qsa_mqa_split_key.prefill
        split_pkg = types.ModuleType("hcu_sglang.models.qwen.qsa_mqa_split_key")

        def _split_prefill(q, k, starts, ends, score_scale=None, *, key_span):
            self.calls.append(f"split_key:{key_span}")
            return "SPLIT_RESULT"
        split_pkg.prefill = _split_prefill
        put("hcu_sglang.models.qwen.qsa_mqa_split_key", split_pkg)

        return self

    def __exit__(self, *exc):
        for name, old in self._saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        return False


def _matched_operands(rows=64, keys=16384):
    """满足 split_key 门槛：q (rows,4,128)、k (keys,1,128)、bf16、同设备。"""
    q = _FakeTensor((rows, 4, 128))
    k = _FakeTensor((keys, 1, 128))
    starts = _FakeTensor((rows,))
    ends = _FakeTensor((rows,))
    return q, k, starts, ends


def _unmatched_operands(rows=32, keys=4096):
    """keys < 8192，不满足 split_key 门槛 -> 会落到上游 tilelang 分支。"""
    q = _FakeTensor((rows, 4, 128))
    k = _FakeTensor((keys, 1, 128))
    starts = _FakeTensor((rows,))
    ends = _FakeTensor((rows,))
    return q, k, starts, ends


class QsaPrefillHipGuardTests(unittest.TestCase):
    def _call(self, *, key_span, is_hip, has_torch_impl, torch_raises=False,
              operands=None):
        from hcu_sglang.models.qwen.qsa_tuning import around_mqa_prefill

        q, k, starts, ends = operands or _unmatched_operands()
        original_calls = []

        def original(*a, **kw):
            original_calls.append(a)
            return "ORIGINAL"

        with _Harness(key_span=key_span, is_hip=is_hip,
                      has_torch_impl=has_torch_impl,
                      torch_raises=torch_raises) as h:
            result = around_mqa_prefill(original, q, k, starts, ends)
        return result, h.calls, original_calls

    def test_matched_shape_prefers_split_key_even_on_hip(self):
        result, calls, originals = self._call(
            key_span=4096, is_hip=True, has_torch_impl=True,
            operands=_matched_operands())
        self.assertEqual(result, "SPLIT_RESULT")
        self.assertEqual(calls, ["split_key:4096"])
        self.assertEqual(originals, [])

    def test_unmatched_shape_on_hip_uses_torch_reference(self):
        """这就是 dtk2610 崩溃的路径：形状不匹配 -> 上游会走 tilelang -> 崩。
        现在应改走 torch 参考实现。"""
        result, calls, originals = self._call(
            key_span=4096, is_hip=True, has_torch_impl=True)
        self.assertEqual(result, "TORCH_RESULT")
        self.assertEqual(calls, ["torch"])
        self.assertEqual(originals, [])

    def test_unmatched_shape_off_hip_defers_to_original(self):
        """非 HIP（CUDA）上 TileLang 可用，不应干预。"""
        result, calls, originals = self._call(
            key_span=4096, is_hip=False, has_torch_impl=True)
        self.assertEqual(result, "ORIGINAL")
        self.assertEqual(calls, [])
        self.assertEqual(len(originals), 1)

    def test_missing_torch_impl_defers_to_original(self):
        """该 sglang 版本没有 torch 参考实现时，交回上游，不假装成功。"""
        result, calls, originals = self._call(
            key_span=4096, is_hip=True, has_torch_impl=False)
        self.assertEqual(result, "ORIGINAL")
        self.assertEqual(calls, [])
        self.assertEqual(len(originals), 1)

    def test_torch_impl_error_defers_to_original(self):
        """torch 路径自身报错时不吞掉，交回上游暴露真实错误。"""
        result, calls, originals = self._call(
            key_span=4096, is_hip=True, has_torch_impl=True, torch_raises=True)
        self.assertEqual(result, "ORIGINAL")
        self.assertEqual(len(originals), 1)

    def test_key_span_zero_still_gets_hip_guard(self):
        """key_span=0 关闭分段优化，但 HIP 守卫仍应生效。"""
        result, calls, originals = self._call(
            key_span=0, is_hip=True, has_torch_impl=True)
        self.assertEqual(result, "TORCH_RESULT")
        self.assertEqual(calls, ["torch"])

    def test_invalid_key_span_raises(self):
        with self.assertRaises(ValueError):
            self._call(key_span=3000, is_hip=True, has_torch_impl=True)


class RowChunkSizeTests(unittest.TestCase):
    def test_zero_budget_defers(self):
        from hcu_sglang.models.qwen.qsa_tuning import row_chunk_size
        self.assertEqual(row_chunk_size(0, 10, 100, 4, lambda *a: "orig"), "orig")

    def test_budget_caps_rows(self):
        from hcu_sglang.models.qwen.qsa_tuning import row_chunk_size
        # 1 MiB / (4096 keys * 4B) = 64 行
        self.assertEqual(row_chunk_size(1, 1000, 4096, 4, lambda *a: 1000), 64)

    def test_invalid_budget_raises(self):
        from hcu_sglang.models.qwen.qsa_tuning import row_chunk_size
        with self.assertRaises(ValueError):
            row_chunk_size(4096, 10, 100, 4, lambda *a: 10)


if __name__ == "__main__":
    unittest.main()
