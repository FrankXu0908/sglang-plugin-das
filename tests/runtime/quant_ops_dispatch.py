# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hcu_sglang.layers.quantization import quant_ops as C
from lightop import op as O


class SmoothAsmDispatchTests(unittest.TestCase):
    def setUp(self):C._w8a8_smooth_asm_supported.cache_clear()
    def tearDown(self):C._w8a8_smooth_asm_supported.cache_clear()
    def args(self,m=17,n=128,k=256):
        return torch.ones(m,k,dtype=torch.int8),torch.ones(n,k,dtype=torch.int8).T,torch.ones(m,1),torch.ones(n,1)

    def test_empty_token_batch_returns_empty_output_without_launching_kernels(self):
        a,b,sa,sb=self.args(m=0,n=512,k=4096)
        for dtype in (torch.bfloat16,torch.float16):
            for bias,config in ((None,None),(torch.ones(512,dtype=dtype),[1,2,3])):
                with self.subTest(dtype=dtype,bias=bias is not None),patch.object(C,'_is_hcu',True),patch.object(O,'gemm_w8a8_smooth_asm',side_effect=AssertionError('empty ASM launch')),patch.object(C.quant_ops,'triton_scaled_mm',side_effect=AssertionError('empty Triton launch')):
                    out=C.triton_scaled_mm(a,b,sa,sb,dtype,bias,config)
                    self.assertEqual(out.shape,(0,512));self.assertEqual(out.dtype,dtype);self.assertEqual(out.device,a.device)

    def test_architecture_probe_rejects_fatal_unsupported_or_unknown_and_caches(self):
        device=torch.device('cuda',0)
        for arch,result in [('gfx928',False),('gfx938:sramecc+',True),('',False)]:
            C._w8a8_smooth_asm_supported.cache_clear()
            with patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(gcnArchName=arch)) as props:
                self.assertEqual(C._w8a8_smooth_asm_supported(device),result);self.assertEqual(C._w8a8_smooth_asm_supported(device),result)
                self.assertEqual(props.call_count,1)
        C._w8a8_smooth_asm_supported.cache_clear()
        with patch.object(torch.cuda,'get_device_properties',side_effect=RuntimeError('probe')):
            self.assertFalse(C._w8a8_smooth_asm_supported(device))

    def test_source_slide_selection_and_complete_provider_contract(self):
        for m,n,tune in [(17,128,1),(240,768,1),(128,1536,1),(32,4096,9)]:
            a,b,sa,sb=self.args(m,n)
            out=torch.zeros(m,n,dtype=torch.bfloat16)
            with patch.object(C,'_is_hcu',True),patch.object(C,'_w8a8_smooth_asm_supported',return_value=True),patch.object(O,'gemm_w8a8_smooth_asm',return_value=out) as asm,patch.object(C.quant_ops,'triton_scaled_mm',side_effect=AssertionError('triton')):
                self.assertIs(C.triton_scaled_mm(a,b,sa,sb,torch.bfloat16),out)
            args=asm.call_args.args;self.assertIs(args[0],a);self.assertIs(args[1],b)
            self.assertEqual(args[4:],(None,torch.bfloat16,m,n,256,True,tune))

    def test_tuned_configuration_bias_partial_k_and_large_m_preserve_triton(self):
        variants=[(self.args(),torch.ones(128),None),(self.args(),None,[1,2,3]),(self.args(k=128),None,None),(self.args(m=129,n=1536),None,None)]
        for (a,b,sa,sb),bias,config in variants:
            with patch.object(C,'_is_hcu',True),patch.object(C,'_w8a8_smooth_asm_supported',return_value=True),patch.object(O,'gemm_w8a8_smooth_asm',side_effect=AssertionError('asm')),patch.object(C.quant_ops,'triton_scaled_mm',return_value='triton') as triton:
                self.assertEqual(C.triton_scaled_mm(a,b,sa,sb,torch.bfloat16,bias,config),'triton')
            self.assertIs(triton.call_args.args[-2],bias);self.assertIs(triton.call_args.args[-1],config)

    def test_architecture_rejection_does_not_call_fatal_provider(self):
        a,b,sa,sb=self.args()
        with patch.object(C,'_is_hcu',True),patch.object(C,'_w8a8_smooth_asm_supported',return_value=False),patch.object(O,'gemm_w8a8_smooth_asm',side_effect=AssertionError('fatal')),patch.object(C.quant_ops,'triton_scaled_mm',return_value='safe'):
            self.assertEqual(C.triton_scaled_mm(a,b,sa,sb,torch.bfloat16),'safe')

    def test_source_provider_exception_uses_original_triton_operands(self):
        a,b,sa,sb=self.args()
        with patch.object(C,'_is_hcu',True),patch.object(C,'_w8a8_smooth_asm_supported',return_value=True),patch.object(O,'gemm_w8a8_smooth_asm',side_effect=RuntimeError('dispatch miss')),patch.object(C.quant_ops,'triton_scaled_mm',return_value='safe') as triton:
            self.assertEqual(C.triton_scaled_mm(a,b,sa,sb,torch.bfloat16),'safe')
        for received,expected in zip(triton.call_args.args[:4],[a,b,sa,sb]):self.assertIs(received,expected)

    def test_non_hcu_skips_asm(self):
        a,b,sa,sb=self.args()
        with patch.object(C,'_is_hcu',False),patch.object(O,'gemm_w8a8_smooth_asm',side_effect=AssertionError('asm')),patch.object(C.quant_ops,'triton_scaled_mm',return_value='native'):
            self.assertEqual(C.triton_scaled_mm(a,b,sa,sb,torch.bfloat16),'native')


if __name__=='__main__':unittest.main()
