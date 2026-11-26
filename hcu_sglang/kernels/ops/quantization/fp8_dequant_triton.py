import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_e4m3fn_weight(Q, Scale, W, SIZE, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < SIZE
    raw = tl.load(Q + offsets, valid, other=0).to(tl.uint16)
    magnitude = raw & 127
    half_bits = (magnitude << 7) | ((raw & 128) << 8)
    decoded = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    decoded = tl.where(magnitude == 127, float("nan"), decoded)
    tl.store(W + offsets, decoded * tl.load(Scale), valid)


def dequantize_e4m3fn(weight, scale):
    restored = torch.empty_like(weight, dtype=torch.bfloat16)
    block = 2048
    size = weight.numel()
    _dequant_e4m3fn_weight[(triton.cdiv(size, block),)](
        weight.view(torch.uint8), scale, restored, size, block, enable_fp_fusion=False,
    )
    return restored
