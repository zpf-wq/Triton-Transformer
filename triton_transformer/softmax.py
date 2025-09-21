import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl
from triton_transformer.utils import calc_num_warps


@triton.jit
def softmax_kernel_forward(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    CAUSAL: tl.constexpr,       # 编译期常量
    BLOCK_SIZE: tl.constexpr,   # 编译期常量
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    # 读入：越界位置设为 -inf，保证 softmax 后为 0
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    if CAUSAL:
        # 对未来位置打 -inf 掩码（按最后一维的序号做因果裁剪）
        row_pos = row_idx % n_cols
        causal_mask = col_offsets > row_pos
        row = row + tl.where(causal_mask, -float('inf'), 0.0)

    # 数值稳定
    row_minus_max = row - tl.max(row, axis=0)

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0) + 1e-20  # 防止全被 mask 时除 0
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def softmax_kernel_backward(
    output_ptr,
    input_ptr,      # 概率 y
    grad_ptr,       # 上游梯度 dL/dy
    grad_row_stride,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,   # 编译期常量
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    probs_row = tl.load(input_ptrs, mask=mask, other=0.0)  # y
    grad_row = tl.load(grad_ptrs, mask=mask, other=0.0)    # dL/dy

    # dX = (dY - sum(dY * Y)) * Y
    dxhat = probs_row * grad_row
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis=0)

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_grad_output, mask=mask)


class _softmax(autograd.Function):
    @classmethod
    def forward(self, ctx, x, causal):
        shape = x.shape
        x = x.contiguous().view(-1, shape[-1])
        n_rows, n_cols = x.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        y = torch.empty_like(x)

        softmax_kernel_forward[(n_rows,)](
            y,
            x,
            x.stride(0),
            y.stride(0),
            n_cols,
            CAUSAL=causal,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        y_full = y.view(*shape)
        if x.requires_grad:
            ctx.save_for_backward(y_full)
        return y_full

    @classmethod
    def backward(self, ctx, grad_probs):
        (probs,) = ctx.saved_tensors
        shape = grad_probs.shape
        grad_probs = grad_probs.contiguous().view(-1, grad_probs.shape[-1])
        probs = probs.contiguous().view(-1, probs.shape[-1])

        n_rows, n_cols = grad_probs.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calc_num_warps(BLOCK_SIZE)

        dx = torch.empty_like(probs)

        softmax_kernel_backward[(n_rows,)](
            dx,
            probs,
            grad_probs,
            grad_probs.stride(0),
            probs.stride(0),
            dx.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        return dx.view(*shape), None


triton_softmax = _softmax.apply


def softmax(x, causal=False, use_triton=False):
    if use_triton:
        return triton_softmax(x, causal)
    else:
        # 非 triton 路径：保持与原逻辑一致
        if causal:
            # 下三角掩码
            L = x.shape[-1]
            mask = torch.ones((L, L), device=x.device, dtype=torch.bool).tril()
            x = x.masked_fill(~mask, float('-inf'))
        return F.softmax(x, dim=-1)