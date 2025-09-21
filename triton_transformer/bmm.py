# triton_transformer/bmm.py
import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl


# (relu(x))^2 的内联实现会直接写在 kernel 里，通过 constexpr 布尔开关控制
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    x_ptr, y_ptr, o_ptr,                  # [B, M, K], [B, K, N], [B, M, N]
    M, N, K,                              # problem sizes
    stride_al, stride_am, stride_ak,      # x strides (B, M, K)
    stride_bl, stride_bk, stride_bn,      # y strides (B, K, N)
    stride_ol, stride_om, stride_on,      # o strides (B, M, N)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    RELU_SQUARED: tl.constexpr,           # 布尔开关：是否应用 (relu(x))^2
):
    pid_batch = tl.program_id(0)
    pid       = tl.program_id(1)

    # tile 排布（swizzle 改善 L2 命中）
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak + pid_batch * stride_al)
    y_ptrs = y_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn + pid_batch * stride_bl)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k = 0
    while k < K:
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0
        )
        y = tl.load(
            y_ptrs,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )

        # 关键改动：把 tile 转成半精度做 dot，acc 仍是 fp32
        x = x.to(tl.float16)    # 若想用 bf16，可改成 tl.bfloat16
        y = y.to(tl.float16)

        acc += tl.dot(x, y)
        k += BLOCK_SIZE_K
        x_ptrs += BLOCK_SIZE_K * stride_ak
        y_ptrs += BLOCK_SIZE_K * stride_bk

    if RELU_SQUARED:
        acc = tl.maximum(acc, 0.0)
        acc = acc * acc

    o_ptrs = o_ptr + (stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + stride_ol * pid_batch)
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_bmm(x: torch.Tensor, y: torch.Tensor, relu_squared: bool = False) -> torch.Tensor:
    """
    x: [B, M, K]
    y: [B, K, N] or [K, N] (broadcast to B)
    relu_squared: 是否在 matmul 后内联应用 (relu(..))^2
    """
    B, M, K = x.shape
    if y.ndim == 2:
        y = y.unsqueeze(0).expand(B, -1, -1)
    _, K2, N = y.shape
    assert K == K2, "Inner dimensions must match (K)"
    assert (K % 32 == 0), "K must be divisible by 32 for current configs."

    o = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

    # autotune 的 grid 回调可读取已选 config 的 BLOCK_SIZE_M/N
    grid = lambda META: (
        B,
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N'])
    )

    bmm_kernel[grid](
        x, y, o,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        # 注意：不要在这里再传 BLOCK_SIZE_* / GROUP_SIZE_M（它们由 autotune config 注入）
        RELU_SQUARED=relu_squared,
    )
    return o


class _relu_squared(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, w):
        # kernel 内直接完成 matmul + relu^2
        o = triton_bmm(x, w, relu_squared=True)
        if x.requires_grad or w.requires_grad:
            ctx.save_for_backward(x, w, o)
        return o

    @classmethod
    def backward(cls, ctx, dy):
        x, w, o = ctx.saved_tensors
        # o = relu(x@w)^2 >= 0
        # d/dz relu(z)^2 = 2*relu(z)*1_{z>0}，而 relu(z) = sqrt(o)
        dy_scaled = torch.sqrt(o.clamp_min(0)).mul_(2.0).mul_(dy)
        dx = triton_bmm(dy_scaled, w.transpose(-1, -2), relu_squared=False)
        dw = triton_bmm(x.transpose(-1, -2), dy_scaled, relu_squared=False)
        return dx, dw


triton_relu_squared = _relu_squared.apply


def fused_relu_squared(x: torch.Tensor, w: torch.Tensor, use_triton: bool = False) -> torch.Tensor:
    if use_triton:
        return triton_relu_squared(x, w)
    return F.relu(x @ w) ** 2