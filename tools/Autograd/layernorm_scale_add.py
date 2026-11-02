"""Autograd Function for the fused LayerNormScaleAdd kernel.

The forward saves the per-row {mean, rstd} statistics produced by the kernel so
the backward does not recompute them. The backward returns five gradients:
dx, dz, dscale, dweight and dbias.

    y = layer_norm(x + scale * z, weight, bias, epsilon)

x and z are [rows, columns]; scale, weight and bias are [columns]. The kernel
accumulates in fp32. Parameter gradients are fp32; x/z gradients use the input
dtype.
"""
import ctypes
import os

import torch
import torch.nn.functional as F

DLL = os.environ.get(
    "ROCMKERNELS_DLL",
    r"F:\Coding\ROCm\ROCmKernels\build\local-release\ROCmKernels.Kernels.Shared.dll",
)
P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_ubyte
lib = ctypes.CDLL(DLL)


def _bind(name, argtypes):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = I
    return fn


_forward = _bind(
    "layernorm_scale_add_forward",
    [P, P, P, P, P, P, P, I, I, ctypes.c_float, U, P],
)
_backward = _bind(
    "layernorm_scale_add_backward",
    [P, P, P, P, P, P, P, P, P, P, P, I, I, U, P],
)


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr()) if tensor is not None else ctypes.c_void_p(None)


def _dtype_code(dtype):
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    raise TypeError(f"LayerNormScaleAdd only supports fp32 and fp16, got {dtype}")


class LayerNormScaleAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, scale, weight, bias, epsilon):
        if x.shape != z.shape:
            raise ValueError("x and z must have the same shape")
        if x.dim() != 2:
            raise ValueError("x and z must be [rows, columns]")
        columns = x.shape[1]
        if scale.numel() != columns:
            raise ValueError(f"scale must have {columns} elements")
        if weight is not None and weight.numel() != columns:
            raise ValueError(f"weight must have {columns} elements")
        if bias is not None and bias.numel() != columns:
            raise ValueError(f"bias must have {columns} elements")
        if not x.is_contiguous() or not z.is_contiguous() or not scale.is_contiguous():
            raise ValueError("x, z and scale must be contiguous")
        if weight is not None and not weight.is_contiguous():
            raise ValueError("weight must be contiguous")
        if bias is not None and not bias.is_contiguous():
            raise ValueError("bias must be contiguous")
        for tensor in (z, scale, weight, bias):
            if tensor is not None and tensor.dtype != x.dtype:
                raise TypeError("x, z, scale, weight and bias must share a dtype")
        if not x.is_cuda:
            raise ValueError("LayerNormScaleAdd runs on CUDA tensors")

        rows = x.shape[0]
        output = torch.empty_like(x)
        statistics = torch.empty(rows * 2, dtype=torch.float32, device=x.device)
        _forward(
            _ptr(x), _ptr(z), _ptr(scale), _ptr(weight), _ptr(bias), _ptr(output), _ptr(statistics),
            rows, columns, float(epsilon), _dtype_code(x.dtype),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        ctx.save_for_backward(x, z, scale, weight, bias, statistics)
        ctx.rows = rows
        ctx.columns = columns
        ctx.dtype = x.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, z, scale, weight, bias, statistics = ctx.saved_tensors
        rows, columns = ctx.rows, ctx.columns
        upstream = grad_output.contiguous()
        if upstream.dtype != ctx.dtype:
            upstream = upstream.to(ctx.dtype)

        gradient_x = torch.empty_like(x)
        gradient_z = torch.empty_like(z)
        gradient_scale = torch.empty(columns, dtype=torch.float32, device=x.device)
        gradient_weight = torch.empty(columns, dtype=torch.float32, device=x.device) if weight is not None else None
        gradient_bias = torch.empty(columns, dtype=torch.float32, device=x.device) if bias is not None else None

        _backward(
            _ptr(x), _ptr(z), _ptr(scale), _ptr(weight), _ptr(statistics), _ptr(upstream),
            _ptr(gradient_x), _ptr(gradient_z), _ptr(gradient_scale), _ptr(gradient_weight), _ptr(gradient_bias),
            rows, columns, _dtype_code(ctx.dtype),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        return gradient_x, gradient_z, gradient_scale, gradient_weight, gradient_bias, None


def layer_norm_scale_add(x, z, scale, weight=None, bias=None, epsilon=1e-5):
    return LayerNormScaleAdd.apply(x, z, scale, weight, bias, epsilon)


def _reference(x, z, scale, weight, bias, epsilon):
    combined = x.float() + scale.float() * z.float()
    return F.layer_norm(combined, (combined.shape[1],), weight.float() if weight is not None else None,
                        bias.float() if bias is not None else None, epsilon)


def _rel(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _check(dtype, rows=8192, columns=128, affine=True):
    torch.manual_seed(0)
    x = torch.randn(rows, columns, device="cuda", dtype=dtype) * 0.5
    z = torch.randn(rows, columns, device="cuda", dtype=dtype) * 0.5
    scale = torch.rand(columns, device="cuda", dtype=dtype) + 0.5
    weight = torch.rand(columns, device="cuda", dtype=dtype) + 0.5 if affine else None
    bias = (torch.rand(columns, device="cuda", dtype=dtype) - 0.5) if affine else None
    for tensor in (x, z, scale, weight, bias):
        if tensor is not None:
            tensor.requires_grad_(True)

    output = layer_norm_scale_add(x, z, scale, weight, bias, 1e-5)
    expected = _reference(x, z, scale, weight, bias, 1e-5)
    output_error = _rel(output, expected)

    inputs = [x, z, scale] + ([weight, bias] if affine else [])
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, inputs, upstream.float())
    grads = {"dx": _rel(actual[0], expected_grads[0]), "dz": _rel(actual[1], expected_grads[1]),
             "dscale": _rel(actual[2], expected_grads[2])}
    if affine:
        grads["dweight"] = _rel(actual[3], expected_grads[3])
        grads["dbias"] = _rel(actual[4], expected_grads[4])
    print(f"{dtype} affine={affine} rows={rows} cols={columns}: output {output_error:.2e} " +
          " ".join(f"{name} {value:.2e}" for name, value in grads.items()))


if __name__ == "__main__":
    _check(torch.float32)
    _check(torch.float16)
    _check(torch.float32, rows=257, columns=100)
    _check(torch.float16, rows=257, columns=100)
    _check(torch.float32, affine=False)
    _check(torch.float16, affine=False)


