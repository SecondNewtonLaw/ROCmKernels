"""Same-process comparison of the torch ops from _torch_ops.py against the ROCmKernels DLL.

Protocol matches the torch harness: 256 MiB cold flush before every sample,
median of 20 samples, minimum 10 ms per sample. Both sides run in one process
so the clock state is identical, and the first case is repeated at the end to
report drift.

Usage:
    F:\\rocm-test\\venv\\Scripts\\python.exe tools/TorchCompare/compare_ops.py
"""
import ctypes
import json
import sys
import time

import torch
import torch.nn.functional as F

DLL = r"F:\Coding\ROCm\ROCmKernels\build\local-release\ROCmKernels.Kernels.Shared.dll"
OUT = r"F:\scratchpad\bench\rocmk_ops.json"

torch.manual_seed(0)
DEV = "cuda"
FLUSH = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=DEV)
SAMPLES, WARMUP, MIN_SAMPLE = 20, 10, 0.010
FP32, FP16 = 0, 1
ROWS = []

lib = ctypes.CDLL(DLL)
P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_ubyte


class Strides(ctypes.Structure):
    _fields_ = [("batch", I), ("head", I), ("token", I)]


class AdamTensor(ctypes.Structure):
    _fields_ = [("parameter", P), ("gradient", P), ("first", P), ("second", P), ("elements", ctypes.c_longlong)]


def bind(name, argtypes):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = I
    return fn


linear_act_nt = bind("linear_act_nt", [P, P, P, P, I, I, I, U, P])
linear_nt = bind("linear_nt", [P, P, P, P, I, I, I, U, P])
linear_act = bind("linear_act_fast", [P, P, P, P, I, I, I, U, P])
linear_plain = bind("linear_fast", [P, P, P, P, I, I, I, U, P])
layernorm = bind("layernorm_scale_add_forward", [P, P, P, P, P, P, P, I, I, ctypes.c_float, U, P])
ce_forward = bind("cross_entropy_fp32_forward", [P, P, P, P, I, I, I, P])
adamw = bind("adamw_multi", [ctypes.POINTER(AdamTensor), I, ctypes.c_longlong, ctypes.c_float, ctypes.c_float,
                             ctypes.c_float, ctypes.c_float, ctypes.c_float, U, P])
stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    unit = max(time.perf_counter() - t, 1e-6)
    reps = max(1, int(MIN_SAMPLE / unit))
    samples = []
    for _ in range(SAMPLES):
        FLUSH.zero_()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) / reps * 1e3)
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def case(name, torch_fn, rocm_fn, per_step=1):
    t_ms, t_min = timeit(torch_fn)
    r_ms, r_min = timeit(rocm_fn)
    ROWS.append({"op": name, "torch_ms": round(t_ms, 4), "rocm_ms": round(r_ms, 4),
                 "torch_min": round(t_min, 4), "rocm_min": round(r_min, 4), "per_step": per_step})
    mark = "WIN " if r_ms <= t_ms else "LOSS"
    print(f"  {name:<44} torch {t_ms:8.3f}  ours {r_ms:8.3f}  {r_ms / t_ms:5.2f}x  {mark}"
          f"   x{per_step:<3} {((t_ms - r_ms) * per_step):+7.2f} ms/step", flush=True)
    with open(OUT, "w") as f:
        json.dump(ROWS, f, indent=1)


# ------------------------------------------------------------------ ffn, d=128 ff=8192 at 8192 tokens
P_, D, E = 8192, 128, 8192
x = torch.randn(P_, D, device=DEV, dtype=torch.float16)
w1 = torch.randn(E, D, device=DEV, dtype=torch.float16) * 0.02
w2 = torch.randn(D, E, device=DEV, dtype=torch.float16) * 0.02
w_qkv = torch.randn(3 * D, D, device=DEV, dtype=torch.float16) * 0.02
w_o = torch.randn(D, D, device=DEV, dtype=torch.float16) * 0.02
h = torch.empty(P_, E, device=DEV, dtype=torch.float16)
w1t = w1.t().contiguous()
w2t = w2.t().contiguous()
w_qkvt = w_qkv.t().contiguous()
w_ot = w_o.t().contiguous()
y = torch.empty(P_, D, device=DEV, dtype=torch.float16)

case("w1 linear+silu 8192x128 -> 8192",
     lambda: F.silu(F.linear(x, w1)),
     lambda: linear_act(ptr(x), ptr(w1t), None, ptr(h), P_, E, D, FP16, stream), 24)

case("w2 linear 8192x8192 -> 128",
     lambda: F.linear(h, w2),
     lambda: linear_plain(ptr(h), ptr(w2t), None, ptr(y), P_, D, E, FP16, stream), 24)

case("qkv linear 8192x128 -> 384",
     lambda: F.linear(x, w_qkv),
     lambda: linear_plain(ptr(x), ptr(w_qkvt), None, ptr(y), P_, 3 * D, D, FP16, stream), 24)

case("o linear 8192x128 -> 128",
     lambda: F.linear(x, w_o),
     lambda: linear_plain(ptr(x), ptr(w_ot), None, ptr(y), P_, D, D, FP16, stream), 24)

# ------------------------------------------------------------------ FFN unfused pair, full and 128-row tiles
h1 = torch.empty(8192, E, device=DEV, dtype=torch.float16)
y1 = torch.empty(8192, D, device=DEV, dtype=torch.float16)
case("ffn up full 8192x128 -> 8192",
     lambda: F.silu(F.linear(x, w1)),
     lambda: linear_act(ptr(x), ptr(w1t), None, ptr(h1), 8192, E, D, FP16, stream), 1)
case("ffn down full 8192x8192 -> 128",
     lambda: F.linear(h1, w2),
     lambda: linear_plain(ptr(h1), ptr(w2t), None, ptr(y1), 8192, D, E, FP16, stream), 1)
case("ffn up 128-row tile",
     lambda: F.silu(F.linear(x[:128], w1)),
     lambda: linear_act(ptr(x), ptr(w1t), None, ptr(h1), 128, E, D, FP16, stream), 1)
case("ffn down 128-row tile",
     lambda: F.linear(h1[:128], w2),
     lambda: linear_plain(ptr(h1), ptr(w2t), None, ptr(y1), 128, D, E, FP16, stream), 1)
case("ffn up 2048-row tile",
     lambda: F.silu(F.linear(x[:2048], w1)),
     lambda: linear_act(ptr(x), ptr(w1t), None, ptr(h1), 2048, E, D, FP16, stream), 1)
case("ffn down 2048-row tile",
     lambda: F.linear(h1[:2048], w2),
     lambda: linear_plain(ptr(h1), ptr(w2t), None, ptr(y1), 2048, D, E, FP16, stream), 1)

# ------------------------------------------------------------------ layer norm block op
res = torch.randn(P_, D, device=DEV, dtype=torch.float16)
ls = torch.randn(D, device=DEV, dtype=torch.float16)
n_w = torch.ones(D, device=DEV, dtype=torch.float16)
n_b = torch.zeros(D, device=DEV, dtype=torch.float16)
ln_out = torch.empty(P_, D, device=DEV, dtype=torch.float16)
ln_stats = torch.empty(P_ * 2, device=DEV, dtype=torch.float32)

case("layernorm scale-add fwd 8192x128 fp16",
     lambda: F.layer_norm(x + ls * res, (D,), n_w, n_b, 1e-5),
     lambda: layernorm(ptr(x), ptr(res), ptr(ls), ptr(n_w), ptr(n_b), ptr(ln_out), ptr(ln_stats), P_, D,
                       ctypes.c_float(1e-5), FP16, stream), 48)

# ------------------------------------------------------------------ losses
MASKED, V = 1200, 55765
logits = torch.randn(MASKED, V, device=DEV, dtype=torch.float32)
targets = torch.randint(0, V, (MASKED,), device=DEV)
losses = torch.empty(MASKED, device=DEV, dtype=torch.float32)
ce_stats = torch.empty(MASKED * 2, device=DEV, dtype=torch.float32)
targets64 = targets.to(torch.int64)

case("cross entropy 1200x55765 fwd (fp32)",
     lambda: F.cross_entropy(logits, targets, reduction="none"),
     lambda: ce_forward(ptr(logits), ptr(targets64), ptr(losses), ptr(ce_stats), MASKED, V, V, stream), 1)

# ------------------------------------------------------------------ adamw, 32.9M params
sizes = [V * 128, 12 * D * E, 12 * E * D, 12 * D * 3 * D]
params = [torch.nn.Parameter(torch.randn(n, device=DEV)) for n in sizes]
for p in params:
    p.grad = torch.randn_like(p)
opt = torch.optim.AdamW([{"params": params, "weight_decay": 0.0}], lr=1e-4)
m = [torch.zeros(n, device=DEV) for n in sizes]
v = [torch.zeros(n, device=DEV) for n in sizes]
descriptors = (AdamTensor * len(sizes))(*[
    AdamTensor(ctypes.c_void_p(params[i].data_ptr()), ctypes.c_void_p(params[i].grad.data_ptr()),
               ctypes.c_void_p(m[i].data_ptr()), ctypes.c_void_p(v[i].data_ptr()), sizes[i]) for i in range(len(sizes))])

case(f"adamw fp32 {sum(sizes) / 1e6:.1f}M params",
     lambda: opt.step(),
     lambda: adamw(descriptors, len(sizes), 1, ctypes.c_float(1e-4), ctypes.c_float(0.9), ctypes.c_float(0.999),
                   ctypes.c_float(1e-8), ctypes.c_float(0.0), FP32, stream), 1)

# ------------------------------------------------------------------ drift check
case("w1 linear+silu (repeat)",
     lambda: F.silu(F.linear(x, w1)),
     lambda: linear_act(ptr(x), ptr(w1t), None, ptr(h), P_, E, D, FP16, stream), 24)
case("w2 linear (repeat)",
     lambda: F.linear(h, w2),
     lambda: linear_plain(ptr(h), ptr(w2t), None, ptr(y), P_, D, E, FP16, stream), 24)

print(f"\nwrote {OUT}")



