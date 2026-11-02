"""Validate the Muon kernels against a torch implementation of reactive.MuonNS.step.

Loads ROCmKernels.Kernels.Shared.dll, runs prepare -> 5 x (gram_split, a@a,
make_b, y@B) -> apply, and compares the parameter against the torch reference
from the same initial state. Prints the per-stage error so a mismatch can be
localised.

    F:\\rocm-test\\venv\\Scripts\\python.exe tools/MuonValidate/validate.py
    ... validate.py --shapes 128x128,384x128,8192x128,128x8192
"""
import argparse
import ctypes
import math

import torch

DLL = r"F:\Coding\ROCm\ROCmKernels\build\local-release\ROCmKernels.Kernels.Shared.dll"
MOMENTUM, WEIGHT_DECAY, LR = 0.95, 0.01, 1e-2
C3, C4, C5 = 3.4445, -4.7750, 2.0315
SPLITS = 8

P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_ubyte

lib = ctypes.CDLL(DLL)
prepare = lib.muon_fp32_prepare
prepare.argtypes = [P, P, P, P, P, I, I, I, ctypes.c_float, P]
prepare.restype = I
gram = lib.muon_fp32_gram
gram.argtypes = [P, P, P, P, I, I, I, ctypes.c_float, ctypes.c_float, ctypes.c_float, I, P]
gram.restype = I
gram_split = lib.muon_fp32_gram_split
gram_split.argtypes = [P, P, P, P, P, I, I, I, ctypes.c_float, ctypes.c_float, ctypes.c_float, I, I, P]
gram_split.restype = I
multiply = lib.muon_fp32_multiply
multiply.argtypes = [P, P, P, P, I, I, I, I, P]
multiply.restype = I
make_b = lib.muon_fp32_make_b
make_b.argtypes = [P, P, P, I, I, ctypes.c_float, ctypes.c_float, ctypes.c_float, P]
make_b.restype = I
apply_kernel = lib.muon_fp32_apply
apply_kernel.argtypes = [P, P, P, P, P, I, I, I, I, I, ctypes.c_float, ctypes.c_float, P]
apply_kernel.restype = I
STREAM = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


class Group:
    """One shape group: count matrices of [rows][columns]."""

    def __init__(self, rows, columns, count, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.rows, self.columns, self.count = rows, columns, count
        self.tall = rows >= columns
        self.m, self.n = (rows, columns) if self.tall else (columns, rows)
        self.parameter = torch.randn(count, rows, columns, device="cuda", generator=g)
        self.gradient = torch.randn(count, rows, columns, device="cuda", generator=g)
        self.buffer = torch.zeros(count, rows, columns, device="cuda")
        self.workspace = torch.empty(count, rows, columns, device="cuda")
        self.sumsq = torch.zeros(count, device="cuda")
        self.flags = torch.zeros(count, device="cuda", dtype=torch.int32)
        self.identity = torch.eye(self.n, device="cuda").expand(count, self.n, self.n).contiguous()
        side = max(1, self.n // 128)
        self.partials = torch.empty(count * SPLITS * side * side * 128 * 128, device="cuda")
        self.a = torch.empty(count, self.n, self.n, device="cuda")
        self.a2 = torch.empty(count, self.n, self.n, device="cuda")
        self.b = torch.empty(count, self.n, self.n, device="cuda")
        self.parameter_ptrs = self._pointers(self.parameter)
        self.gradient_ptrs = self._pointers(self.gradient)
        self.buffer_ptrs = self._pointers(self.buffer)

    def _pointers(self, tensor):
        return torch.tensor([tensor[i].data_ptr() for i in range(self.count)], device="cuda", dtype=torch.int64)

    def snapshot(self):
        return (self.parameter.clone(), self.gradient.clone(), self.buffer.clone())

    def restore(self, state):
        self.parameter.copy_(state[0])
        self.gradient.copy_(state[1])
        self.buffer.copy_(state[2])


def reference(group, state):
    """reactive.MuonNS.step for one shape group, batched over the count."""
    parameter, gradient, buffer = state
    buffer = buffer.lerp(gradient, 1 - MOMENTUM)
    parameter = parameter * (1 - LR * WEIGHT_DECAY)
    z = gradient.lerp(buffer, MOMENTUM)
    x = z.clone()
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2)
    x = x / (x.flatten(1).norm(dim=1)[:, None, None] + 1e-7)
    for _ in range(5):
        a = x @ x.transpose(1, 2)
        x = C3 * x + (C4 * a + C5 * (a @ a)) @ x
    u = x.transpose(1, 2) if transposed else x
    return parameter + u * (-LR * 0.2 * math.sqrt(max(group.rows, group.columns))), buffer, z


def run_kernels(group, report=None):
    m, n, c = group.m, group.n, group.count
    # prepare takes the parameter's own [rows][columns] and transposes internally.
    prepare(ptr(group.gradient_ptrs), ptr(group.buffer_ptrs), ptr(group.workspace), ptr(group.flags), ptr(group.sumsq), c, group.rows,
            group.columns, MOMENTUM, STREAM)
    if report is not None:
        report["workspace"] = group.workspace.view(c, group.m, group.n).clone()
        report["sumsq"] = group.sumsq.clone()
    for iteration in range(5):
        scaled = iteration == 0
        gram_split(ptr(group.workspace), None, ptr(group.a), ptr(group.partials), ptr(group.sumsq), c, m, n, 1.0, 0.0, 0.0, scaled,
                   SPLITS if m >= 1024 else 1, STREAM)
        if report is not None and iteration == 0:
            report["a"] = group.a.clone()
        multiply(ptr(group.a), ptr(group.a), ptr(group.a2), ptr(group.sumsq), c, n, n, 0, STREAM)
        make_b(ptr(group.a), ptr(group.a2), ptr(group.b), c, n, C3, C4, C5, STREAM)
        multiply(ptr(group.workspace), ptr(group.b), ptr(group.workspace), ptr(group.sumsq), c, m, n, scaled, STREAM)
        if report is not None and iteration == 0:
            report["x1"] = group.workspace.view(c, m, n).clone()
    apply_kernel(ptr(group.workspace), ptr(group.identity), ptr(group.parameter_ptrs), ptr(group.flags), ptr(group.sumsq), c, m, n, 0,
                 0 if group.tall else 1, LR * WEIGHT_DECAY, LR * 0.2 * math.sqrt(max(group.rows, group.columns)), STREAM)


def check(rows, columns, count, seed):
    group = Group(rows, columns, count, seed)
    state = group.snapshot()
    expected, expected_buffer, expected_z = reference(group, state)
    report = {}
    run_kernels(group, report)
    torch.cuda.synchronize()

    parameter_error = ((group.parameter - expected).norm() / expected.norm()).item()
    buffer_error = ((group.buffer - expected_buffer).norm() / expected_buffer.norm()).item()
    z = expected_z.view(count, group.rows, group.columns)
    if not group.tall:
        z = z.transpose(1, 2)
    workspace_error = ((report["workspace"] - z).norm() / z.norm()).item()
    x = z.contiguous()
    x = x / (x.flatten(1).norm(dim=1)[:, None, None] + 1e-7)
    reference_a = x.transpose(1, 2) @ x
    gram_error = ((report["a"] - reference_a).norm() / reference_a.norm()).item()
    reference_x1 = C3 * x + x @ (C4 * reference_a + C5 * (reference_a @ reference_a))
    x1_error = ((report["x1"] - reference_x1).norm() / reference_x1.norm()).item()
    print(f"{count:2d}x{rows}x{columns}: param {parameter_error:.2e}  buffer {buffer_error:.2e}  "
          f"stage: workspace {workspace_error:.2e}  gram {gram_error:.2e}  x1 {x1_error:.2e}")
    return parameter_error


def check_oracle(root):
    """Compare the torch reference in this file against retriever.reactive.MuonNS."""
    import sys

    sys.path.insert(0, root)
    from retriever.reactive import MuonNS

    for rows, columns, seed in ((128, 128, 0), (8192, 128, 1), (128, 128, 2)):
        torch.manual_seed(seed)
        parameter = torch.randn(rows, columns) * 0.02
        gradient = torch.randn(rows, columns)
        buffer = torch.zeros(rows, columns)
        expected, expected_buffer, _ = reference(
            type("Group", (), {"rows": rows, "columns": columns})(),
            (parameter.unsqueeze(0), gradient.unsqueeze(0), buffer.unsqueeze(0)))
        expected, expected_buffer = expected[0], expected_buffer[0]

        live = torch.nn.Parameter(parameter.clone())
        live.grad = gradient.clone()
        optimizer = MuonNS([{"params": [live]}], lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
        state = optimizer.state[live]
        state["momentum"] = buffer.clone()
        state["g2"] = torch.zeros((), device=live.device)
        optimizer.step()

        parameter_error = ((expected - live.detach()).norm() / live.detach().norm()).item()
        buffer_error = ((expected_buffer - state["momentum"]).norm() / state["momentum"].norm()).item()
        print(f"oracle {rows}x{columns}: reference vs reactive.MuonNS  param {parameter_error:.2e}  buffer {buffer_error:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="128x128,384x128,8192x128,128x8192,256x128,128x1024")
    parser.add_argument("--oracle", default="", help="path to the repo containing retriever/reactive.py")
    args = parser.parse_args()
    if args.oracle:
        check_oracle(args.oracle)
    worst = 0.0
    for shape in args.shapes.split(","):
        rows, columns = (int(value) for value in shape.split("x"))
        count = 1 if rows * columns > 4_000_000 else 4
        worst = max(worst, check(rows, columns, count, 1234))
    print(f"worst parameter error {worst:.2e}")


if __name__ == "__main__":
    main()





