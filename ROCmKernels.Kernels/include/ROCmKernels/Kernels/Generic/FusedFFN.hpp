#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

// Fused FFN: out = (silu(x @ w1^T + b1) @ w2^T) + b2, with the hidden activation
// kept in shared memory. x is [rows][128] and out is [rows][128]; w1 is stored
// K-major as [128][hidden] and w2 as [hidden][128].
extern "C" ROCMKERNELS_EXPORT int ffn_up_silu_down(const void *x, const void *w1, const void *b1, const void *w2, const void *b2, void *out,
        int rows, int hidden, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// out[r][o] = sum_h silu(sum_k x[r][k] * w1[k][h] + b1[h]) * w2[h][o] + b2[o].
    /// b1 and b2 may be null. Width and output are fixed at 128, hidden must be a multiple of 64.
    inline void FfnUpSiluDown(const void *x, const void *w1, const void *b1, const void *w2, const void *b2, void *out, const int rows,
            const int hidden, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(ffn_up_silu_down(x, w1, b1, w2, b2, out, rows, hidden, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
