#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

// The pieces a training step needs on top of the forward GEMM: the fp16
// transpose that turns dy into dy^T for the weight gradient, and the SiLU
// derivative that turns dh into dz.
extern "C" ROCMKERNELS_EXPORT int transpose_fp16(const void *source, void *destination, int rows, int columns, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int silu_backward_fp16(const void *dh, const void *z, void *dz, void *h, float *dbias, int rows,
        int columns, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// destination[c][r] = source[r][c], fp16.
    inline void TransposeFp16(const void *source, void *destination, const int rows, const int columns, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(transpose_fp16(source, destination, rows, columns, stream));
    }

    /// h = silu(z) and dz = dh * silu'(z); h and dbias may be null.
    /// dbias[c] = sum_r dz[r][c], which is the bias gradient of the linear that produced z.
    inline void SiluBackwardFp16(const void *dh, const void *z, void *dz, void *h, float *dbias, const int rows, const int columns,
            hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(silu_backward_fp16(dh, z, dz, h, dbias, rows, columns, stream));
    }
} // namespace ROCmKernels::Kernels

