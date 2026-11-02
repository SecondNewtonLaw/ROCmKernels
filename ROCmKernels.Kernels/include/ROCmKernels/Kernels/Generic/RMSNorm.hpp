#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int rms_norm_forward(const void *x, const void *residual, const void *scale, void *output, float *rstd, int rows,
        int columns, float epsilon, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// y = s * rstd * scale with rstd = rsqrt(mean(s^2) + epsilon) over s = x + residual.
    ///
    /// residual and scale may be null, in which case the term is skipped. rstd receives one value per row and may be null.
    inline void RMSNormForward(const void *x, const void *residual, const void *scale, void *output, float *rstd, const int rows,
            const int columns, const float epsilon, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(rms_norm_forward(x, residual, scale, output, rstd, rows, columns, epsilon, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
