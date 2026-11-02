#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int linear_act_fast(const void *input, const void *weight, const void *bias, void *output, int rows, int columns,
        int inner, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// output[r][c] = silu(sum_k input[r][k] * weight[k][c] + bias[c]); bias may be null.
    inline void LinearActFast(const void *input, const void *weight, const void *bias, void *output, const int rows, const int columns,
            const int inner, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(linear_act_fast(input, weight, bias, output, rows, columns, inner, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
