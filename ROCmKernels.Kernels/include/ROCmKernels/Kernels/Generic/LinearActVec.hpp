#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int linear_act_vec(const void *input, const void *weight, const void *bias, void *output, int columns, int inner,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// output[c] = silu(sum_k input[k] * weight[c][k] + bias[c]); bias may be null.
    inline void LinearActVec(const void *input, const void *weight, const void *bias, void *output, const int columns, const int inner,
            const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(linear_act_vec(input, weight, bias, output, columns, inner, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
