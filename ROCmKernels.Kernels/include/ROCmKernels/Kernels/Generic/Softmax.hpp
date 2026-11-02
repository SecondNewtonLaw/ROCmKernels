#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int softmax_forward(const void *input, const bool *mask, void *output, int rows, int columns, float scale,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// output = softmax(input * scale) over each row.
    ///
    /// mask may be null; a false entry is forced to -INFINITY and contributes 0 to the output. A fully masked row produces zeros.
    inline void SoftmaxForward(const void *input, const bool *mask, void *output, const int rows, const int columns, const float scale,
            const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(softmax_forward(input, mask, output, rows, columns, scale, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
