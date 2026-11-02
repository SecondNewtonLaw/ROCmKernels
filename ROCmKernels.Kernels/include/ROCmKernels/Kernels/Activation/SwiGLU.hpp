#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int swiglu_fp32_forward(const float *input, float *output, int count, int width, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int swiglu_fp32_backward(const float *input, const float *upstream, float *gradient, int count, int width, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// output[r * width + c] = silu(input[r * 2 * width + c]) * input[r * 2 * width + width + c].
    inline void SwiGLUForward(const float *input, float *output, const int count, const int width, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(swiglu_fp32_forward(input, output, count, width, stream));
    }

    /// Writes both input halves, laid out like the forward input.
    inline void SwiGLUBackward(const float *input, const float *upstream, float *gradient, const int count, const int width,
            hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(swiglu_fp32_backward(input, upstream, gradient, count, width, stream));
    }
} // namespace ROCmKernels::Kernels
