#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int add_norm_fp32_forward(const float *x, const float *residual, const float *weight, const float *bias, float *summed,
        float *normalized, float *statistics, int rows, float epsilon, hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int add_norm_fp32_backward(const float *summed, const float *upstream, const float *residual_upstream,
        const float *weight, const float *statistics, float *gradient, float *weight_gradient, float *bias_gradient, int rows,
        hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Residual add and LayerNorm for rows of 384 features.
    ///
    /// bias may be null. statistics receives {mean, reciprocal standard deviation} per row.
    inline void AddNormForward(const float *x, const float *residual, const float *weight, const float *bias, float *summed,
            float *normalized, float *statistics, const int rows, const float epsilon = 1e-5f, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(add_norm_fp32_forward(x, residual, weight, bias, summed, normalized, statistics, rows, epsilon, stream));
    }

    /// gradient, weightGradient and biasGradient are each optional; residual_upstream may be null.
    inline void AddNormBackward(const float *summed, const float *upstream, const float *residual_upstream, const float *weight,
            const float *statistics, float *gradient, float *weight_gradient, float *bias_gradient, const int rows,
            hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(add_norm_fp32_backward(summed, upstream, residual_upstream, weight, statistics, gradient, weight_gradient,
                bias_gradient, rows, stream));
    }
} // namespace ROCmKernels::Kernels
