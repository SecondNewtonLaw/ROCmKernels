#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int layernorm_scale_add_forward(const void *x, const void *z, const void *scale, const void *weight,
        const void *bias, void *output, float *statistics, int rows, int columns, float epsilon, ROCmKernels::Kernels::DType dtype,
        hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int layernorm_scale_add_backward(const void *x, const void *z, const void *scale, const void *weight,
        const float *statistics, const void *upstream, void *gradientX, void *gradientZ, float *gradientScale, float *gradientWeight,
        float *gradientBias, int rows, int columns, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// y = ((x + scale * z) - mean) * rstd * weight + bias with per-row statistics over s = x + scale * z.
    ///
    /// weight and bias may be null, in which case the affine term is skipped. statistics receives {mean, rstd} per row.
    inline void LayerNormScaleAddForward(const void *x, const void *z, const void *scale, const void *weight, const void *bias, void *output,
            float *statistics, const int rows, const int columns, const float epsilon, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(layernorm_scale_add_forward(x, z, scale, weight, bias, output, statistics, rows, columns, epsilon, dtype, stream));
    }

    /// gradientScale is always written; gradientWeight is skipped when weight is null.
    /// statistics is the {mean, rstd} pair produced by the forward pass.
    inline void LayerNormScaleAddBackward(const void *x, const void *z, const void *scale, const void *weight, const float *statistics,
            const void *upstream, void *gradientX, void *gradientZ, float *gradientScale, float *gradientWeight, float *gradientBias,
            const int rows, const int columns, const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(layernorm_scale_add_backward(x, z, scale, weight, statistics, upstream, gradientX, gradientZ, gradientScale,
                gradientWeight, gradientBias, rows, columns, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
