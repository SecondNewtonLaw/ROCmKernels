#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Kernels/Strides.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

namespace ROCmKernels::Kernels {
    /// Pairing of the features that take part in the rotation.
    enum class RopeStyle : unsigned char { Normal, NeoX };
} // namespace ROCmKernels::Kernels

extern "C" ROCMKERNELS_EXPORT int rope_forward(const void *input, void *output, const int *positions, int batch, int heads, int tokens,
        int headDim, int rotaryDim, float theta, ROCmKernels::Kernels::RopeStyle style, ROCmKernels::Kernels::Strides is,
        ROCmKernels::Kernels::Strides os, ROCmKernels::Kernels::DType dtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Rotates features [0, rotaryDim) of every [batch][head][token] row in place or out of place.
    ///
    /// positions holds one index per (batch, token). Features past rotaryDim are copied unchanged.
    inline void RopeForward(const void *input, void *output, const int *positions, const int batch, const int heads, const int tokens,
            const int headDim, const int rotaryDim, const float theta, const RopeStyle style, const Strides &is, const Strides &os,
            const DType dtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(rope_forward(input, output, positions, batch, heads, tokens, headDim, rotaryDim, theta, style, is, os, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
