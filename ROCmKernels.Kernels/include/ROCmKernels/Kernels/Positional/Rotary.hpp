#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>
#include <ROCmKernels/Kernels/Strides.hpp>

#include <hip/hip_runtime.h>

extern "C" ROCMKERNELS_EXPORT int rotary_fp32(const float *q, const float *k, const float *cos, const float *sin, float *oq, float *ok, int batch,
        int heads, int length, int backward, ROCmKernels::Kernels::Strides qs, ROCmKernels::Kernels::Strides ks,
        ROCmKernels::Kernels::Strides cs, ROCmKernels::Kernels::Strides ss, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Rotary embedding for 32-wide heads. Outputs are token-major.
    ///
    /// Input strides are honoured for q, k, cos and sin. cos/sin hold 32 values per token.
    inline void Rotary(const float *q, const float *k, const float *cos, const float *sin, float *oq, float *ok, const int batch,
            const int heads, const int length, const bool backward, const Strides &qs, const Strides &ks, const Strides &cs,
            const Strides &ss, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(rotary_fp32(q, k, cos, sin, oq, ok, batch, heads, length, backward ? 1 : 0, qs, ks, cs, ss, stream));
    }
} // namespace ROCmKernels::Kernels
