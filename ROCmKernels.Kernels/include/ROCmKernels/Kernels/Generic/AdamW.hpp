#pragma once

#include <ROCmKernels/Kernels/DType.hpp>
#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ROCmKernels::Kernels {
    /// One parameter tensor in a fused AdamW update. Both moments are fp32 regardless of the storage dtype.
    struct AdamWTensor {
        void *parameter;
        const void *gradient;
        float *firstMoment;
        float *secondMoment;
        std::int64_t elements;
    };
} // namespace ROCmKernels::Kernels

extern "C" ROCMKERNELS_EXPORT int adamw_multi(const ROCmKernels::Kernels::AdamWTensor *tensors, int tensorCount, long long step,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, ROCmKernels::Kernels::DType dtype,
        hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Fused decoupled-weight-decay AdamW over tensorCount parameter tensors.
    ///
    /// All tensors are updated by one launch; tensors with elements <= 0 are skipped. step is 1-based.
    inline void AdamWMulti(const AdamWTensor *tensors, const int tensorCount, const std::int64_t step, const float learningRate,
            const float beta1, const float beta2, const float epsilon, const float weightDecay, const DType dtype,
            hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(adamw_multi(tensors, tensorCount, step, learningRate, beta1, beta2, epsilon, weightDecay, dtype, stream));
    }
} // namespace ROCmKernels::Kernels
