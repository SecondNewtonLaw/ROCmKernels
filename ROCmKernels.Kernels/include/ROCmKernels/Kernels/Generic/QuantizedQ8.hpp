#pragma once

#include <ROCmKernels/Common/Error.hpp>
#include <ROCmKernels/Common/Export.hpp>
#include <ROCmKernels/Kernels/DType.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ROCmKernels::Kernels {
    /// llama.cpp Q8_0 block: one fp16 scale and 32 signed quants.
    static constexpr int kQ8Block = 32;

    struct BlockQ8_0 {
        __half scale;
        std::int8_t values[kQ8Block];
    };

    static_assert(sizeof(BlockQ8_0) == 34);
} // namespace ROCmKernels::Kernels

extern "C" ROCMKERNELS_EXPORT int quantize_q8_0(const float *input, ROCmKernels::Kernels::BlockQ8_0 *output, long long elements,
        hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int dequantize_q8_0(const ROCmKernels::Kernels::BlockQ8_0 *input, float *output, long long elements,
        hipStream_t stream);
extern "C" ROCMKERNELS_EXPORT int quantized_gemv_q8_0(const ROCmKernels::Kernels::BlockQ8_0 *input,
        const ROCmKernels::Kernels::BlockQ8_0 *weights, const void *bias, void *output, int columns, int inner,
        ROCmKernels::Kernels::DType outputDtype, hipStream_t stream);

namespace ROCmKernels::Kernels {
    /// Quantizes elements, a multiple of kQ8Block, using scale = max(abs(x)) / 127 per block.
    inline void QuantizeQ8_0(const float *input, BlockQ8_0 *output, const long long elements, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(quantize_q8_0(input, output, elements, stream));
    }

    /// Expands Q8_0 blocks back to fp32.
    inline void DequantizeQ8_0(const BlockQ8_0 *input, float *output, const long long elements, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(dequantize_q8_0(input, output, elements, stream));
    }

    /// output[c] = sum_k input[k] * weights[c][k] + bias[c] over dequantized values; bias may be null.
    inline void QuantizedGEMVQ8_0(const BlockQ8_0 *input, const BlockQ8_0 *weights, const void *bias, void *output, const int columns,
            const int inner, const DType outputDtype, hipStream_t stream = nullptr) {
        ROCMKERNELS_HIP_CHECK(quantized_gemv_q8_0(input, weights, bias, output, columns, inner, outputDtype, stream));
    }
} // namespace ROCmKernels::Kernels
