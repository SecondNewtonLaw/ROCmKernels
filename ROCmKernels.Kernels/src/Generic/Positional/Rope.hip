// Rotary position embedding, ported from llama.cpp ggml-cuda rope.cu (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/Rope.hpp>

#include <hip/hip_runtime.h>

#include <cmath>

namespace {
    constexpr int kThreads = 128;

    template<typename T, bool NeoX>
    __global__ void rope_forward_kernel(const T *input, T *output, const int *positions, const int headDim, const int rotaryDim,
            const float thetaScale, const int tokens, const ROCmKernels::Kernels::Strides is, const ROCmKernels::Kernels::Strides os) {
        const int token = blockIdx.x;
        const int head = blockIdx.y;
        const int sample = blockIdx.z;
        const int inputBase = sample * is.batch + head * is.head + token * is.token;
        const int outputBase = sample * os.batch + head * os.head + token * os.token;
        const int half = rotaryDim / 2;
        const int position = positions[sample * tokens + token];

        for (int index = threadIdx.x; index < half; index += blockDim.x) {
            const int low = NeoX ? index : 2 * index;
            const int high = NeoX ? index + half : 2 * index + 1;

            // Both values are read before either store, so an aliased output stays correct.
            const float x0 = ROCmKernels::Kernels::LoadScalar(input + inputBase + low);
            const float x1 = ROCmKernels::Kernels::LoadScalar(input + inputBase + high);

            const float angle = static_cast<float>(position) * powf(thetaScale, static_cast<float>(index));
            float sine;
            float cosine;
            sincosf(angle, &sine, &cosine);

            ROCmKernels::Kernels::StoreScalar(output + outputBase + low, x0 * cosine - x1 * sine);
            ROCmKernels::Kernels::StoreScalar(output + outputBase + high, x0 * sine + x1 * cosine);
        }

        for (int index = rotaryDim + threadIdx.x; index < headDim; index += blockDim.x) {
            ROCmKernels::Kernels::StoreScalar(output + outputBase + index, ROCmKernels::Kernels::LoadScalar(input + inputBase + index));
        }
    }

    template<typename T>
    int launch(const void *input, void *output, const int *positions, const int batch, const int heads, const int tokens, const int headDim,
            const int rotaryDim, const float theta, const ROCmKernels::Kernels::RopeStyle style, const ROCmKernels::Kernels::Strides is,
            const ROCmKernels::Kernels::Strides os, const hipStream_t stream) {
        // theta^(-2i/rotaryDim) is raised to i in the kernel.
        const float thetaScale = rotaryDim > 0 ? std::pow(theta, -2.0f / static_cast<float>(rotaryDim)) : 1.0f;
        const dim3 grid(tokens, heads, batch);
        const dim3 block(kThreads);

        if (style == ROCmKernels::Kernels::RopeStyle::NeoX) {
            hipLaunchKernelGGL((rope_forward_kernel<T, true>), grid, block, 0, stream, static_cast<const T *>(input), static_cast<T *>(output),
                    positions, headDim, rotaryDim, thetaScale, tokens, is, os);
        } else {
            hipLaunchKernelGGL((rope_forward_kernel<T, false>), grid, block, 0, stream, static_cast<const T *>(input), static_cast<T *>(output),
                    positions, headDim, rotaryDim, thetaScale, tokens, is, os);
        }

        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int rope_forward(const void *input, void *output, const int *positions, int batch, int heads, int tokens,
        int headDim, int rotaryDim, float theta, ROCmKernels::Kernels::RopeStyle style, ROCmKernels::Kernels::Strides is,
        ROCmKernels::Kernels::Strides os, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;
    using ROCmKernels::Kernels::RopeStyle;

    if (!input || !output || !positions || batch <= 0 || heads <= 0 || tokens <= 0 || headDim <= 0 || rotaryDim < 0 || rotaryDim % 2 != 0 ||
        rotaryDim > headDim || !(theta > 0.0f) || (style != RopeStyle::Normal && style != RopeStyle::NeoX))
        return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return launch<float>(input, output, positions, batch, heads, tokens, headDim, rotaryDim, theta, style, is, os, stream);
        case DType::Float16:
            return launch<__half>(input, output, positions, batch, heads, tokens, headDim, rotaryDim, theta, style, is, os, stream);
    }

    return hipErrorInvalidValue;
}
