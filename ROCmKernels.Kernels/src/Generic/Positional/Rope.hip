// Rotary position embedding, ported from llama.cpp ggml-cuda rope.cu (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/Rope.hpp>

#include <hip/hip_runtime.h>

#include <cmath>

namespace {
    constexpr int kThreads = 256;

    // One block per token covers every head, so each angle's sincos is computed once instead of once per head.
    template<typename T, bool NeoX>
    __global__ void rope_forward_kernel(const T *input, T *output, const int *positions, const int heads, const int headDim,
            const int rotaryDim, const float thetaScale, const int tokens, const ROCmKernels::Kernels::Strides is,
            const ROCmKernels::Kernels::Strides os) {
        extern __shared__ float2 rotation[];

        const int token = blockIdx.x;
        const int sample = blockIdx.y;
        const int half = rotaryDim / 2;
        const int position = positions[sample * tokens + token];

        for (int index = threadIdx.x; index < half; index += blockDim.x) {
            const float angle = static_cast<float>(position) * powf(thetaScale, static_cast<float>(index));
            float2 value;
            sincosf(angle, &value.y, &value.x);
            rotation[index] = value;
        }
        __syncthreads();

        // Two adjacent angles per thread so every access is a pair; needs even offsets for the pair alignment.
        const bool paired = half % 2 == 0 && (is.batch | is.head | is.token | os.batch | os.head | os.token) % 2 == 0;
        if (paired) {
            const int quarter = half / 2;
            for (int item = threadIdx.x; item < heads * quarter; item += blockDim.x) {
                const int head = item / quarter;
                const int index = 2 * (item - head * quarter);
                const int inputBase = sample * is.batch + head * is.head + token * is.token;
                const int outputBase = sample * os.batch + head * os.head + token * os.token;
                const int first = NeoX ? index : 2 * index;
                const int second = NeoX ? index + half : 2 * index + 2;

                const float2 a = ROCmKernels::Kernels::LoadPair(input + inputBase + first);
                const float2 b = ROCmKernels::Kernels::LoadPair(input + inputBase + second);
                const float2 r0 = rotation[index];
                const float2 r1 = rotation[index + 1];
                // NeoX pairs hold (x0[i], x0[i+1]) and (x1[i], x1[i+1]); normal pairs hold (x0[i], x1[i]) and (x0[i+1], x1[i+1]).
                const float2 x0 = NeoX ? a : make_float2(a.x, b.x);
                const float2 x1 = NeoX ? b : make_float2(a.y, b.y);
                const float2 y0 = make_float2(x0.x * r0.x - x1.x * r0.y, x0.y * r1.x - x1.y * r1.y);
                const float2 y1 = make_float2(x0.x * r0.y + x1.x * r0.x, x0.y * r1.y + x1.y * r1.x);
                ROCmKernels::Kernels::StorePair(output + outputBase + first, NeoX ? y0 : make_float2(y0.x, y1.x));
                ROCmKernels::Kernels::StorePair(output + outputBase + second, NeoX ? y1 : make_float2(y0.y, y1.y));
            }
        }

        for (int item = threadIdx.x; !paired && item < heads * half; item += blockDim.x) {
            const int head = item / half;
            const int index = item - head * half;
            const int inputBase = sample * is.batch + head * is.head + token * is.token;
            const int outputBase = sample * os.batch + head * os.head + token * os.token;
            const int low = NeoX ? index : 2 * index;
            const int high = NeoX ? index + half : 2 * index + 1;

            // Both values are read before either store, so an aliased output stays correct.
            const float x0 = ROCmKernels::Kernels::LoadScalar(input + inputBase + low);
            const float x1 = ROCmKernels::Kernels::LoadScalar(input + inputBase + high);
            const float2 r = rotation[index];

            ROCmKernels::Kernels::StoreScalar(output + outputBase + low, x0 * r.x - x1 * r.y);
            ROCmKernels::Kernels::StoreScalar(output + outputBase + high, x0 * r.y + x1 * r.x);
        }

        const int tail = headDim - rotaryDim;
        for (int item = threadIdx.x; item < heads * tail; item += blockDim.x) {
            const int head = item / tail;
            const int index = rotaryDim + item - head * tail;
            ROCmKernels::Kernels::StoreScalar(output + sample * os.batch + head * os.head + token * os.token + index,
                    ROCmKernels::Kernels::LoadScalar(input + sample * is.batch + head * is.head + token * is.token + index));
        }
    }

    template<typename T>
    int launch(const void *input, void *output, const int *positions, const int batch, const int heads, const int tokens, const int headDim,
            const int rotaryDim, const float theta, const ROCmKernels::Kernels::RopeStyle style, const ROCmKernels::Kernels::Strides is,
            const ROCmKernels::Kernels::Strides os, const hipStream_t stream) {
        // theta^(-2i/rotaryDim) is raised to i in the kernel.
        const float thetaScale = rotaryDim > 0 ? std::pow(theta, -2.0f / static_cast<float>(rotaryDim)) : 1.0f;
        const dim3 grid(tokens, batch);
        const dim3 block(kThreads);
        const std::size_t shared = static_cast<std::size_t>(rotaryDim / 2) * sizeof(float2);

        if (style == ROCmKernels::Kernels::RopeStyle::NeoX) {
            hipLaunchKernelGGL((rope_forward_kernel<T, true>), grid, block, shared, stream, static_cast<const T *>(input),
                    static_cast<T *>(output), positions, heads, headDim, rotaryDim, thetaScale, tokens, is, os);
        } else {
            hipLaunchKernelGGL((rope_forward_kernel<T, false>), grid, block, shared, stream, static_cast<const T *>(input),
                    static_cast<T *>(output), positions, heads, headDim, rotaryDim, thetaScale, tokens, is, os);
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
