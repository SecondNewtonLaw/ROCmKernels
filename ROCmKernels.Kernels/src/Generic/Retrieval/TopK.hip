// Per-row top-k with a wave-shuffle merge, originally ported from llama.cpp ggml-cuda top-k.cu and argsort.cu
// (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/TopK.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <math.h>

namespace {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    constexpr int kMaxK = 8;

    // True when a must precede b in the descending output; equal values order by class index.
    __device__ __forceinline__ bool candidate_before(const float valueA, const int indexA, const float valueB, const int indexB) {
        return valueA > valueB || (valueA == valueB && indexA < indexB);
    }

    // Drops the previous last entry and bubbles the candidate up a sorted descending list.
    template<int K>
    __device__ __forceinline__ void insert_candidate(float *values, int *indices, const float value, const int index) {
        if (!(value > values[K - 1])) return;

        values[K - 1] = value;
        indices[K - 1] = index;

        #pragma unroll
        for (int slot = K - 1; slot > 0; --slot) {
            if (values[slot] > values[slot - 1]) {
                const float otherValue = values[slot - 1];
                const int otherIndex = indices[slot - 1];
                values[slot - 1] = values[slot];
                indices[slot - 1] = indices[slot];
                values[slot] = otherValue;
                indices[slot] = otherIndex;
            }
        }
    }

    // Merges the sorted per-lane lists of a wave: K rounds pick the best head across lanes and pop it from its owner.
    // Every lane receives the merged list.
    template<int K>
    __device__ __forceinline__ void wave_top_k(float *values, int *indices, float *mergedValues, int *mergedIndices) {
        #pragma unroll
        for (int round = 0; round < K; ++round) {
            float bestValue = values[0];
            int bestIndex = indices[0];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float otherValue = __shfl_xor(bestValue, offset, 32);
                const int otherIndex = __shfl_xor(bestIndex, offset, 32);
                if (candidate_before(otherValue, otherIndex, bestValue, bestIndex)) {
                    bestValue = otherValue;
                    bestIndex = otherIndex;
                }
            }
            mergedValues[round] = bestValue;
            mergedIndices[round] = bestIndex;

            if (indices[0] == bestIndex && values[0] == bestValue) {
                #pragma unroll
                for (int slot = 0; slot < K - 1; ++slot) {
                    values[slot] = values[slot + 1];
                    indices[slot] = indices[slot + 1];
                }
                values[K - 1] = -INFINITY;
                indices[K - 1] = -1;
            }
        }
    }

    template<typename T, int K>
    __global__ void top_k_forward_kernel(const T *scores, float *values, int *indices, const int classes) {
        __shared__ float sharedValues[kWarps][K];
        __shared__ int sharedIndices[kWarps][K];

        const std::size_t row = blockIdx.x;
        const T *rowScores = scores + row * classes;

        // Phase 1: each thread keeps a sorted top-k of its strided classes.
        float bestValues[K];
        int bestIndices[K];
        #pragma unroll
        for (int slot = 0; slot < K; ++slot) {
            bestValues[slot] = -INFINITY;
            bestIndices[slot] = -1;
        }

        for (int column = threadIdx.x; column < classes; column += kThreads) {
            insert_candidate<K>(bestValues, bestIndices, ROCmKernels::Kernels::LoadScalar(rowScores + column), column);
        }

        // Phase 2: merge each wave's lists, then wave 0 merges the per-wave results.
        const int lane = threadIdx.x % 32;
        const int wave = threadIdx.x / 32;
        float mergedValues[K];
        int mergedIndices[K];
        wave_top_k<K>(bestValues, bestIndices, mergedValues, mergedIndices);
        if (lane == 0) {
            #pragma unroll
            for (int slot = 0; slot < K; ++slot) {
                sharedValues[wave][slot] = mergedValues[slot];
                sharedIndices[wave][slot] = mergedIndices[slot];
            }
        }
        __syncthreads();
        if (wave != 0) return;

        #pragma unroll
        for (int slot = 0; slot < K; ++slot) {
            bestValues[slot] = lane < kWarps ? sharedValues[lane][slot] : -INFINITY;
            bestIndices[slot] = lane < kWarps ? sharedIndices[lane][slot] : -1;
        }
        wave_top_k<K>(bestValues, bestIndices, mergedValues, mergedIndices);
        #pragma unroll
        for (int slot = 0; slot < K; ++slot) {
            if (slot == lane) {
                values[row * K + slot] = mergedValues[slot];
                indices[row * K + slot] = mergedIndices[slot];
            }
        }
    }

    template<typename T, int K>
    int launch_forward(const void *scores, float *values, int *indices, const int rows, const int classes, hipStream_t stream) {
        hipLaunchKernelGGL((top_k_forward_kernel<T, K>), dim3(rows), dim3(kThreads), 0, stream, static_cast<const T *>(scores), values,
                indices, classes);
        return hipGetLastError();
    }

    template<typename T>
    int launch_forward(const void *scores, float *values, int *indices, const int rows, const int classes, const int k, hipStream_t stream) {
        switch (k) {
            case 1: return launch_forward<T, 1>(scores, values, indices, rows, classes, stream);
            case 2: return launch_forward<T, 2>(scores, values, indices, rows, classes, stream);
            case 3: return launch_forward<T, 3>(scores, values, indices, rows, classes, stream);
            case 4: return launch_forward<T, 4>(scores, values, indices, rows, classes, stream);
            case 5: return launch_forward<T, 5>(scores, values, indices, rows, classes, stream);
            case 6: return launch_forward<T, 6>(scores, values, indices, rows, classes, stream);
            case 7: return launch_forward<T, 7>(scores, values, indices, rows, classes, stream);
            case 8: return launch_forward<T, 8>(scores, values, indices, rows, classes, stream);
        }

        return hipErrorInvalidValue;
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int top_k_forward(const void *scores, float *values, int *indices, const int rows, const int classes, const int k,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    if (!scores || !values || !indices || rows <= 0 || classes <= 0 || k < 1 || k > kMaxK || k > classes) return hipErrorInvalidValue;

    switch (dtype) {
        case ROCmKernels::Kernels::DType::Float32:
            return launch_forward<float>(scores, values, indices, rows, classes, k, stream);
        case ROCmKernels::Kernels::DType::Float16:
            return launch_forward<__half>(scores, values, indices, rows, classes, k, stream);
    }

    return hipErrorInvalidValue;
}
