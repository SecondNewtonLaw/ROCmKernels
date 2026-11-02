// Per-row top-k with a bitonic merge, ported from llama.cpp ggml-cuda top-k.cu and argsort.cu
// (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/TopK.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <math.h>

namespace {
    constexpr int kThreads = 256;
    constexpr int kMaxK = 8;
    constexpr int kCandidates = kThreads * kMaxK;

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

    // Sorts the shared candidate pairs into the order the output expects.
    __device__ __forceinline__ void bitonic_sort_descending(float *values, int *indices) {
        for (int span = 2; span <= kCandidates; span <<= 1) {
            for (int stride = span >> 1; stride > 0; stride >>= 1) {
                for (int i = threadIdx.x; i < kCandidates; i += kThreads) {
                    const int partner = i ^ stride;
                    if (partner < i) continue;

                    const bool lowerHalfFirst = (i & span) == 0;
                    const bool swap = lowerHalfFirst ? candidate_before(values[partner], indices[partner], values[i], indices[i])
                                                     : candidate_before(values[i], indices[i], values[partner], indices[partner]);
                    if (swap) {
                        const float otherValue = values[partner];
                        const int otherIndex = indices[partner];
                        values[partner] = values[i];
                        indices[partner] = indices[i];
                        values[i] = otherValue;
                        indices[i] = otherIndex;
                    }
                }
                __syncthreads();
            }
        }
    }

    template<typename T, int K>
    __global__ void top_k_forward_kernel(const T *scores, float *values, int *indices, const int classes) {
        __shared__ float sharedValues[kCandidates];
        __shared__ int sharedIndices[kCandidates];

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

        // Phase 2: one candidate block per thread, padding the tail with empty slots.
        #pragma unroll
        for (int slot = 0; slot < K; ++slot) {
            sharedValues[threadIdx.x * K + slot] = bestValues[slot];
            sharedIndices[threadIdx.x * K + slot] = bestIndices[slot];
        }
        for (int slot = kThreads * K + threadIdx.x; slot < kCandidates; slot += kThreads) {
            sharedValues[slot] = -INFINITY;
            sharedIndices[slot] = -1;
        }
        __syncthreads();

        // Phase 3: merge the candidates with a block-wide bitonic sort.
        bitonic_sort_descending(sharedValues, sharedIndices);

        // Phase 4: the sorted head is the answer.
        if (static_cast<int>(threadIdx.x) < K) {
            values[row * K + threadIdx.x] = sharedValues[threadIdx.x];
            indices[row * K + threadIdx.x] = sharedIndices[threadIdx.x];
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
