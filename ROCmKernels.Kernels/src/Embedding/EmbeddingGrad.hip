//
// Created by Dottik on 3/10/2026.
//
#include <ROCmKernels/Kernels/Embedding/EmbeddingGrad.hpp>

#include <hip/hip_runtime.h>
#include <cstdint>

static __device__ __forceinline__ void compensated_add(float value, float& sum, float& correction) {
    value -= correction;
    const float next = sum + value;
    correction = isfinite(next) ? (next - sum) - value : 0.0f;
    sum = next;
}

template<bool VECTORIZED>
static __global__ void densify_rows(const int64_t* keys, const int64_t* order, const float* values,
                                   float* dense, int nnz, int rows, int columns) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    int first = 0, last = 0;
    if (lane == 0 && nnz && row >= keys[0] && row <= keys[nnz - 1]) {
        int low = 0, high = nnz;
        while (low < high) {
            const int middle = low + (high - low) / 2;
            if (keys[middle] < row) low = middle + 1;
            else high = middle;
        }
        first = low;
        if (low < nnz && keys[low] == row) {
            high = nnz;
            while (low < high) {
                const int middle = low + (high - low) / 2;
                if (keys[middle] <= row) low = middle + 1;
                else high = middle;
            }
        }
        last = low;
    }
    first = __shfl(first, 0, 32);
    last = __shfl(last, 0, 32);
    if constexpr (VECTORIZED) {
        for (int column = lane * 4; column < columns; column += 128) {
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 correction = sum;
            for (int position = first; position < last; ++position) {
                const float4 value = reinterpret_cast<const float4*>(values)[(order[position] * columns + column) / 4];
                compensated_add(value.x, sum.x, correction.x);
                compensated_add(value.y, sum.y, correction.y);
                compensated_add(value.z, sum.z, correction.z);
                compensated_add(value.w, sum.w, correction.w);
            }
            reinterpret_cast<float4*>(dense)[(static_cast<int64_t>(row) * columns + column) / 4] = sum;
        }
    } else {
        for (int column = lane; column < columns; column += 32) {
            float sum = 0.0f, correction = 0.0f;
            for (int position = first; position < last; ++position)
                compensated_add(values[order[position] * columns + column], sum, correction);
            dense[static_cast<int64_t>(row) * columns + column] = sum;
        }
    }
}

extern "C" ROCMKERNELS_EXPORT int embedding_grad_fp32(const int64_t* keys, const int64_t* order, const float* values,
                                          float* dense, int nnz, int rows, int columns, hipStream_t stream) {
    if (nnz < 0 || rows < 1 || columns < 1 || !dense || (nnz && (!keys || !order || !values)))
        return hipErrorInvalidValue;
    hipLaunchKernelGGL((densify_rows<false>), dim3((rows + 3) / 4), dim3(128), 0, stream,
                      keys, order, values, dense, nnz, rows, columns);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int embedding_grad_fp32_tuned(const int64_t* keys, const int64_t* order, const float* values,
                                                float* dense, int nnz, int rows, int columns, int vectorized, hipStream_t stream) {
    if (!vectorized || columns % 4 || reinterpret_cast<uintptr_t>(values) % 16 || reinterpret_cast<uintptr_t>(dense) % 16)
        return embedding_grad_fp32(keys, order, values, dense, nnz, rows, columns, stream);
    if (nnz < 0 || rows < 1 || columns < 1 || !dense || (nnz && (!keys || !order || !values)))
        return hipErrorInvalidValue;
    hipLaunchKernelGGL((densify_rows<true>), dim3((rows + 3) / 4), dim3(128), 0, stream,
                      keys, order, values, dense, nnz, rows, columns);
    return hipGetLastError();
}
