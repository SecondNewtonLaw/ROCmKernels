// Row gather and scatter, ported from llama.cpp ggml-cuda getrows.cu and set-rows.cu
// (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/RowGather.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>

namespace {
    constexpr int kThreads = 256;
    constexpr int kMaxTiles = 65535;

    // One block per row, one thread tile per column chunk.
    template<typename T>
    __global__ void get_rows_kernel(const T *table, const int *indices, T *output, const int columns) {
        using ROCmKernels::Kernels::LoadScalar;
        using ROCmKernels::Kernels::StoreScalar;

        const int row = static_cast<int>(blockIdx.x);
        const int target = indices[row];
        const T *source = table + static_cast<std::size_t>(target) * columns;
        T *destination = output + static_cast<std::size_t>(row) * columns;

        for (int column = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x); column < columns;
                column += static_cast<int>(gridDim.y) * static_cast<int>(blockDim.x)) {
            StoreScalar(destination + column, LoadScalar(source + column));
        }
    }

    template<typename T>
    __global__ void set_rows_kernel(const T *input, const int *indices, T *table, const int columns) {
        using ROCmKernels::Kernels::LoadScalar;
        using ROCmKernels::Kernels::StoreScalar;

        const int row = static_cast<int>(blockIdx.x);
        const int target = indices[row];
        const T *source = input + static_cast<std::size_t>(row) * columns;
        T *destination = table + static_cast<std::size_t>(target) * columns;

        for (int column = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x); column < columns;
                column += static_cast<int>(gridDim.y) * static_cast<int>(blockDim.x)) {
            StoreScalar(destination + column, LoadScalar(source + column));
        }
    }

    __global__ void add_rows_kernel(const float *input, const int *indices, float *table, const int columns) {
        const int row = static_cast<int>(blockIdx.x);
        const int target = indices[row];
        const float *source = input + static_cast<std::size_t>(row) * columns;
        float *destination = table + static_cast<std::size_t>(target) * columns;

        for (int column = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x); column < columns;
                column += static_cast<int>(gridDim.y) * static_cast<int>(blockDim.x)) {
            atomicAdd(destination + column, source[column]);
        }
    }

    int tile_count(const int columns) {
        const int tiles = columns / kThreads + (columns % kThreads != 0 ? 1 : 0);
        return tiles < kMaxTiles ? tiles : kMaxTiles;
    }

    template<typename T>
    int launch_get_rows(const void *table, const int *indices, void *output, const int rows, const int columns, const hipStream_t stream) {
        hipLaunchKernelGGL((get_rows_kernel<T>), dim3(rows, tile_count(columns)), dim3(kThreads), 0, stream, static_cast<const T *>(table),
                indices, static_cast<T *>(output), columns);
        return hipGetLastError();
    }

    template<typename T>
    int launch_set_rows(const void *input, const int *indices, void *table, const int rows, const int columns, const hipStream_t stream) {
        hipLaunchKernelGGL((set_rows_kernel<T>), dim3(rows, tile_count(columns)), dim3(kThreads), 0, stream, static_cast<const T *>(input),
                indices, static_cast<T *>(table), columns);
        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int get_rows_forward(const void *table, const int *indices, void *output, const int rows, const int columns,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;

    if (!table || !indices || !output || rows <= 0 || columns <= 0) return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return launch_get_rows<float>(table, indices, output, rows, columns, stream);
        case DType::Float16:
            return launch_get_rows<__half>(table, indices, output, rows, columns, stream);
    }

    return hipErrorInvalidValue;
}

extern "C" ROCMKERNELS_EXPORT int set_rows_forward(const void *input, const int *indices, void *table, const int rows, const int columns,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;

    if (!input || !indices || !table || rows <= 0 || columns <= 0) return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return launch_set_rows<float>(input, indices, table, rows, columns, stream);
        case DType::Float16:
            return launch_set_rows<__half>(input, indices, table, rows, columns, stream);
    }

    return hipErrorInvalidValue;
}

extern "C" ROCMKERNELS_EXPORT int add_rows_forward(const void *input, const int *indices, void *table, const int rows, const int columns,
        ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;

    if (!input || !indices || !table || rows <= 0 || columns <= 0) return hipErrorInvalidValue;
    if (dtype == DType::Float16) return hipErrorNotSupported;
    if (dtype != DType::Float32) return hipErrorInvalidValue;

    hipLaunchKernelGGL((add_rows_kernel), dim3(rows, tile_count(columns)), dim3(kThreads), 0, stream, static_cast<const float *>(input),
            indices, static_cast<float *>(table), columns);
    return hipGetLastError();
}
