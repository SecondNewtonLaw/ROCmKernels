// Broadcast binary and strided 2D copy, ported from llama.cpp ggml-cuda binbcast.cu and cpy.cu (MIT, Copyright (c) 2023-2026 The ggml authors).
#include <ROCmKernels/Kernels/Generic/BroadcastCopy.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {
    constexpr int kThreads = 256;
    constexpr long long kMaxBlocks = 65535;

    bool RangesOverlap(const void *left, const std::size_t leftBytes, const void *right, const std::size_t rightBytes) {
        const std::uintptr_t leftBegin = reinterpret_cast<std::uintptr_t>(left);
        const std::uintptr_t rightBegin = reinterpret_cast<std::uintptr_t>(right);
        return leftBegin < rightBegin + rightBytes && rightBegin < leftBegin + leftBytes;
    }

    template<ROCmKernels::Kernels::BinaryOp Op>
    __device__ __forceinline__ float Apply(const float a, const float b) {
        using ROCmKernels::Kernels::BinaryOp;

        if constexpr (Op == BinaryOp::Add) return a + b;
        else if constexpr (Op == BinaryOp::Sub) return a - b;
        else if constexpr (Op == BinaryOp::Mul) return a * b;
        else return a / b;
    }

    template<typename T, ROCmKernels::Kernels::BinaryOp Op, ROCmKernels::Kernels::BroadcastMode Mode>
    __global__ void broadcast_binary_kernel(const T *a, const T *b, T *out, const int rows, const int columns) {
        using ROCmKernels::Kernels::BroadcastMode;

        const int row = blockIdx.x;
        if (row >= rows) return;

        const long long rowBase = static_cast<long long>(row) * columns;
        const T *aRow = a + rowBase;
        T *outRow = out + rowBase;

        for (long long column = threadIdx.x; column < columns; column += blockDim.x) {
            long long bIndex = 0;
            if constexpr (Mode == BroadcastMode::None) bIndex = rowBase + column;
            else if constexpr (Mode == BroadcastMode::Row) bIndex = column;
            else if constexpr (Mode == BroadcastMode::Column) bIndex = row;

            const float result = Apply<Op>(ROCmKernels::Kernels::LoadScalar(aRow + column), ROCmKernels::Kernels::LoadScalar(b + bIndex));
            ROCmKernels::Kernels::StoreScalar(outRow + column, result);
        }
    }

    template<typename T>
    __global__ void copy_2d_kernel(const T *src, T *dst, const int rows, const int columns, const int sourceRowStride,
            const int destinationRowStride) {
        const int row = blockIdx.x;
        if (row >= rows) return;

        const T *sourceRow = src + static_cast<long long>(row) * sourceRowStride;
        T *destinationRow = dst + static_cast<long long>(row) * destinationRowStride;

        for (long long column = threadIdx.x; column < columns; column += blockDim.x) destinationRow[column] = sourceRow[column];
    }

    template<typename T>
    __global__ void fill_kernel(T *dst, const float value, const long long elements) {
        const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
        for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; index < elements; index += stride) {
            ROCmKernels::Kernels::StoreScalar(dst + index, value);
        }
    }

    template<typename T, ROCmKernels::Kernels::BinaryOp Op>
    int launch_broadcast_mode(const T *a, const T *b, T *out, const int rows, const int columns,
            const ROCmKernels::Kernels::BroadcastMode mode, const hipStream_t stream) {
        using ROCmKernels::Kernels::BroadcastMode;

        const dim3 grid(static_cast<unsigned int>(rows));
        const dim3 block(static_cast<unsigned int>(columns < kThreads ? columns : kThreads));

        switch (mode) {
            case BroadcastMode::None:
                hipLaunchKernelGGL((broadcast_binary_kernel<T, Op, BroadcastMode::None>), grid, block, 0, stream, a, b, out, rows, columns);
                break;
            case BroadcastMode::Row:
                hipLaunchKernelGGL((broadcast_binary_kernel<T, Op, BroadcastMode::Row>), grid, block, 0, stream, a, b, out, rows, columns);
                break;
            case BroadcastMode::Column:
                hipLaunchKernelGGL((broadcast_binary_kernel<T, Op, BroadcastMode::Column>), grid, block, 0, stream, a, b, out, rows, columns);
                break;
            case BroadcastMode::Scalar:
                hipLaunchKernelGGL((broadcast_binary_kernel<T, Op, BroadcastMode::Scalar>), grid, block, 0, stream, a, b, out, rows, columns);
                break;
        }

        return hipGetLastError();
    }

    template<typename T>
    int launch_broadcast(const T *a, const T *b, T *out, const int rows, const int columns, const ROCmKernels::Kernels::BinaryOp op,
            const ROCmKernels::Kernels::BroadcastMode mode, const hipStream_t stream) {
        using ROCmKernels::Kernels::BinaryOp;

        switch (op) {
            case BinaryOp::Add:
                return launch_broadcast_mode<T, BinaryOp::Add>(a, b, out, rows, columns, mode, stream);
            case BinaryOp::Sub:
                return launch_broadcast_mode<T, BinaryOp::Sub>(a, b, out, rows, columns, mode, stream);
            case BinaryOp::Mul:
                return launch_broadcast_mode<T, BinaryOp::Mul>(a, b, out, rows, columns, mode, stream);
            case BinaryOp::Div:
                return launch_broadcast_mode<T, BinaryOp::Div>(a, b, out, rows, columns, mode, stream);
        }

        return hipErrorInvalidValue;
    }

    template<typename T>
    int launch_copy_2d(const T *src, T *dst, const int rows, const int columns, const int sourceRowStride, const int destinationRowStride,
            const hipStream_t stream) {
        const dim3 grid(static_cast<unsigned int>(rows));
        const dim3 block(static_cast<unsigned int>(columns < kThreads ? columns : kThreads));

        hipLaunchKernelGGL((copy_2d_kernel<T>), grid, block, 0, stream, src, dst, rows, columns, sourceRowStride, destinationRowStride);
        return hipGetLastError();
    }

    template<typename T> int launch_fill(T *dst, const float value, const long long elements, const hipStream_t stream) {
        const long long requestedBlocks = (elements + kThreads - 1) / kThreads;
        const unsigned int blocks = static_cast<unsigned int>(requestedBlocks < kMaxBlocks ? requestedBlocks : kMaxBlocks);

        hipLaunchKernelGGL((fill_kernel<T>), dim3(blocks), dim3(kThreads), 0, stream, dst, value, elements);
        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int broadcast_binary_forward(const void *a, const void *b, void *out, int rows, int columns,
        ROCmKernels::Kernels::BinaryOp op, ROCmKernels::Kernels::BroadcastMode mode, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::BinaryOp;
    using ROCmKernels::Kernels::BroadcastMode;
    using ROCmKernels::Kernels::DType;

    if (!a || !b || !out || rows <= 0 || columns <= 0 || (op != BinaryOp::Add && op != BinaryOp::Sub && op != BinaryOp::Mul && op != BinaryOp::Div) ||
        (mode != BroadcastMode::None && mode != BroadcastMode::Row && mode != BroadcastMode::Column && mode != BroadcastMode::Scalar) ||
        (dtype != DType::Float32 && dtype != DType::Float16))
        return hipErrorInvalidValue;

    const long long outputElements = static_cast<long long>(rows) * columns;
    long long operandElements = 1;
    if (mode == BroadcastMode::None) operandElements = outputElements;
    else if (mode == BroadcastMode::Row) operandElements = columns;
    else if (mode == BroadcastMode::Column) operandElements = rows;
    const std::size_t elementSize = dtype == DType::Float16 ? sizeof(__half) : sizeof(float);
    const std::size_t outputBytes = static_cast<std::size_t>(outputElements) * elementSize;

    // Only exact in-place aliasing is safe; partial overlap would let one row clobber another row's input.
    if (a != out && RangesOverlap(a, outputBytes, out, outputBytes)) return hipErrorInvalidValue;

    // b is reused across rows, so an overlapping output would make the result depend on block ordering.
    if (RangesOverlap(b, static_cast<std::size_t>(operandElements) * elementSize, out, outputBytes)) return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return launch_broadcast<float>(static_cast<const float *>(a), static_cast<const float *>(b), static_cast<float *>(out), rows, columns,
                    op, mode, stream);
        case DType::Float16:
            return launch_broadcast<__half>(static_cast<const __half *>(a), static_cast<const __half *>(b), static_cast<__half *>(out), rows,
                    columns, op, mode, stream);
    }

    return hipErrorInvalidValue;
}

extern "C" ROCMKERNELS_EXPORT int copy_2d_forward(const void *src, void *dst, int rows, int columns, int sourceRowStride,
        int destinationRowStride, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;

    if (!src || !dst || rows <= 0 || columns <= 0 || sourceRowStride < columns || destinationRowStride < columns) return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return launch_copy_2d<float>(static_cast<const float *>(src), static_cast<float *>(dst), rows, columns, sourceRowStride,
                    destinationRowStride, stream);
        case DType::Float16:
            return launch_copy_2d<__half>(static_cast<const __half *>(src), static_cast<__half *>(dst), rows, columns, sourceRowStride,
                    destinationRowStride, stream);
    }

    return hipErrorInvalidValue;
}

extern "C" ROCMKERNELS_EXPORT int fill_forward(void *dst, float value, long long elements, ROCmKernels::Kernels::DType dtype, hipStream_t stream) {
    using ROCmKernels::Kernels::DType;

    if (elements < 0 || (dtype != DType::Float32 && dtype != DType::Float16)) return hipErrorInvalidValue;
    if (elements == 0) return hipSuccess;
    if (!dst) return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return launch_fill<float>(static_cast<float *>(dst), value, elements, stream);
        case DType::Float16:
            return launch_fill<__half>(static_cast<__half *>(dst), value, elements, stream);
    }

    return hipErrorInvalidValue;
}
