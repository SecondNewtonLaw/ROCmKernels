#include <ROCmKernels/Kernels/Generic/LinearBackward.hpp>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>

namespace {
    constexpr int kTile = 32;

    // destination[c][r] = source[r][c] through a 32x32 shared tile.
    __global__ void transpose_fp16_kernel(const __half *source, __half *destination, const int rows, const int columns) {
        __shared__ __half tile[kTile][kTile + 1];
        const int rowBase = blockIdx.y * kTile;
        const int columnBase = blockIdx.x * kTile;
        const int column = columnBase + threadIdx.x;
        for (int offset = 0; offset < kTile; offset += 8) {
            const int row = rowBase + threadIdx.y + offset;
            tile[threadIdx.y + offset][threadIdx.x] = row < rows && column < columns ? source[static_cast<std::size_t>(row) * columns + column]
                                                                                     : __float2half_rn(0.0f);
        }
        __syncthreads();
        for (int offset = 0; offset < kTile; offset += 8) {
            const int targetRow = columnBase + threadIdx.y + offset;
            const int targetColumn = rowBase + threadIdx.x;
            if (targetRow < columns && targetColumn < rows)
                destination[static_cast<std::size_t>(targetRow) * rows + targetColumn] = tile[threadIdx.x][threadIdx.y + offset];
        }
    }

    // From z this emits both the activation h = silu(z) and dz = dh * silu'(z),
    // because the backward needs h for dW2 and the derivative for dz.
    __global__ void silu_backward_kernel(const __half *dh, const __half *z, __half *dz, __half *h, const std::size_t elements) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= elements) return;
        const float zValue = __half2float(z[index]);
        const float sigmoid = 1.0f / (1.0f + __expf(-zValue));
        const float activation = zValue * sigmoid;
        if (h != nullptr) h[index] = __float2half_rn(activation);
        const float derivative = sigmoid * (1.0f + zValue * (1.0f - sigmoid));
        dz[index] = __float2half_rn(__half2float(dh[index]) * derivative);
    }

    // One thread per column, summing dz over rows: db = sum_rows(dz).
    __global__ void bias_backward_kernel(const __half *dz, float *dbias, const int rows, const int columns) {
        const int column = blockIdx.x * blockDim.x + threadIdx.x;
        if (column >= columns) return;
        float total = 0.0f;
        for (int row = 0; row < rows; ++row) total += __half2float(dz[static_cast<std::size_t>(row) * columns + column]);
        dbias[column] = total;
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int transpose_fp16(const void *source, void *destination, int rows, int columns, hipStream_t stream) {
    if (!source || !destination || rows <= 0 || columns <= 0) return hipErrorInvalidValue;
    const dim3 grid((columns + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    hipLaunchKernelGGL(transpose_fp16_kernel, grid, dim3(kTile, 8), 0, stream, static_cast<const __half *>(source),
                       static_cast<__half *>(destination), rows, columns);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int silu_backward_fp16(const void *dh, const void *z, void *dz, void *h, float *dbias, int rows,
        int columns, hipStream_t stream) {
    if (!dh || !z || !dz || rows <= 0 || columns <= 0) return hipErrorInvalidValue;
    const std::size_t elements = static_cast<std::size_t>(rows) * columns;
    hipLaunchKernelGGL(silu_backward_kernel, dim3((elements + 255) / 256), dim3(256), 0, stream, static_cast<const __half *>(dh),
                       static_cast<const __half *>(z), static_cast<__half *>(dz), static_cast<__half *>(h), elements);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    if (dbias != nullptr) {
        hipLaunchKernelGGL(bias_backward_kernel, dim3((columns + 255) / 256), dim3(256), 0, stream, static_cast<const __half *>(dz), dbias, rows,
                           columns);
    }
    return hipGetLastError();
}


