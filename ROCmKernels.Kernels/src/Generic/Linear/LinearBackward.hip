#include <ROCmKernels/Kernels/Generic/LinearBackward.hpp>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {
    constexpr int kTile = 32;
    constexpr int kBiasSplits = 8;

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
        if (h != nullptr) h[index] = __float2half_rn(zValue * sigmoid);
        const float derivative = sigmoid * (1.0f + zValue * (1.0f - sigmoid));
        dz[index] = __float2half_rn(__half2float(dh[index]) * derivative);
    }

    // Same gradients, but sigmoid comes from the forward so the backward has no exp.
    __global__ void silu_backward_s_kernel(const __half *dh, const __half *z, const __half *s, __half *dz, __half *h, const std::size_t elements) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= elements) return;
        const float zValue = __half2float(z[index]);
        const float sigmoid = __half2float(s[index]);
        if (h != nullptr) h[index] = __float2half_rn(zValue * sigmoid);
        dz[index] = __float2half_rn(__half2float(dh[index]) * sigmoid * (1.0f + zValue * (1.0f - sigmoid)));
    }

    // h = silu(z) and s = sigmoid(z) come from the forward. silu'(z) = s + h(1-s),
    // so this needs neither z nor exp and does not rewrite h. Eight elements per
    // thread; the tail up to seven elements is handled by the last block.
    __global__ void silu_backward_hs_kernel(const __half *dh, const __half *h, const __half *s, __half *dz, const std::size_t elements) {
        const std::size_t vectors = elements / 8;
        const std::size_t vectorIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (vectorIndex < vectors) {
            const std::size_t base = vectorIndex * 8;
            const uint4 dhv = *reinterpret_cast<const uint4 *>(dh + base);
            const uint4 hv = *reinterpret_cast<const uint4 *>(h + base);
            const uint4 sv = *reinterpret_cast<const uint4 *>(s + base);
            const __half *dhh = reinterpret_cast<const __half *>(&dhv);
            const __half *hh = reinterpret_cast<const __half *>(&hv);
            const __half *ss = reinterpret_cast<const __half *>(&sv);
            __align__(16) __half out[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float sigmoid = __half2float(ss[i]);
                const float activation = __half2float(hh[i]);
                out[i] = __float2half_rn(__half2float(dhh[i]) * (sigmoid + activation * (1.0f - sigmoid)));
            }
            *reinterpret_cast<uint4 *>(dz + base) = *reinterpret_cast<const uint4 *>(out);
            return;
        }
        const std::size_t tail = vectorIndex - vectors;
        if (tail < elements - vectors * 8) {
            const std::size_t index = vectors * 8 + tail;
            const float sigmoid = __half2float(s[index]);
            const float activation = __half2float(h[index]);
            dz[index] = __float2half_rn(__half2float(dh[index]) * (sigmoid + activation * (1.0f - sigmoid)));
        }
    }

    __global__ void silu_backward_hs_scalar_kernel(const __half *dh, const __half *h, const __half *s, __half *dz, const std::size_t elements) {
        const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= elements) return;
        const float sigmoid = __half2float(s[index]);
        const float activation = __half2float(h[index]);
        dz[index] = __float2half_rn(__half2float(dh[index]) * (sigmoid + activation * (1.0f - sigmoid)));
    }

    // db = sum_rows(dz), split over row chunks. The chunks accumulate into dbias
    // in an arbitrary order, so the sum is reproducible only to fp32 rounding.
    __global__ void bias_backward_kernel(const __half *dz, float *dbias, const int rows, const int columns) {
        const int column = blockIdx.x * blockDim.x + threadIdx.x;
        if (column >= columns) return;
        const int begin = static_cast<int>(static_cast<long long>(rows) * blockIdx.y / kBiasSplits);
        const int end = static_cast<int>(static_cast<long long>(rows) * (blockIdx.y + 1) / kBiasSplits);
        float total = 0.0f;
        for (int row = begin; row < end; ++row) total += __half2float(dz[static_cast<std::size_t>(row) * columns + column]);
        atomicAdd(&dbias[column], total);
    }

    void launch_bias(const __half *dz, float *dbias, const int rows, const int columns, const hipStream_t stream) {
        (void) hipMemsetAsync(dbias, 0, sizeof(float) * columns, stream);
        hipLaunchKernelGGL(bias_backward_kernel, dim3((columns + 255) / 256, kBiasSplits), dim3(256), 0, stream, dz, dbias, rows, columns);
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int transpose_fp16(const void *source, void *destination, int rows, int columns, hipStream_t stream) {
    if (!source || !destination || rows <= 0 || columns <= 0) return hipErrorInvalidValue;
    const dim3 grid((columns + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    hipLaunchKernelGGL(transpose_fp16_kernel, grid, dim3(kTile, 8), 0, stream, static_cast<const __half *>(source),
                       static_cast<__half *>(destination), rows, columns);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int silu_backward_fp16(const void *dh, const void *z, void *dz, void *h, float *dbias, int rows, int columns,
        hipStream_t stream) {
    if (!dh || !z || !dz || rows <= 0 || columns <= 0) return hipErrorInvalidValue;
    const std::size_t elements = static_cast<std::size_t>(rows) * columns;
    hipLaunchKernelGGL(silu_backward_kernel, dim3((elements + 255) / 256), dim3(256), 0, stream, static_cast<const __half *>(dh),
                       static_cast<const __half *>(z), static_cast<__half *>(dz), static_cast<__half *>(h), elements);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    if (dbias != nullptr) launch_bias(static_cast<const __half *>(dz), dbias, rows, columns, stream);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int silu_backward_s_fp16(const void *dh, const void *z, const void *s, void *dz, void *h, float *dbias, int rows,
        int columns, hipStream_t stream) {
    if (!dh || !z || !s || !dz || rows <= 0 || columns <= 0) return hipErrorInvalidValue;
    const std::size_t elements = static_cast<std::size_t>(rows) * columns;
    hipLaunchKernelGGL(silu_backward_s_kernel, dim3((elements + 255) / 256), dim3(256), 0, stream, static_cast<const __half *>(dh),
                       static_cast<const __half *>(z), static_cast<const __half *>(s), static_cast<__half *>(dz), static_cast<__half *>(h),
                       elements);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    if (dbias != nullptr) launch_bias(static_cast<const __half *>(dz), dbias, rows, columns, stream);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int silu_backward_hs_fp16(const void *dh, const void *h, const void *s, void *dz, float *dbias, int rows,
        int columns, hipStream_t stream) {
    if (!dh || !h || !s || !dz || rows <= 0 || columns <= 0) return hipErrorInvalidValue;
    const std::size_t elements = static_cast<std::size_t>(rows) * columns;
    const bool aligned = ((reinterpret_cast<std::uintptr_t>(dh) | reinterpret_cast<std::uintptr_t>(h) |
                           reinterpret_cast<std::uintptr_t>(s) | reinterpret_cast<std::uintptr_t>(dz)) & 15) == 0;
    if (aligned) {
        const std::size_t vectors = elements / 8;
        const std::size_t blocks = (vectors + 255) / 256 + (elements > vectors * 8 ? 1 : 0);
        hipLaunchKernelGGL(silu_backward_hs_kernel, dim3(static_cast<unsigned>(blocks)), dim3(256), 0, stream,
                           static_cast<const __half *>(dh), static_cast<const __half *>(h), static_cast<const __half *>(s),
                           static_cast<__half *>(dz), elements);
    } else {
        hipLaunchKernelGGL(silu_backward_hs_scalar_kernel, dim3((elements + 255) / 256), dim3(256), 0, stream,
                           static_cast<const __half *>(dh), static_cast<const __half *>(h), static_cast<const __half *>(s),
                           static_cast<__half *>(dz), elements);
    }
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    if (dbias != nullptr) launch_bias(static_cast<const __half *>(dz), dbias, rows, columns, stream);
    return hipGetLastError();
}
