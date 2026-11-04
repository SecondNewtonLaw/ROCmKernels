//
// Created by Dottik on 2/10/2026.
//
#include <ROCmKernels/Kernels/Loss/CrossEntropy.hpp>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <cstddef>

static constexpr int THREADS = 256;

// Folds value into a running (maximum, sum of exp(x - maximum)) pair.
static __device__ __forceinline__ void online_add(float& maximum, float& total, const float value) {
    if (value > maximum) {
        total = total * expf(maximum - value) + 1.0f;
        maximum = value;
    } else {
        total += expf(value - maximum);
    }
}

static __device__ __forceinline__ void online_merge(float& maximum, float& total, const float otherMaximum, const float otherTotal) {
    const float merged = fmaxf(maximum, otherMaximum);
    total = merged == -INFINITY ? 0.0f : total * expf(maximum - merged) + otherTotal * expf(otherMaximum - merged);
    maximum = merged;
}

// Each wave merges its lanes, then wave 0 merges the per-wave pairs.
static __device__ __forceinline__ void online_reduce_block(float& maximum, float& total, float* partial_max, float* partial_sum) {
    for (int offset = 16; offset; offset >>= 1)
        online_merge(maximum, total, __shfl_down(maximum, offset), __shfl_down(total, offset));
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    if (lane == 0) {
        partial_max[wave] = maximum;
        partial_sum[wave] = total;
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int other = 1; other < THREADS / 32; ++other) online_merge(maximum, total, partial_max[other], partial_sum[other]);
}

// One read of the row: each thread keeps an online max and sum, then the pairs are merged across the block.
static __global__ void loss_forward(const float* logits, const int64_t* targets, float* losses, float* statistics,
                                   int classes, int stride) {
    __shared__ float partial_max[THREADS / 32];
    __shared__ float partial_sum[THREADS / 32];
    const int row = blockIdx.x;
    const float* values = logits + static_cast<size_t>(row) * stride;
    float maximum = -INFINITY;
    float total = 0.0f;
    int scalarBegin = 0;
    if (reinterpret_cast<uintptr_t>(values) % 16 == 0) {
        scalarBegin = classes / 4 * 4;
        #pragma unroll 2
        for (int column = threadIdx.x * 4; column < scalarBegin; column += THREADS * 4) {
            const float4 v = *reinterpret_cast<const float4*>(values + column);
            online_add(maximum, total, v.x);
            online_add(maximum, total, v.y);
            online_add(maximum, total, v.z);
            online_add(maximum, total, v.w);
        }
    }
    for (int column = scalarBegin + threadIdx.x; column < classes; column += THREADS)
        online_add(maximum, total, values[column]);

    online_reduce_block(maximum, total, partial_max, partial_sum);
    if (threadIdx.x == 0) {
        const int64_t target = targets[row];
        assert(target == -100 || (target >= 0 && target < classes));
        const float logarithm = logf(total);
        statistics[row * 2] = maximum;
        statistics[row * 2 + 1] = logarithm;
        losses[row] = target == -100 ? 0.0f : logarithm - (values[target] - maximum);
    }
}

// Same reduction over fp16 storage. The stored logits and the fp16 gradient carry about 2^-11
// relative precision, so the sum uses the hardware exp; the fp32 path keeps the accurate one.
static __device__ __forceinline__ void online_add_half(float& maximum, float& total, const float value) {
    if (value > maximum) {
        total = total * __expf(maximum - value) + 1.0f;
        maximum = value;
    } else {
        total += __expf(value - maximum);
    }
}

static __device__ __forceinline__ void online_merge_half(float& maximum, float& total, const float otherMaximum, const float otherTotal) {
    const float merged = fmaxf(maximum, otherMaximum);
    total = merged == -INFINITY ? 0.0f : total * __expf(maximum - merged) + otherTotal * __expf(otherMaximum - merged);
    maximum = merged;
}

static __device__ __forceinline__ void online_reduce_block_half(float& maximum, float& total, float* partial_max, float* partial_sum) {
    for (int offset = 16; offset; offset >>= 1)
        online_merge_half(maximum, total, __shfl_down(maximum, offset), __shfl_down(total, offset));
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    if (lane == 0) {
        partial_max[wave] = maximum;
        partial_sum[wave] = total;
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int other = 1; other < THREADS / 32; ++other) online_merge_half(maximum, total, partial_max[other], partial_sum[other]);
}

// Same reduction over fp16 storage; every value is widened and accumulated in fp32.
static __global__ void loss_forward_fp16(const __half* logits, const int64_t* targets, float* losses, float* statistics,
                                        int classes, int stride) {
    __shared__ float partial_max[THREADS / 32];
    __shared__ float partial_sum[THREADS / 32];
    const int row = blockIdx.x;
    const __half* values = logits + static_cast<size_t>(row) * stride;
    float maximum = -INFINITY;
    float total = 0.0f;
    int scalarBegin = 0;
    if (reinterpret_cast<uintptr_t>(values) % 16 == 0) {
        scalarBegin = classes / 8 * 8;
        #pragma unroll 2
        for (int column = threadIdx.x * 8; column < scalarBegin; column += THREADS * 8) {
            const uint4 raw = *reinterpret_cast<const uint4*>(values + column);
            const __half* packed = reinterpret_cast<const __half*>(&raw);
            #pragma unroll
            for (int index = 0; index < 8; ++index) online_add_half(maximum, total, __half2float(packed[index]));
        }
    }
    for (int column = scalarBegin + threadIdx.x; column < classes; column += THREADS)
        online_add_half(maximum, total, __half2float(values[column]));

    online_reduce_block_half(maximum, total, partial_max, partial_sum);
    if (threadIdx.x == 0) {
        const int64_t target = targets[row];
        assert(target == -100 || (target >= 0 && target < classes));
        const float logarithm = logf(total);
        statistics[row * 2] = maximum;
        statistics[row * 2 + 1] = logarithm;
        losses[row] = target == -100 ? 0.0f : logarithm - (__half2float(values[target]) - maximum);
    }
}

static __global__ void loss_backward(const float* logits, const int64_t* targets, const float* statistics,
                                    const float* upstream, float* gradient, int classes, int stride, int upstream_stride) {
    const int column = blockIdx.x * THREADS + threadIdx.x;
    const int row = blockIdx.y;
    if (column >= classes) return;
    const int64_t target = targets[row];
    const float maximum = statistics[row * 2];
    const float logarithm = statistics[row * 2 + 1];
    float value;
    if (target == -100) value = isnan(logarithm) ? NAN : 0.0f;
    else {
        const float scale = upstream[row * upstream_stride];
        const float probability = expf((logits[row * stride + column] - maximum) - logarithm);
        value = fmaf(probability, scale, column == target ? -scale : 0.0f);
    }
    gradient[row * classes + column] = value;
}

// gradient[row][column] = p * scale - onehot * scale, read from fp16 logits and written as fp16.
// The softmax term is p / rows in the mean-reduction case, which is below the fp16 subnormal
// range once classes is large; the gradient is then effectively one-hot, the same value the
// graph would get from casting an fp32 gradient back down.
static __global__ void loss_backward_fp16(const __half* logits, const int64_t* targets, const float* statistics,
                                         const float* upstream, __half* gradient, int classes, int stride, int upstream_stride) {
    const int row = blockIdx.y;
    const int base = (blockIdx.x * THREADS + threadIdx.x) * 8;
    if (base >= classes) return;
    const int64_t target = targets[row];
    const __half* values = logits + static_cast<size_t>(row) * stride;
    __half* output = gradient + static_cast<size_t>(row) * classes;
    const float maximum = statistics[row * 2];
    const float logarithm = statistics[row * 2 + 1];
    const bool ignored = target == -100;
    const float scale = ignored ? 0.0f : upstream[row * upstream_stride];
    const bool aligned = ((reinterpret_cast<uintptr_t>(values) | reinterpret_cast<uintptr_t>(output)) & 15) == 0;

    if (aligned && base + 8 <= classes) {
        const uint4 raw = *reinterpret_cast<const uint4*>(values + base);
        const __half* packed = reinterpret_cast<const __half*>(&raw);
        __align__(16) __half result[8];
        #pragma unroll
        for (int index = 0; index < 8; ++index) {
            const int column = base + index;
            result[index] = __float2half_rn(ignored ? (isnan(logarithm) ? NAN : 0.0f)
                    : fmaf(__expf((__half2float(packed[index]) - maximum) - logarithm), scale, column == target ? -scale : 0.0f));
        }
        *reinterpret_cast<uint4*>(output + base) = *reinterpret_cast<const uint4*>(result);
        return;
    }

    const int end = base + 8 < classes ? base + 8 : classes;
    for (int column = base; column < end; ++column) {
        output[column] = __float2half_rn(ignored ? (isnan(logarithm) ? NAN : 0.0f)
                : fmaf(__expf((__half2float(values[column]) - maximum) - logarithm), scale, column == target ? -scale : 0.0f));
    }
}

extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp32_forward(const float* logits, const int64_t* targets,
                                                 float* losses, float* statistics, int rows, int classes,
                                                 int stride, hipStream_t stream) {
    if (!logits || !targets || !losses || !statistics || rows <= 0 || classes <= 0 || stride < 0)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(loss_forward, dim3(rows), dim3(THREADS), 0, stream, logits, targets, losses, statistics,
                      classes, stride);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp32_backward(const float* logits, const int64_t* targets,
                                                  const float* statistics, const float* upstream, float* gradient,
                                                  int rows, int classes, int stride, int upstream_stride, hipStream_t stream) {
    if (!logits || !targets || !statistics || !upstream || !gradient || rows <= 0 || rows > 65535 ||
        classes <= 0 || stride < 0 || upstream_stride < 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(loss_backward, dim3((classes + THREADS - 1) / THREADS, rows), dim3(THREADS), 0,
                      stream, logits, targets, statistics, upstream, gradient, classes, stride, upstream_stride);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp16_forward(const void* logits, const int64_t* targets,
                                                 float* losses, float* statistics, int rows, int classes,
                                                 int stride, hipStream_t stream) {
    if (!logits || !targets || !losses || !statistics || rows <= 0 || classes <= 0 || stride < 0)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(loss_forward_fp16, dim3(rows), dim3(THREADS), 0, stream,
                      static_cast<const __half*>(logits), targets, losses, statistics, classes, stride);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int cross_entropy_fp16_backward(const void* logits, const int64_t* targets,
                                                  const float* statistics, const float* upstream, void* gradient,
                                                  int rows, int classes, int stride, int upstream_stride, hipStream_t stream) {
    if (!logits || !targets || !statistics || !upstream || !gradient || rows <= 0 || rows > 65535 ||
        classes <= 0 || stride < 0 || upstream_stride < 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(loss_backward_fp16, dim3((classes + THREADS * 8 - 1) / (THREADS * 8), rows), dim3(THREADS), 0,
                      stream, static_cast<const __half*>(logits), targets, statistics, upstream,
                      static_cast<__half*>(gradient), classes, stride, upstream_stride);
    return hipGetLastError();
}
