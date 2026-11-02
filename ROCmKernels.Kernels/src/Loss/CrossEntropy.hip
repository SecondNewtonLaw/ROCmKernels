//
// Created by Dottik on 2/10/2026.
//
#include <ROCmKernels/Kernels/Loss/CrossEntropy.hpp>

#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>

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

    for (int offset = 16; offset; offset >>= 1)
        online_merge(maximum, total, __shfl_down(maximum, offset), __shfl_down(total, offset));
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    if (lane == 0) {
        partial_max[wave] = maximum;
        partial_sum[wave] = total;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int other = 1; other < THREADS / 32; ++other) online_merge(maximum, total, partial_max[other], partial_sum[other]);
        const int64_t target = targets[row];
        assert(target == -100 || (target >= 0 && target < classes));
        const float logarithm = logf(total);
        statistics[row * 2] = maximum;
        statistics[row * 2 + 1] = logarithm;
        losses[row] = target == -100 ? 0.0f : logarithm - (values[target] - maximum);
    }
}

static __global__ void loss_backward(const float* logits, const int64_t* targets, const float* statistics,
                                    const float* upstream, float* gradient, int classes, int stride, int grad_stride) {
    const int column = blockIdx.x * THREADS + threadIdx.x;
    const int row = blockIdx.y;
    if (column >= classes) return;
    const int64_t target = targets[row];
    const float maximum = statistics[row * 2];
    const float logarithm = statistics[row * 2 + 1];
    float value;
    if (target == -100) value = isnan(logarithm) ? NAN : 0.0f;
    else {
        const float scale = upstream[row * grad_stride];
        const float probability = expf((logits[row * stride + column] - maximum) - logarithm);
        value = fmaf(probability, scale, column == target ? -scale : 0.0f);
    }
    gradient[row * classes + column] = value;
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
                                                  int rows, int classes, int stride, int grad_stride, hipStream_t stream) {
    if (!logits || !targets || !statistics || !upstream || !gradient || rows <= 0 || rows > 65535 ||
        classes <= 0 || stride < 0 || grad_stride < 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(loss_backward, dim3((classes + THREADS - 1) / THREADS, rows), dim3(THREADS), 0,
                      stream, logits, targets, statistics, upstream, gradient, classes, stride, grad_stride);
    return hipGetLastError();
}
