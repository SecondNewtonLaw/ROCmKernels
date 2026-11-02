//
// Created by Dottik on 2/10/2026.
//
#include <ROCmKernels/Kernels/Loss/CrossEntropy.hpp>

#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>

static constexpr int THREADS = 256;

template<bool MAXIMUM>
static __device__ float wave_reduce(float value) {
    for (int offset = 16; offset; offset >>= 1) {
        const float other = __shfl_down(value, offset);
        if constexpr (MAXIMUM) value = fmaxf(value, other);
        else value += other;
    }
    return value;
}

template<bool MAXIMUM>
static __device__ float block_reduce(float value, float* partial) {
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    value = wave_reduce<MAXIMUM>(value);
    if (lane == 0) partial[wave] = value;
    __syncthreads();
    if (wave == 0) {
        value = lane < THREADS / 32 ? partial[lane] : MAXIMUM ? -INFINITY : 0.0f;
        value = wave_reduce<MAXIMUM>(value);
        if (lane == 0) partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

static __global__ void loss_forward(const float* logits, const int64_t* targets, float* losses, float* statistics,
                                   int classes, int stride) {
    __shared__ float partial_max[THREADS / 32];
    __shared__ float partial_sum[THREADS / 32];
    const int row = blockIdx.x;
    const float* values = logits + row * stride;
    float maximum = -INFINITY;
    #pragma unroll 4
    for (int column = threadIdx.x; column < classes; column += THREADS)
        maximum = fmaxf(maximum, values[column]);
    maximum = block_reduce<true>(maximum, partial_max);
    float total = 0.0f;
    #pragma unroll 4
    for (int column = threadIdx.x; column < classes; column += THREADS)
        total += expf(values[column] - maximum);
    total = block_reduce<false>(total, partial_sum);
    if (threadIdx.x == 0) {
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
