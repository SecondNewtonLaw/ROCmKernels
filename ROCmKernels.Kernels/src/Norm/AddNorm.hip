//
// Created by Dottik on 2/10/2026.
//
#include <ROCmKernels/Kernels/Norm/AddNorm.hpp>

#include <hip/hip_runtime.h>
#include <math.h>

static constexpr int WIDTH = 384;
static constexpr int THREADS = 128;

static __device__ float reduce_sum(float value, float* partial) {
    for (int offset = 16; offset; offset >>= 1) value += __shfl_down(value, offset);
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    if (lane == 0) partial[wave] = value;
    __syncthreads();
    if (wave == 0) {
        value = lane < THREADS / 32 ? partial[lane] : 0.0f;
        for (int offset = 16; offset; offset >>= 1) value += __shfl_down(value, offset);
        if (lane == 0) partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}

static __global__ void add_norm_forward(const float* x, const float* residual, const float* weight,
                                        const float* bias, float* summed, float* normalized,
                                        float* statistics, float epsilon) {
    __shared__ float mean_partial[4], variance_partial[4];
    const int row = blockIdx.x;
    float values[3], total = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int index = row * WIDTH + threadIdx.x + i * THREADS;
        values[i] = x[index] + residual[index];
        total += values[i];
    }
    const float mean = reduce_sum(total, mean_partial) / WIDTH;
    float variance = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float centered = values[i] - mean;
        variance += centered * centered;
    }
    const float reciprocal = rsqrtf(reduce_sum(variance, variance_partial) / WIDTH + epsilon);
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int column = threadIdx.x + i * THREADS;
        summed[row * WIDTH + column] = values[i];
        normalized[row * WIDTH + column] = ((values[i] - mean) * reciprocal) * weight[column] +
                                           (bias ? bias[column] : 0.0f);
    }
    if (threadIdx.x == 0) {
        statistics[row * 2] = mean;
        statistics[row * 2 + 1] = reciprocal;
    }
}

static __global__ void add_norm_input_backward(const float* summed, const float* upstream,
                                               const float* residual_upstream, const float* weight,
                                               const float* statistics, float* gradient) {
    __shared__ float plain_partial[4], centered_partial[4];
    const int row = blockIdx.x;
    const float mean = statistics[row * 2];
    const float reciprocal = statistics[row * 2 + 1];
    float values[3], centered[3], plain_sum = 0.0f, centered_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int column = threadIdx.x + i * THREADS;
        const int index = row * WIDTH + column;
        values[i] = upstream[index] * weight[column];
        centered[i] = (summed[index] - mean) * reciprocal;
        plain_sum += values[i];
        centered_sum += values[i] * centered[i];
    }
    const float plain = reduce_sum(plain_sum, plain_partial) / WIDTH;
    const float centered_mean = reduce_sum(centered_sum, centered_partial) / WIDTH;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int index = row * WIDTH + threadIdx.x + i * THREADS;
        gradient[index] = ((values[i] - plain) - centered[i] * centered_mean) * reciprocal +
                          (residual_upstream ? residual_upstream[index] : 0.0f);
    }
}

template<bool WEIGHT, bool BIAS>
static __global__ void add_norm_parameter_backward(const float* summed, const float* upstream,
                                                   const float* statistics, float* weight_gradient,
                                                   float* bias_gradient, int rows) {
    __shared__ float gamma_partial[32][9], beta_partial[32][9];
    const int feature = threadIdx.x % 8;
    const int sample = threadIdx.x / 8;
    const int column = blockIdx.x * 8 + feature;
    float gamma = 0.0f, beta = 0.0f;
    for (int row = sample; row < rows; row += 32) {
        const float value = upstream[row * WIDTH + column];
        if constexpr (WEIGHT)
            gamma += value * ((summed[row * WIDTH + column] - statistics[row * 2]) * statistics[row * 2 + 1]);
        if constexpr (BIAS) beta += value;
    }
    if constexpr (WEIGHT) gamma_partial[sample][feature] = gamma;
    if constexpr (BIAS) beta_partial[sample][feature] = beta;
    __syncthreads();
    if (sample == 0) {
        gamma = beta = 0.0f;
        for (int i = 0; i < 32; ++i) {
            if constexpr (WEIGHT) gamma += gamma_partial[i][feature];
            if constexpr (BIAS) beta += beta_partial[i][feature];
        }
        if constexpr (WEIGHT) weight_gradient[column] = gamma;
        if constexpr (BIAS) bias_gradient[column] = beta;
    }
}

extern "C" ROCMKERNELS_EXPORT int add_norm_fp32_forward(const float* x, const float* residual, const float* weight,
                                           const float* bias, float* summed, float* normalized,
                                           float* statistics, int rows, float epsilon, hipStream_t stream) {
    if (!x || !residual || !weight || !summed || !normalized || !statistics || rows <= 0)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(add_norm_forward, dim3(rows), dim3(THREADS), 0, stream,
                      x, residual, weight, bias, summed, normalized, statistics, epsilon);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int add_norm_fp32_backward(const float* summed, const float* upstream,
                                            const float* residual_upstream, const float* weight,
                                            const float* statistics, float* gradient,
                                            float* weight_gradient, float* bias_gradient, int rows, hipStream_t stream) {
    if (!summed || !upstream || !weight || !statistics || rows <= 0 || rows > 32 * 65535)
        return hipErrorInvalidValue;
    if (gradient) {
        hipLaunchKernelGGL(add_norm_input_backward, dim3(rows), dim3(THREADS), 0, stream,
                          summed, upstream, residual_upstream, weight, statistics, gradient);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) return error;
    }
    if (weight_gradient || bias_gradient) {
        if (weight_gradient && bias_gradient)
            hipLaunchKernelGGL((add_norm_parameter_backward<true, true>), dim3(WIDTH / 8), dim3(256), 0,
                              stream, summed, upstream, statistics, weight_gradient, bias_gradient, rows);
        else if (weight_gradient)
            hipLaunchKernelGGL((add_norm_parameter_backward<true, false>), dim3(WIDTH / 8), dim3(256), 0,
                              stream, summed, upstream, statistics, weight_gradient, bias_gradient, rows);
        else
            hipLaunchKernelGGL((add_norm_parameter_backward<false, true>), dim3(WIDTH / 8), dim3(256), 0,
                              stream, summed, upstream, statistics, weight_gradient, bias_gradient, rows);
    }
    return hipGetLastError();
}
