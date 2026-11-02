//
// Created by Dottik on 2/10/2026.
//
#include <ROCmKernels/Kernels/Activation/SwiGLU.hpp>

#include <hip/hip_runtime.h>
#include <math.h>

static __global__ void swiglu_forward(const float* input, float* output, int count, int width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int source = (index / width) * (2 * width) + index % width;
    const float value = input[source];
    const float denominator = 1.0f + expf(-value);
    output[index] = (value / denominator) * input[source + width];
}

static __global__ void swiglu_backward(const float* input, const float* upstream, float* gradient,
                                      int count, int width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int source = (index / width) * (2 * width) + index % width;
    const float value = input[source];
    const float denominator = 1.0f + expf(-value);
    const float sigmoid = 1.0f / denominator;
    gradient[source] = ((upstream[index] * input[source + width]) * sigmoid) *
                       fmaf(value, 1.0f - sigmoid, 1.0f);
    gradient[source + width] = upstream[index] * (value / denominator);
}

extern "C" ROCMKERNELS_EXPORT int swiglu_fp32_forward(const float* input, float* output, int count, int width,
                                         hipStream_t stream) {
    if (!input || !output || count <= 0 || width <= 0 || count % width || count > 1073741823) return hipErrorInvalidValue;
    hipLaunchKernelGGL(swiglu_forward, dim3((count + 255) / 256), dim3(256), 0, stream,
                       input, output, count, width);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int swiglu_fp32_backward(const float* input, const float* upstream, float* gradient,
                                          int count, int width, hipStream_t stream) {
    if (!input || !upstream || !gradient || count <= 0 || width <= 0 || count % width || count > 1073741823) return hipErrorInvalidValue;
    hipLaunchKernelGGL(swiglu_backward, dim3((count + 255) / 256), dim3(256), 0, stream,
                       input, upstream, gradient, count, width);
    return hipGetLastError();
}
