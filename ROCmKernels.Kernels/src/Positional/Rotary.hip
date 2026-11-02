//
// Created by Dottik on 2/10/2026.
//
#include <ROCmKernels/Kernels/Positional/Rotary.hpp>

#include <hip/hip_runtime.h>

using ROCmKernels::Kernels::Strides;

template<bool BACKWARD>
static __global__ void rotary(const float* q, const float* k, const float* cos, const float* sin,
                             float* oq, float* ok, int batch, int heads, int length,
                             Strides qs, Strides ks, Strides cs, Strides ss) {
    (void)batch;
    const int column = threadIdx.x % 16;
    const int head = blockIdx.x * 4 + threadIdx.x / 16;
    if (head >= heads) return;
    const int token = blockIdx.y;
    const int sample = blockIdx.z;
    const int index = ((sample * length + token) * heads + head) * 32 + column;
    const int qi = sample * qs.batch + head * qs.head + token * qs.token;
    const int ki = sample * ks.batch + head * ks.head + token * ks.token;
    const int ci = sample * cs.batch + token * cs.token;
    const int si = sample * ss.batch + token * ss.token;
    const float cosine_lo = cos[ci + column];
    const float cosine_hi = cos[ci + column + 16];
    const float sine_lo = sin[si + column];
    const float sine_hi = sin[si + column + 16];
    const float qlo = q[qi + column], qhi = q[qi + column + 16];
    const float klo = k[ki + column], khi = k[ki + column + 16];
    if constexpr (BACKWARD) {
        oq[index] = qlo * cosine_lo + qhi * sine_hi;
        oq[index + 16] = qhi * cosine_hi + -(qlo * sine_lo);
        ok[index] = klo * cosine_lo + khi * sine_hi;
        ok[index + 16] = khi * cosine_hi + -(klo * sine_lo);
    } else {
        oq[index] = qlo * cosine_lo + (-qhi) * sine_lo;
        oq[index + 16] = qhi * cosine_hi + qlo * sine_hi;
        ok[index] = klo * cosine_lo + (-khi) * sine_lo;
        ok[index + 16] = khi * cosine_hi + klo * sine_hi;
    }
}

extern "C" ROCMKERNELS_EXPORT int rotary_fp32(const float* q, const float* k, const float* cos, const float* sin,
                                  float* oq, float* ok, int batch, int heads, int length, int backward,
                                  Strides qs, Strides ks, Strides cs, Strides ss, hipStream_t stream) {
    if (!q || !k || !cos || !sin || !oq || !ok || batch <= 0 || heads <= 0 || length <= 0)
        return hipErrorInvalidValue;
    if (backward)
        hipLaunchKernelGGL((rotary<true>), dim3((heads + 3) / 4, length, batch), dim3(64), 0, stream, q, k, cos, sin, oq, ok,
                          batch, heads, length, qs, ks, cs, ss);
    else
        hipLaunchKernelGGL((rotary<false>), dim3((heads + 3) / 4, length, batch), dim3(64), 0, stream, q, k, cos, sin, oq, ok,
                          batch, heads, length, qs, ks, cs, ss);
    return hipGetLastError();
}
