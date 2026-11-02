//
// Created by Dottik on 2/10/2026.
//
#include <ROCmKernels/Kernels/Attention/Attention.hpp>

#include <hip/hip_runtime.h>
#include <math.h>

using ROCmKernels::Kernels::Strides;

static constexpr int ROWS = 4;
static constexpr float SCALE_ROOT = 0.4204482076268573f;

struct Input {
    const float* data;
    Strides strides;
    __device__ float read(int batch, int head, int token, int column) const {
        return data[batch * strides.batch + head * strides.head + token * strides.token + column];
    }
};

static __device__ float wave_sum(float value) {
    for (int offset = 16; offset; offset >>= 1) value += __shfl_down(value, offset);
    return __shfl(value, 0);
}

static __device__ float wave_max(float value) {
    for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_down(value, offset));
    return __shfl(value, 0);
}

template<bool BACKWARD>
static __global__ void attention_query(Input q, Input k, Input v, const bool* mask,
                                      Input output, const float* lse, Input upstream,
                                      float* result, float* statistics, int heads, int queries, int keys, int radius,
                                      Strides destination) {
    __shared__ float shared_k[32][33];
    __shared__ float shared_v[32][33];
    const int lane = threadIdx.x % 32;
    const int query = blockIdx.x * ROWS + threadIdx.x / 32;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int qrow = (batch * heads + head) * queries + query;
    const bool valid_query = query < queries;
    const float query_value = valid_query ? q.read(batch, head, query, lane) * SCALE_ROOT : 0.0f;
    float upstream_value = 0.0f;
    float distance = 0.0f;
    float normalizer = 0.0f;
    float maximum = -INFINITY;
    if constexpr (BACKWARD) {
        upstream_value = valid_query ? upstream.read(batch, head, query, lane) : 0.0f;
        const float previous = valid_query ? output.read(batch, head, query, lane) : 0.0f;
        distance = wave_sum(upstream_value * previous);
        maximum = valid_query ? lse[qrow * 2] : -INFINITY;
        normalizer = valid_query ? lse[qrow * 2 + 1] : 0.0f;
        if (valid_query && lane == 0) statistics[qrow] = distance;
    }
    const int first = radius < 0 ? 0 : max(0, static_cast<int>(blockIdx.x) * ROWS - radius);
    const int last = radius < 0 ? keys : min(keys, static_cast<int>(blockIdx.x) * ROWS + ROWS + radius);
    float total = 0.0f;
    float accumulator = 0.0f;
    for (int start = first; start < last; start += 32) {
        for (int index = threadIdx.x; index < 1024; index += ROWS * 32) {
            const int row = index / 32;
            const int column = index % 32;
            const int key = start + row;
            const bool valid_key = key < last && mask[batch * keys + key];
            shared_k[row][column] = valid_key ? k.read(batch, head, key, column) * SCALE_ROOT : 0.0f;
            shared_v[row][column] = valid_key ? v.read(batch, head, key, column) : 0.0f;
        }
        __syncthreads();
        float score = 0.0f;
        float value_score = 0.0f;
        #pragma unroll 4
        for (int column = 0; column < 32; ++column) {
            score = fmaf(__shfl(query_value, column), shared_k[lane][column], score);
            if constexpr (BACKWARD)
                value_score = fmaf(__shfl(upstream_value, column), shared_v[lane][column], value_score);
        }
        const int key = start + lane;
        const bool active = valid_query && key < last && mask[batch * keys + key] &&
                            (radius < 0 || abs(query - key) <= radius);
        score = active ? score : -INFINITY;
        if constexpr (BACKWARD) {
            const float probability = active && maximum != -INFINITY ? expf(score - maximum) / normalizer : 0.0f;
            const float derivative = probability * value_score - probability * distance;
            #pragma unroll 4
            for (int row = 0; row < 32; ++row)
                accumulator = fmaf(__shfl(derivative, row), shared_k[row][lane], accumulator);
        } else {
            const float next_maximum = fmaxf(maximum, wave_max(score));
            const float old_weight = isfinite(maximum) ? expf(maximum - next_maximum) : 0.0f;
            const float probability = active ? expf(score - next_maximum) : 0.0f;
            total = total * old_weight + wave_sum(probability);
            accumulator *= old_weight;
            #pragma unroll 4
            for (int row = 0; row < 32; ++row)
                accumulator = fmaf(__shfl(probability, row), shared_v[row][lane], accumulator);
            maximum = next_maximum;
        }
        __syncthreads();
    }
    if (valid_query) {
        if constexpr (BACKWARD) {
            result[qrow * 32 + lane] = accumulator * SCALE_ROOT;
        } else {
            result[batch * destination.batch + head * destination.head + query * destination.token + lane] =
                total > 0 ? accumulator / total : isnan(total) ? NAN : 0.0f;
            if (lane == 0) {
                statistics[qrow * 2] = total > 0 ? maximum : isnan(total) ? NAN : -INFINITY;
                statistics[qrow * 2 + 1] = total > 0 ? total : isnan(total) ? NAN : 0.0f;
            }
        }
    }
}

static __global__ void attention_key_value(Input q, Input k, Input v, const bool* mask,
                                          const float* lse, Input upstream, const float* delta,
                                          float* gradient_k, float* gradient_v, int heads, int queries,
                                          int keys, int radius) {
    __shared__ float shared_q[32][33];
    __shared__ float shared_upstream[32][33];
    const int lane = threadIdx.x % 32;
    const int key = blockIdx.x * ROWS + threadIdx.x / 32;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int krow = (batch * heads + head) * keys + key;
    const bool valid_key = key < keys && mask[batch * keys + key];
    const float key_value = valid_key ? k.read(batch, head, key, lane) * SCALE_ROOT : 0.0f;
    const float value = valid_key ? v.read(batch, head, key, lane) : 0.0f;
    const int first = radius < 0 ? 0 : max(0, static_cast<int>(blockIdx.x) * ROWS - radius);
    const int last = radius < 0 ? queries : min(queries, static_cast<int>(blockIdx.x) * ROWS + ROWS + radius);
    float accumulator_k = 0.0f;
    float accumulator_v = 0.0f;
    for (int start = first; start < last; start += 32) {
        for (int index = threadIdx.x; index < 1024; index += ROWS * 32) {
            const int row = index / 32;
            const int column = index % 32;
            const int query = start + row;
            shared_q[row][column] = query < last ? q.read(batch, head, query, column) * SCALE_ROOT : 0.0f;
            shared_upstream[row][column] = query < last ? upstream.read(batch, head, query, column) : 0.0f;
        }
        __syncthreads();
        float score = 0.0f;
        float value_score = 0.0f;
        #pragma unroll 4
        for (int column = 0; column < 32; ++column) {
            score = fmaf(shared_q[lane][column], __shfl(key_value, column), score);
            value_score = fmaf(shared_upstream[lane][column], __shfl(value, column), value_score);
        }
        const int query = start + lane;
        const int qrow = (batch * heads + head) * queries + query;
        const bool active = valid_key && query < last && (radius < 0 || abs(query - key) <= radius);
        const float maximum = active ? lse[qrow * 2] : -INFINITY;
        const float normalizer = active ? lse[qrow * 2 + 1] : 0.0f;
        const float probability = active && maximum != -INFINITY ? expf(score - maximum) / normalizer : 0.0f;
        const float derivative = probability * value_score - probability * (active ? delta[qrow] : 0.0f);
        #pragma unroll 4
        for (int row = 0; row < 32; ++row) {
            accumulator_k = fmaf(__shfl(derivative, row), shared_q[row][lane], accumulator_k);
            accumulator_v = fmaf(__shfl(probability, row), shared_upstream[row][lane], accumulator_v);
        }
        __syncthreads();
    }
    if (key < keys) {
        gradient_k[krow * 32 + lane] = accumulator_k * SCALE_ROOT;
        gradient_v[krow * 32 + lane] = accumulator_v;
    }
}

static __global__ void attention_single_query_kv(Input q, Input k, Input v, const bool* mask,
                                               const float* lse, Input upstream, const float* delta,
                                               float* gradient_k, float* gradient_v, int heads, int keys) {
    const int lane = threadIdx.x % 32;
    const int key = blockIdx.x * ROWS + threadIdx.x / 32;
    if (key >= keys) return;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int row = ((batch * heads + head) * keys + key) * 32 + lane;
    const int qrow = batch * heads + head;
    const float query = q.read(batch, head, 0, lane);
    const float grad = upstream.read(batch, head, 0, lane);
    const float score = wave_sum((query * SCALE_ROOT) * (k.read(batch, head, key, lane) * SCALE_ROOT));
    const float value_score = wave_sum(grad * v.read(batch, head, key, lane));
    const float probability = mask[batch * keys + key] && lse[qrow * 2] != -INFINITY ? expf(score - lse[qrow * 2]) / lse[qrow * 2 + 1] : 0.0f;
    const float derivative = probability * value_score - probability * delta[qrow];
    gradient_k[row] = (derivative * (query * SCALE_ROOT)) * SCALE_ROOT;
    gradient_v[row] = probability * grad;
}

#include "AttentionTile32.h"
#include "AttentionFlash32.h"

extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward(const float* q, const float* k, const float* v, const bool* mask,
                                            float* output, float* lse, int batch, int heads, int queries,
                                            int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, hipStream_t stream) {
    if (!q || !k || !v || !mask || !output || !lse || batch <= 0 || heads <= 0 || queries <= 0 || keys <= 0)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL((attention_query<false>), dim3((queries + ROWS - 1) / ROWS, heads, batch),
                      dim3(ROWS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{nullptr, os}), nullptr, (Input{nullptr, os}), output, lse,
                      heads, queries, keys, radius, os);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_backward(const float* q, const float* k, const float* v, const bool* mask,
                                             const float* output, const float* lse, const float* upstream,
                                             float* gradient_q, float* gradient_k, float* gradient_v, float* delta,
                                             int batch, int heads, int queries, int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, Strides us, hipStream_t stream) {
    if (!q || !k || !v || !mask || !output || !lse || !upstream || !gradient_q || !gradient_k || !gradient_v || !delta ||
        batch <= 0 || heads <= 0 || queries <= 0 || keys <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL((attention_query<true>), dim3((queries + ROWS - 1) / ROWS, heads, batch), dim3(ROWS * 32), 0,
                      stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{output, os}), lse, (Input{upstream, us}), gradient_q, delta, heads, queries, keys, radius, os);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    if (queries == 1 && radius < 0) {
        hipLaunchKernelGGL(attention_single_query_kv, dim3((keys + ROWS - 1) / ROWS, heads, batch), dim3(ROWS * 32), 0,
                          stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, lse, (Input{upstream, us}), delta, gradient_k, gradient_v, heads, keys);
    } else {
        hipLaunchKernelGGL(attention_key_value, dim3((keys + ROWS - 1) / ROWS, heads, batch), dim3(ROWS * 32), 0,
                          stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, lse, (Input{upstream, us}), delta, gradient_k, gradient_v, heads, queries, keys, radius);
    }
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward_tiled(const float* q, const float* k, const float* v, const bool* mask,
                                                  float* output, float* lse, int batch, int heads, int queries,
                                                  int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, hipStream_t stream) {
    if (queries < 32) return attention_fp32_forward(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    hipLaunchKernelGGL((attention_query_tiled<false>), dim3((queries + TILE_ROWS - 1) / TILE_ROWS, heads, batch),
                      dim3(ROWS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{nullptr, os}), nullptr, (Input{nullptr, os}), output, lse,
                      heads, queries, keys, radius, os);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_backward_tiled(const float* q, const float* k, const float* v, const bool* mask,
                                                   const float* output, const float* lse, const float* upstream,
                                                   float* gradient_q, float* gradient_k, float* gradient_v, float* delta,
                                                   int batch, int heads, int queries, int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, Strides us, hipStream_t stream) {
    if (queries < 32) return attention_fp32_backward(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                   delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    hipLaunchKernelGGL((attention_query_tiled<true>), dim3((queries + TILE_ROWS - 1) / TILE_ROWS, heads, batch),
                      dim3(ROWS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{output, os}), lse, (Input{upstream, us}), gradient_q, delta,
                      heads, queries, keys, radius, os);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    hipLaunchKernelGGL((attention_key_value_tiled<4>), dim3((keys + TILE_ROWS - 1) / TILE_ROWS, heads, batch),
                      dim3(ROWS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, lse, (Input{upstream, us}), delta, gradient_k, gradient_v,
                      heads, queries, keys, radius);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_statistics(const float* q, const float* k, const bool* mask, float* lse,
                                               int batch, int heads, int queries, int keys, int radius,
                                               Strides qs, Strides ks, hipStream_t stream) {
    if (!q || !k || !mask || !lse || batch <= 0 || heads <= 0 || queries <= 0 || keys <= 0 || radius < -1)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL((attention_query_tiled<false, 4, 8, true, true>), dim3((queries + 31) / 32, heads, batch),
                      dim3(256), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{nullptr, {}}), mask,
                      (Input{nullptr, {}}), nullptr, (Input{nullptr, {}}), nullptr, lse, heads, queries, keys, radius, (Strides{}));
    return hipGetLastError();
}

template<int GROUP, int WARPS = ROWS, bool SCALAR = true, bool NORMALIZED_FORWARD = false, bool CACHE_LOCAL = false>
static int attention_forward_group(const float* q, const float* k, const float* v, const bool* mask,
                                   float* output, float* lse, int batch, int heads, int queries,
                                   int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, hipStream_t stream) {
    if (queries < 32) return attention_fp32_forward(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    constexpr int count = WARPS * GROUP;
    if constexpr (NORMALIZED_FORWARD) {
        hipLaunchKernelGGL((attention_query_tiled<false, GROUP, WARPS, SCALAR, true>), dim3((queries + count - 1) / count, heads, batch),
                          dim3(WARPS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{nullptr, vs}), mask,
                          (Input{nullptr, os}), nullptr, (Input{nullptr, os}), nullptr, lse, heads, queries, keys, radius, os);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) return error;
    }
    if (NORMALIZED_FORWARD && radius < 0) {
        hipLaunchKernelGGL((attention_query_tiled<false, GROUP, WARPS, SCALAR, false, false, NORMALIZED_FORWARD, true>), dim3((queries + count - 1) / count, heads, batch),
                          dim3(WARPS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{nullptr, os}), lse, (Input{nullptr, os}), output, lse,
                          heads, queries, keys, radius, os);
    } else {
        hipLaunchKernelGGL((attention_query_tiled<false, GROUP, WARPS, SCALAR, false, false, NORMALIZED_FORWARD, false, CACHE_LOCAL>), dim3((queries + count - 1) / count, heads, batch),
                          dim3(WARPS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{nullptr, os}), NORMALIZED_FORWARD ? lse : nullptr, (Input{nullptr, os}), output, lse,
                          heads, queries, keys, radius, os);
    }
    return hipGetLastError();
}

template<int GROUP, int WARPS = ROWS, bool SCALAR = true, bool SOFTMAX_DELTA = false, int QUERY_GROUP = 0, int QUERY_WARPS = 0>
static int attention_backward_group(const float* q, const float* k, const float* v, const bool* mask,
                                    const float* output, const float* lse, const float* upstream,
                                    float* gradient_q, float* gradient_k, float* gradient_v, float* delta,
                                    int batch, int heads, int queries, int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, Strides us, hipStream_t stream, int stage = 0) {
    if (queries < 32) return attention_fp32_backward(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                   delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    constexpr int count = WARPS * GROUP;
    if (SOFTMAX_DELTA && (stage == 0 || stage == 1)) {
        hipLaunchKernelGGL((attention_query_tiled<true, GROUP, WARPS, SCALAR, true>), dim3((queries + count - 1) / count, heads, batch),
                          dim3(WARPS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{output, os}), lse,
                          (Input{upstream, us}), nullptr, delta, heads, queries, keys, radius, os);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess) return error;
    }
    if (stage == 1) return hipSuccess;
    constexpr int query_warps = SOFTMAX_DELTA ? (QUERY_WARPS ? QUERY_WARPS : 4) : WARPS;
    constexpr int query_group = SOFTMAX_DELTA ? (QUERY_GROUP ? QUERY_GROUP : 4) : GROUP;
    constexpr int query_count = query_group * query_warps;
    if (stage == 0 || stage == 2) {
        if (SOFTMAX_DELTA && QUERY_GROUP == 0 && radius >= 0) {
            hipLaunchKernelGGL((attention_query_tiled<true, 1, 16, SCALAR, false, true>), dim3((queries + 15) / 16, heads, batch),
                              dim3(512), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{output, os}), lse, (Input{upstream, us}), gradient_q, delta,
                              heads, queries, keys, radius, os);
        } else {
            hipLaunchKernelGGL((attention_query_tiled<true, query_group, query_warps, SCALAR, false, SOFTMAX_DELTA>), dim3((queries + query_count - 1) / query_count, heads, batch),
                              dim3(query_warps * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{output, os}), lse, (Input{upstream, us}), gradient_q, delta,
                              heads, queries, keys, radius, os);
        }
        hipError_t error = hipGetLastError();
        if (error != hipSuccess) return error;
    }
    if (stage == 2) return hipSuccess;
    hipLaunchKernelGGL((attention_key_value_tiled<GROUP, WARPS, SCALAR, SOFTMAX_DELTA>), dim3((keys + count - 1) / count, heads, batch),
                      dim3(WARPS * 32), 0, stream, (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, lse, (Input{upstream, us}), delta, gradient_k, gradient_v,
                      heads, queries, keys, radius);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_backward_profile(const float* q, const float* k, const float* v, const bool* mask,
                                                      const float* output, const float* lse, const float* upstream,
                                                      float* gradient_q, float* gradient_k, float* gradient_v, float* delta,
                                                      int batch, int heads, int queries, int keys, int radius, int stage,
                                                      Strides qs, Strides ks, Strides vs, Strides os, Strides us, hipStream_t stream) {
    if (stage < 1 || stage > 3 || queries < 32 || batch < 1 || heads < 1 || keys < 1 || radius < -1 ||
        !q || !k || !v || !mask || !output || !lse || !upstream || !gradient_q || !gradient_k || !gradient_v || !delta)
        return hipErrorInvalidValue;
    return attention_backward_group<4, 16, true, true>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                      delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream, stage);
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_query_profile(const float* q, const float* k, const float* v, const bool* mask,
                                                  const float* output, const float* lse, const float* upstream,
                                                  float* gradient_q, float* gradient_k, float* gradient_v, float* delta,
                                                  int batch, int heads, int queries, int keys, int radius, int tuning,
                                                  Strides qs, Strides ks, Strides vs, Strides os, Strides us, hipStream_t stream) {
    if (queries < 32 || batch < 1 || heads < 1 || keys < 1 || radius < -1 ||
        !q || !k || !v || !mask || !output || !lse || !upstream || !gradient_q || !gradient_k || !gradient_v || !delta)
        return hipErrorInvalidValue;
    switch (tuning) {
        case 14: return attention_backward_group<4, 16, true, true, 1, 4>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream, 2);
        case 18: return attention_backward_group<4, 16, true, true, 1, 8>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream, 2);
        case 116: return attention_backward_group<4, 16, true, true, 1, 16>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream, 2);
        case 24: return attention_backward_group<4, 16, true, true, 2, 4>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream, 2);
        case 28: return attention_backward_group<4, 16, true, true, 2, 8>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream, 2);
        case 44: return attention_backward_group<4, 16, true, true, 4, 4>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v, delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream, 2);
        default: return hipErrorInvalidValue;
    }
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward_tuned(const float* q, const float* k, const float* v, const bool* mask,
                                                  float* output, float* lse, int batch, int heads, int queries,
                                                  int keys, int radius, int group, Strides qs, Strides ks, Strides vs, Strides os, hipStream_t stream) {
    if (group == 2) return attention_forward_group<2>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    if (group == 8) return attention_forward_group<8>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    if (group == 4) return attention_forward_group<4, 4, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    if (group == 84) return attention_forward_group<4, 8>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    if (group == 164) return attention_forward_group<4, 16>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    if (group == 1164) return attention_forward_group<2, 8, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    if (group == 1264 && radius == 64) return attention_forward_group<2, 8, true, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    if (group == 1264) return attention_forward_group<2, 8, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    return hipErrorInvalidValue;
}

template<int WARPS, int GROUP, int KEY_TILE, bool FAST_EXP, bool VECTOR_BROADCAST>
static int attention_forward_flash(const float* q, const float* k, const float* v, const bool* mask,
                                   float* output, float* lse, int batch, int heads, int queries,
                                   int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, hipStream_t stream) {
    if (!q || !k || !v || !mask || !output || !lse || batch <= 0 || heads <= 0 || queries <= 0 || keys <= 0 || radius < -1)
        return hipErrorInvalidValue;
    if (queries < 32) return attention_fp32_forward(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
    constexpr int TILE_ROWS = WARPS * GROUP;
    hipLaunchKernelGGL((attention_query_flash<WARPS, GROUP, KEY_TILE, FAST_EXP, VECTOR_BROADCAST>),
                      dim3((queries + TILE_ROWS - 1) / TILE_ROWS, heads, batch), dim3(WARPS * 32), 0, stream,
                      (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, output, lse, heads, queries, keys, radius, os);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward_flash(const float* q, const float* k, const float* v, const bool* mask,
                                                  float* output, float* lse, int batch, int heads, int queries,
                                                  int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, hipStream_t stream) {
    return attention_forward_flash<16, 4, 32, true, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_forward_flash_tuned(const float* q, const float* k, const float* v, const bool* mask,
                                                        float* output, float* lse, int batch, int heads, int queries,
                                                        int keys, int radius, int variant, Strides qs, Strides ks, Strides vs, Strides os, hipStream_t stream) {
    switch (variant) {
        case 0: return attention_forward_flash<4, 4, 64, false, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 1: return attention_forward_flash<4, 4, 64, true, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 2: return attention_forward_flash<4, 4, 64, false, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 3: return attention_forward_flash<4, 4, 64, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 4: return attention_forward_flash<8, 4, 64, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 5: return attention_forward_flash<8, 2, 64, true, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 6: return attention_forward_flash<8, 2, 64, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 7: return attention_forward_flash<16, 2, 64, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 8: return attention_forward_flash<16, 2, 32, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 9: return attention_forward_flash<8, 2, 32, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 10: return attention_forward_flash<16, 4, 32, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 11: return attention_forward_flash<16, 4, 64, true, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 12: return attention_forward_flash<8, 2, 64, false, true>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 13: return attention_forward_flash<16, 4, 32, true, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 14: return attention_forward_flash<8, 4, 32, true, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 15: return attention_forward_flash<16, 4, 32, false, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 16: return attention_forward_flash<8, 8, 32, true, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 17: return attention_forward_flash<4, 8, 32, true, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        case 18: return attention_forward_flash<32, 4, 32, true, false>(q, k, v, mask, output, lse, batch, heads, queries, keys, radius, qs, ks, vs, os, stream);
        default: return hipErrorInvalidValue;
    }
}

extern "C" ROCMKERNELS_EXPORT int attention_fp32_backward_tuned(const float* q, const float* k, const float* v, const bool* mask,
                                                   const float* output, const float* lse, const float* upstream,
                                                   float* gradient_q, float* gradient_k, float* gradient_v, float* delta,
                                                   int batch, int heads, int queries, int keys, int radius, int group, Strides qs, Strides ks, Strides vs, Strides os, Strides us, hipStream_t stream) {
    if (group == 2) return attention_backward_group<2>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                     delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    if (group == 8) return attention_backward_group<8>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                     delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    if (group == 4) return attention_backward_group<4, 4, false>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                               delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    if (group == 84) return attention_backward_group<4, 8>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                         delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    if (group == 164) return attention_backward_group<4, 16>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                           delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    if (group == 1164) return attention_backward_group<4, 16, true, true>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                                      delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    if (group == 1264) return attention_backward_group<4, 16, true, true>(q, k, v, mask, output, lse, upstream, gradient_q, gradient_k, gradient_v,
                                                                      delta, batch, heads, queries, keys, radius, qs, ks, vs, os, us, stream);
    return hipErrorInvalidValue;
}

#ifdef ROCMKERNELS_PROFILE_SHIM
// The OpenCL runtime cannot patch device pointers inside a by-value struct, so the
// profiling build exposes the kernels with plain pointer arguments. The wrapper is
// inlined away; the profiled loop is the production one.
extern "C" __global__ void attention_flash_profile(const float* q, const float* k, const float* v, const bool* mask, float* output, float* lse,
        int heads, int queries, int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os) {
    attention_query_flash_body<16, 4, 32, true, false>(Input{q, qs}, Input{k, ks}, Input{v, vs}, mask, output, lse, heads, queries, keys, radius, os);
}

extern "C" __global__ void attention_tiled_profile(const float* q, const float* k, const float* v, const bool* mask, float* output, float* lse,
        int heads, int queries, int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os) {
    attention_query_tiled_body<false, 4, 4, true>(Input{q, qs}, Input{k, ks}, Input{v, vs}, mask, Input{nullptr, os}, nullptr,
            Input{nullptr, os}, output, lse, heads, queries, keys, radius, os);
}
#endif

