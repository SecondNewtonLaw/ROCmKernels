#include <ROCmKernels/Kernels/Attention/Attention64.hpp>

#include <hip/hip_runtime.h>
#include <math.h>

using ROCmKernels::Kernels::Strides;

static constexpr int ROWS = 16;
static constexpr int D = 64;
static constexpr float SCALE_ROOT = 0.35355339059327373f;

struct Input {
    const float* data;
    Strides strides;
    __device__ float read(int batch, int head, int token, int column) const {
        return data[batch * strides.batch + head * strides.head + token * strides.token + column];
    }
};

static __device__ __forceinline__ float bcast(float value, int lane) {
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(value), lane));
}

static __device__ float wave_sum(float value) {
    for (int offset = 16; offset; offset >>= 1) value += __shfl_down(value, offset);
    return bcast(value, 0);
}

static __device__ float wave_max(float value) {
    for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_down(value, offset));
    return bcast(value, 0);
}

template<bool BACKWARD>
static __global__ void attention64_query(Input q, Input k, Input v, const bool* mask,
                                         Input output, const float* lse, Input upstream,
                                         float* result, float* statistics, int heads, int queries, int keys, int radius,
                                         Strides destination) {
    __shared__ float shared_k[32][D + 1];
    __shared__ float shared_v[32][D + 1];
    const int lane = threadIdx.x % 32;
    const int query = blockIdx.x * ROWS + threadIdx.x / 32;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int qrow = (batch * heads + head) * queries + query;
    const bool valid_query = query < queries;
    const float q0 = valid_query ? q.read(batch, head, query, lane) * SCALE_ROOT : 0.0f;
    const float q1 = valid_query ? q.read(batch, head, query, lane + 32) * SCALE_ROOT : 0.0f;
    float u0 = 0.0f, u1 = 0.0f, distance = 0.0f, normalizer = 0.0f, maximum = -INFINITY;
    if constexpr (BACKWARD) {
        u0 = valid_query ? upstream.read(batch, head, query, lane) : 0.0f;
        u1 = valid_query ? upstream.read(batch, head, query, lane + 32) : 0.0f;
        const float o0 = valid_query ? output.read(batch, head, query, lane) : 0.0f;
        const float o1 = valid_query ? output.read(batch, head, query, lane + 32) : 0.0f;
        distance = wave_sum(u0 * o0 + u1 * o1);
        maximum = valid_query ? lse[qrow * 2] : -INFINITY;
        normalizer = valid_query ? lse[qrow * 2 + 1] : 0.0f;
        if (valid_query && lane == 0) statistics[qrow] = distance;
    }
    const int first = radius < 0 ? 0 : max(0, static_cast<int>(blockIdx.x) * ROWS - radius);
    const int last = radius < 0 ? keys : min(keys, static_cast<int>(blockIdx.x) * ROWS + ROWS + radius);
    float total = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    for (int start = first; start < last; start += 32) {
        for (int index = threadIdx.x; index < 32 * D; index += ROWS * 32) {
            const int row = index / D;
            const int column = index % D;
            const int key = start + row;
            const bool valid_key = key < last && mask[batch * keys + key];
            shared_k[row][column] = valid_key ? k.read(batch, head, key, column) * SCALE_ROOT : 0.0f;
            shared_v[row][column] = valid_key ? v.read(batch, head, key, column) : 0.0f;
        }
        __syncthreads();
        float score = 0.0f, value_score = 0.0f;
        #pragma unroll 4
        for (int column = 0; column < 32; ++column) {
            score = fmaf(bcast(q0, column), shared_k[lane][column], score);
            score = fmaf(bcast(q1, column), shared_k[lane][column + 32], score);
            if constexpr (BACKWARD) {
                value_score = fmaf(bcast(u0, column), shared_v[lane][column], value_score);
                value_score = fmaf(bcast(u1, column), shared_v[lane][column + 32], value_score);
            }
        }
        const int key = start + lane;
        const bool active = valid_query && key < last && mask[batch * keys + key] &&
                            (radius < 0 || abs(query - key) <= radius);
        score = active ? score : -INFINITY;
        if constexpr (BACKWARD) {
            const float probability = active && maximum != -INFINITY ? expf(score - maximum) / normalizer : 0.0f;
            const float derivative = probability * value_score - probability * distance;
            #pragma unroll 4
            for (int row = 0; row < 32; ++row) {
                const float d = bcast(derivative, row);
                acc0 = fmaf(d, shared_k[row][lane], acc0);
                acc1 = fmaf(d, shared_k[row][lane + 32], acc1);
            }
        } else {
            const float next_maximum = fmaxf(maximum, wave_max(score));
            const float old_weight = isfinite(maximum) ? expf(maximum - next_maximum) : 0.0f;
            const float probability = active ? expf(score - next_maximum) : 0.0f;
            total = total * old_weight + wave_sum(probability);
            acc0 *= old_weight;
            acc1 *= old_weight;
            #pragma unroll 4
            for (int row = 0; row < 32; ++row) {
                const float p = bcast(probability, row);
                acc0 = fmaf(p, shared_v[row][lane], acc0);
                acc1 = fmaf(p, shared_v[row][lane + 32], acc1);
            }
            maximum = next_maximum;
        }
        __syncthreads();
    }
    if (valid_query) {
        if constexpr (BACKWARD) {
            result[qrow * D + lane] = acc0 * SCALE_ROOT;
            result[qrow * D + lane + 32] = acc1 * SCALE_ROOT;
        } else {
            const float scale = total > 0 ? 1.0f / total : 0.0f;
            const bool bad = isnan(total);
            const int base = batch * destination.batch + head * destination.head + query * destination.token;
            ((float*)result)[base + lane] = bad ? NAN : acc0 * scale;
            ((float*)result)[base + lane + 32] = bad ? NAN : acc1 * scale;
            if (lane == 0) {
                statistics[qrow * 2] = total > 0 ? maximum : bad ? NAN : -INFINITY;
                statistics[qrow * 2 + 1] = total > 0 ? total : bad ? NAN : 0.0f;
            }
        }
    }
}

static __global__ void attention64_key_value(Input q, Input k, Input v, const bool* mask,
                                             const float* lse, Input upstream, const float* delta,
                                             float* gradient_k, float* gradient_v, int heads, int queries,
                                             int keys, int radius) {
    __shared__ float shared_q[32][D + 1];
    __shared__ float shared_u[32][D + 1];
    const int lane = threadIdx.x % 32;
    const int key = blockIdx.x * ROWS + threadIdx.x / 32;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int krow = (batch * heads + head) * keys + key;
    const bool valid_key = key < keys && mask[batch * keys + key];
    const float k0 = valid_key ? k.read(batch, head, key, lane) * SCALE_ROOT : 0.0f;
    const float k1 = valid_key ? k.read(batch, head, key, lane + 32) * SCALE_ROOT : 0.0f;
    const float v0 = valid_key ? v.read(batch, head, key, lane) : 0.0f;
    const float v1 = valid_key ? v.read(batch, head, key, lane + 32) : 0.0f;
    const int first = radius < 0 ? 0 : max(0, static_cast<int>(blockIdx.x) * ROWS - radius);
    const int last = radius < 0 ? queries : min(queries, static_cast<int>(blockIdx.x) * ROWS + ROWS + radius);
    float gk0 = 0.0f, gk1 = 0.0f, gv0 = 0.0f, gv1 = 0.0f;
    for (int start = first; start < last; start += 32) {
        for (int index = threadIdx.x; index < 32 * D; index += ROWS * 32) {
            const int row = index / D;
            const int column = index % D;
            const int query = start + row;
            shared_q[row][column] = query < last ? q.read(batch, head, query, column) * SCALE_ROOT : 0.0f;
            shared_u[row][column] = query < last ? upstream.read(batch, head, query, column) : 0.0f;
        }
        __syncthreads();
        float score = 0.0f, value_score = 0.0f;
        #pragma unroll 4
        for (int column = 0; column < 32; ++column) {
            score = fmaf(shared_q[lane][column], bcast(k0, column), score);
            score = fmaf(shared_q[lane][column + 32], bcast(k1, column), score);
            value_score = fmaf(shared_u[lane][column], bcast(v0, column), value_score);
            value_score = fmaf(shared_u[lane][column + 32], bcast(v1, column), value_score);
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
            const float d = bcast(derivative, row);
            const float p = bcast(probability, row);
            gk0 = fmaf(d, shared_q[row][lane], gk0);
            gk1 = fmaf(d, shared_q[row][lane + 32], gk1);
            gv0 = fmaf(p, shared_u[row][lane], gv0);
            gv1 = fmaf(p, shared_u[row][lane + 32], gv1);
        }
        __syncthreads();
    }
    if (key < keys) {
        gradient_k[krow * D + lane] = gk0 * SCALE_ROOT;
        gradient_k[krow * D + lane + 32] = gk1 * SCALE_ROOT;
        gradient_v[krow * D + lane] = gv0;
        gradient_v[krow * D + lane + 32] = gv1;
    }
}

extern "C" ROCMKERNELS_EXPORT int attention64_fp32_forward(const float* q, const float* k, const float* v, const bool* mask,
                                               float* output, float* lse, int batch, int heads, int queries,
                                               int keys, int radius, Strides qs, Strides ks, Strides vs, Strides os, hipStream_t stream) {
    if (!q || !k || !v || !mask || !output || !lse || batch <= 0 || heads <= 0 || queries <= 0 || keys <= 0)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL((attention64_query<false>), dim3((queries + ROWS - 1) / ROWS, heads, batch), dim3(ROWS * 32), 0, stream,
                       (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{nullptr, os}), nullptr, (Input{nullptr, os}),
                       output, lse, heads, queries, keys, radius, os);
    return hipGetLastError();
}

extern "C" ROCMKERNELS_EXPORT int attention64_fp32_backward(const float* q, const float* k, const float* v, const bool* mask,
                                                const float* output, const float* lse, const float* upstream,
                                                float* gradient_q, float* gradient_k, float* gradient_v, float* delta,
                                                int batch, int heads, int queries, int keys, int radius,
                                                Strides qs, Strides ks, Strides vs, Strides os, Strides us, hipStream_t stream) {
    if (!q || !k || !v || !mask || !output || !lse || !upstream || !gradient_q || !gradient_k || !gradient_v || !delta ||
        batch <= 0 || heads <= 0 || queries <= 0 || keys <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL((attention64_query<true>), dim3((queries + ROWS - 1) / ROWS, heads, batch), dim3(ROWS * 32), 0, stream,
                       (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, (Input{output, os}), lse, (Input{upstream, us}),
                       gradient_q, delta, heads, queries, keys, radius, os);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return error;
    hipLaunchKernelGGL(attention64_key_value, dim3((keys + ROWS - 1) / ROWS, heads, batch), dim3(ROWS * 32), 0, stream,
                       (Input{q, qs}), (Input{k, ks}), (Input{v, vs}), mask, lse, (Input{upstream, us}), delta,
                       gradient_k, gradient_v, heads, queries, keys, radius);
    return hipGetLastError();
}
