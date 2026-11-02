#include <ROCmKernels/Kernels/Generic/AttentionGeneric.hpp>

#include <hip/hip_runtime.h>

#include <cmath>

using ROCmKernels::Kernels::DType;
using ROCmKernels::Kernels::LoadScalar;
using ROCmKernels::Kernels::StoreScalar;
using ROCmKernels::Kernels::Strides;

namespace {
    constexpr int kWarps = 8;
    constexpr int kKeyTile = 32;
    // Padding keeps both the row-major staging writes and the transposed reads bank-conflict free.
    constexpr int kSharedPitch = 65;

    __device__ __forceinline__ float Broadcast(const float value, const int lane) {
        return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(value), lane));
    }

    __device__ __forceinline__ float WaveMaximum(const float value) {
        float result = value;
        for (int offset = 16; offset; offset >>= 1) result = fmaxf(result, __shfl_down(result, offset));
        return Broadcast(result, 0);
    }

    __device__ __forceinline__ float WaveTotal(const float value) {
        float result = value;
        for (int offset = 16; offset; offset >>= 1) result += __shfl_down(result, offset);
        return Broadcast(result, 0);
    }

    /// One warp owns one query row. Lane l scores key start + l by rebuilding the query dot product
    /// from lane-local registers, so no cross-lane reduction is needed before the softmax.
    template<typename T, int HEAD_WIDTH>
    __global__ void attention_generic_query(const T *q, const T *k, const T *v, const bool *mask, T *output, float *lse, const int heads,
            const int queries, const int keys, const int radius, const Strides qs, const Strides ks, const Strides vs, const Strides os,
            const float scale) {
        __shared__ float sharedK[kKeyTile][kSharedPitch];
        __shared__ float sharedV[kKeyTile][kSharedPitch];

        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        const int query = blockIdx.x * kWarps + warp;
        const int head = blockIdx.y;
        const int batch = blockIdx.z;
        const bool validQuery = query < queries;

        const T *queryBase = q + batch * qs.batch + head * qs.head;
        const T *keyBase = k + batch * ks.batch + head * ks.head;
        const T *valueBase = v + batch * vs.batch + head * vs.head;
        T *outputBase = output + batch * os.batch + head * os.head;
        const bool *maskBase = mask + batch * keys;

        float query0 = 0.0f;
        float query1 = 0.0f;
        if (validQuery && lane < HEAD_WIDTH) query0 = LoadScalar(queryBase + query * qs.token + lane);
        if constexpr (HEAD_WIDTH == 64) {
            if (validQuery) query1 = LoadScalar(queryBase + query * qs.token + lane + 32);
        }

        // The tile loop is block-wide so every warp reaches the barriers; the band check skips the tiles
        // that lie outside the owning query's window.
        const int blockFirst = blockIdx.x * kWarps;
        const int first = radius < 0 ? 0 : max(0, blockFirst - radius);
        const int last = radius < 0 ? keys : min(keys, blockFirst + kWarps + radius);
        const int queryFirst = radius < 0 ? 0 : max(0, query - radius);
        const int queryLast = radius < 0 ? keys : min(keys, query + radius + 1);

        float maximum = -INFINITY;
        float total = 0.0f;
        float accumulator0 = 0.0f;
        float accumulator1 = 0.0f;

        for (int start = first; start < last; start += kKeyTile) {
            for (int index = threadIdx.x; index < kKeyTile * HEAD_WIDTH; index += kWarps * 32) {
                const int row = index / HEAD_WIDTH;
                const int column = index % HEAD_WIDTH;
                const int key = start + row;
                const bool visible = key < last && maskBase[key];
                sharedK[row][column] = visible ? LoadScalar(keyBase + key * ks.token + column) : 0.0f;
                sharedV[row][column] = visible ? LoadScalar(valueBase + key * vs.token + column) : 0.0f;
            }
            __syncthreads();

            if (start < queryLast && start + kKeyTile > queryFirst) {
                float score = 0.0f;
                #pragma unroll
                for (int feature = 0; feature < HEAD_WIDTH; ++feature)
                    score = fmaf(Broadcast(feature < 32 ? query0 : query1, feature % 32), sharedK[lane][feature], score);

                const int key = start + lane;
                const bool active = validQuery && key < last && maskBase[key] && (radius < 0 || abs(query - key) <= radius);
                score = active ? score * scale : -INFINITY;

                const float next = fmaxf(maximum, WaveMaximum(score));
                const float old = isfinite(maximum) ? __expf(maximum - next) : 0.0f;
                const float weight = active ? __expf(score - next) : 0.0f;
                total = total * old + WaveTotal(weight);
                accumulator0 *= old;
                accumulator1 *= old;
                // Fully unrolled: readlane requires an immediate lane selector.
                #pragma unroll
                for (int row = 0; row < kKeyTile; ++row) {
                    const float probability = Broadcast(weight, row);
                    const float value0 = lane < HEAD_WIDTH ? sharedV[row][lane] : 0.0f;
                    accumulator0 = fmaf(probability, value0, accumulator0);
                    if constexpr (HEAD_WIDTH == 64) {
                        const float value1 = sharedV[row][lane + 32];
                        accumulator1 = fmaf(probability, value1, accumulator1);
                    }
                }
                maximum = next;
            }
            __syncthreads();
        }

        if (validQuery) {
            T *destination = outputBase + query * os.token;
            if (lane < HEAD_WIDTH) StoreScalar(destination + lane, total > 0 ? accumulator0 / total : 0.0f);
            if constexpr (HEAD_WIDTH == 64) StoreScalar(destination + lane + 32, total > 0 ? accumulator1 / total : 0.0f);
            if (lane == 0) {
                const int row = (batch * heads + head) * queries + query;
                lse[row * 2] = total > 0 ? maximum : -INFINITY;
                lse[row * 2 + 1] = total > 0 ? total : 0.0f;
            }
        }
    }

    template<typename T>
    hipError_t launch(const void *q, const void *k, const void *v, const bool *mask, void *output, float *lse, const int batch,
            const int heads, const int queries, const int keys, const int headWidth, const int radius, const Strides &qs, const Strides &ks,
            const Strides &vs, const Strides &os, const hipStream_t stream) {
        const dim3 grid((queries + kWarps - 1) / kWarps, heads, batch);
        const dim3 block(kWarps * 32);
        const float scale = 1.0f / std::sqrt(static_cast<float>(headWidth));

        const T *typedQ = static_cast<const T *>(q);
        const T *typedK = static_cast<const T *>(k);
        const T *typedV = static_cast<const T *>(v);
        T *typedOutput = static_cast<T *>(output);

        switch (headWidth) {
            case 16:
                hipLaunchKernelGGL((attention_generic_query<T, 16>), grid, block, 0, stream, typedQ, typedK, typedV, mask, typedOutput, lse,
                        heads, queries, keys, radius, qs, ks, vs, os, scale);
                break;
            case 32:
                hipLaunchKernelGGL((attention_generic_query<T, 32>), grid, block, 0, stream, typedQ, typedK, typedV, mask, typedOutput, lse,
                        heads, queries, keys, radius, qs, ks, vs, os, scale);
                break;
            case 64:
                hipLaunchKernelGGL((attention_generic_query<T, 64>), grid, block, 0, stream, typedQ, typedK, typedV, mask, typedOutput, lse,
                        heads, queries, keys, radius, qs, ks, vs, os, scale);
                break;
            default:
                return hipErrorInvalidValue;
        }
        return hipGetLastError();
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int attention_generic_forward(const void *q, const void *k, const void *v, const bool *mask, void *output,
        float *lse, int batch, int heads, int queries, int keys, int headWidth, int radius, Strides qs, Strides ks, Strides vs, Strides os,
        DType dtype, hipStream_t stream) {
    if (!q || !k || !v || !mask || !output || !lse || batch <= 0 || heads <= 0 || queries <= 0 || keys <= 0) return hipErrorInvalidValue;
    if (headWidth != 16 && headWidth != 32 && headWidth != 64) return hipErrorInvalidValue;
    if (radius < -1) return hipErrorInvalidValue;

    switch (dtype) {
        case DType::Float32:
            return static_cast<int>(launch<float>(q, k, v, mask, output, lse, batch, heads, queries, keys, headWidth, radius, qs, ks, vs, os,
                    stream));
        case DType::Float16:
            return static_cast<int>(launch<__half>(q, k, v, mask, output, lse, batch, heads, queries, keys, headWidth, radius, qs, ks, vs, os,
                    stream));
        default:
            return hipErrorInvalidValue;
    }
}
