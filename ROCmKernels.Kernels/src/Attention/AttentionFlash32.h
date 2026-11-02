#pragma once

static __device__ __forceinline__ float Component(const float2 &value, const int index) {
    return index == 0 ? value.x : value.y;
}

static __device__ __forceinline__ float Component(const float4 &value, const int index) {
    return index == 0 ? value.x : index == 1 ? value.y : index == 2 ? value.z : value.w;
}

template<bool FAST>
static __device__ __forceinline__ float FlashExp(const float value) {
    if constexpr (FAST) return __expf(value);
    else return expf(value);
}

/// Flash attention forward for 32-wide heads.
///
/// A warp owns GROUP queries. Each lane owns one feature of those queries and one key of the
/// current tile, so scores are produced without a cross-lane reduction. K and V are staged in
/// shared memory. The softmax weights can be written to shared memory and read back as one
/// vector broadcast per key instead of GROUP scalar lane reads.
template<int WARPS, int GROUP, int KEY_TILE, bool FAST_EXP, bool VECTOR_WEIGHT>
static __device__ void attention_query_flash_body(Input q, Input k, Input v, const bool *mask, float *result, float *statistics, const int heads,
        const int queries, const int keys, const int radius, const Strides destination) {
    constexpr int TILE_ROWS = WARPS * GROUP;
    constexpr int SLICES = KEY_TILE / 32;
    static_assert(KEY_TILE == 32 || KEY_TILE == 64);
    static_assert(!VECTOR_WEIGHT || GROUP == 2 || GROUP == 4);

    using WeightType = std::conditional_t<GROUP == 2, float2, float4>;
    __shared__ __align__(16) float shared_k[KEY_TILE][33];
    __shared__ __align__(16) float shared_v[KEY_TILE][33];
    __shared__ __align__(16) WeightType shared_p[VECTOR_WEIGHT ? WARPS : 1][VECTOR_WEIGHT ? KEY_TILE : 1];

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int base = blockIdx.x * TILE_ROWS + warp * GROUP;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;

    const float *queryBase = q.data + batch * q.strides.batch + head * q.strides.head;
    const float *keyBase = k.data + batch * k.strides.batch + head * k.strides.head;
    const float *valueBase = v.data + batch * v.strides.batch + head * v.strides.head;
    float *outputBase = result + batch * destination.batch + head * destination.head;
    const bool *maskBase = mask + batch * keys;

    float query[GROUP];
    float accumulator[GROUP] = {};
    float maximum[GROUP];
    float total[GROUP] = {};
    #pragma unroll
    for (int index = 0; index < GROUP; ++index) {
        const int row = base + index;
        query[index] = row < queries ? queryBase[row * q.strides.token + lane] * SCALE_ROOT : 0.0f;
        maximum[index] = -INFINITY;
    }

    const int tileFirst = blockIdx.x * TILE_ROWS;
    const int first = radius < 0 ? 0 : max(0, tileFirst - radius);
    const int last = radius < 0 ? keys : min(keys, tileFirst + TILE_ROWS + radius);

    for (int start = first; start < last; start += KEY_TILE) {
        for (int index = threadIdx.x; index < KEY_TILE * 32; index += WARPS * 32) {
            const int row = index / 32;
            const int column = index % 32;
            const int key = start + row;
            const bool valid = key < last && maskBase[key];
            shared_k[row][column] = valid ? keyBase[key * k.strides.token + column] * SCALE_ROOT : 0.0f;
            shared_v[row][column] = valid ? valueBase[key * v.strides.token + column] : 0.0f;
        }
        __syncthreads();

        float score[GROUP][SLICES] = {};
        #pragma unroll 4
        for (int column = 0; column < 32; ++column) {
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                const float keyValue = shared_k[lane + 32 * slice][column];
                #pragma unroll
                for (int index = 0; index < GROUP; ++index)
                    score[index][slice] = fmaf(wave_read<true>(query[index], column), keyValue, score[index][slice]);
            }
        }

        float weight[GROUP][SLICES];
        #pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            const int key = start + lane + 32 * slice;
            const bool validKey = key < last && maskBase[key];
            #pragma unroll
            for (int index = 0; index < GROUP; ++index)
                if (!validKey || base + index >= queries || (radius >= 0 && abs(base + index - key) > radius)) score[index][slice] = -INFINITY;
        }

        #pragma unroll
        for (int index = 0; index < GROUP; ++index) {
            float tileMaximum = score[index][0];
            #pragma unroll
            for (int slice = 1; slice < SLICES; ++slice) tileMaximum = fmaxf(tileMaximum, score[index][slice]);
            const float nextMaximum = fmaxf(maximum[index], wave_max(tileMaximum));
            const float oldWeight = isfinite(maximum[index]) ? FlashExp<FAST_EXP>(maximum[index] - nextMaximum) : 0.0f;
            float tileTotal = 0.0f;
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                weight[index][slice] = score[index][slice] == -INFINITY ? 0.0f : FlashExp<FAST_EXP>(score[index][slice] - nextMaximum);
                tileTotal += weight[index][slice];
            }
            total[index] = total[index] * oldWeight + wave_sum(tileTotal);
            accumulator[index] *= oldWeight;
            maximum[index] = nextMaximum;
        }

        if constexpr (VECTOR_WEIGHT) {
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                const int row = lane + 32 * slice;
                if constexpr (GROUP == 2) shared_p[warp][row] = make_float2(weight[0][slice], weight[1][slice]);
                else shared_p[warp][row] = make_float4(weight[0][slice], weight[1][slice], weight[2][slice], weight[3][slice]);
            }
            __syncwarp();
        }

        #pragma unroll 4
        for (int row = 0; row < 32; ++row) {
            const float value = shared_v[row][lane];
            if constexpr (VECTOR_WEIGHT) {
                const WeightType broadcast = shared_p[warp][row];
                #pragma unroll
                for (int index = 0; index < GROUP; ++index)
                    accumulator[index] = fmaf(Component(broadcast, index), value, accumulator[index]);
            } else {
                #pragma unroll
                for (int index = 0; index < GROUP; ++index)
                    accumulator[index] = fmaf(wave_read<true>(weight[index][0], row), value, accumulator[index]);
            }
        }
        if constexpr (SLICES == 2) {
            #pragma unroll 4
            for (int row = 32; row < 64; ++row) {
                const float value = shared_v[row][lane];
                if constexpr (VECTOR_WEIGHT) {
                    const WeightType broadcast = shared_p[warp][row];
                    #pragma unroll
                    for (int index = 0; index < GROUP; ++index)
                        accumulator[index] = fmaf(Component(broadcast, index), value, accumulator[index]);
                } else {
                    #pragma unroll
                    for (int index = 0; index < GROUP; ++index)
                        accumulator[index] = fmaf(wave_read<true>(weight[index][1], row % 32), value, accumulator[index]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int index = 0; index < GROUP; ++index) {
        const int row = base + index;
        if (row >= queries) continue;
        const int statisticsRow = (batch * heads + head) * queries + row;
        const float normalizer = total[index];
        outputBase[row * destination.token + lane] =
                normalizer > 0 ? accumulator[index] / normalizer : isnan(normalizer) ? NAN : 0.0f;
        if (lane == 0) {
            statistics[statisticsRow * 2] = normalizer > 0 ? maximum[index] : isnan(normalizer) ? NAN : -INFINITY;
            statistics[statisticsRow * 2 + 1] = normalizer > 0 ? normalizer : isnan(normalizer) ? NAN : 0.0f;
        }
    }
}

template<int WARPS, int GROUP, int KEY_TILE, bool FAST_EXP, bool VECTOR_WEIGHT>
static __global__ void attention_query_flash(Input q, Input k, Input v, const bool *mask, float *result, float *statistics, const int heads,
        const int queries, const int keys, const int radius, const Strides destination) {
    attention_query_flash_body<WARPS, GROUP, KEY_TILE, FAST_EXP, VECTOR_WEIGHT>(q, k, v, mask, result, statistics, heads, queries, keys, radius,
            destination);
}
