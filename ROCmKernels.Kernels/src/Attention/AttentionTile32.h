//
// Created by Dottik on 2/10/2026.
//
static constexpr int TILE_ROWS = ROWS * 4;

template<bool SCALAR>
static __device__ float wave_read(float value, int lane) {
    if constexpr (SCALAR) return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(value), lane));
    else return __shfl(value, lane);
}

template<bool BACKWARD, int GROUP = 4, int WARPS = ROWS, bool SCALAR = true, bool STATISTICS_ONLY = false, bool SOFTMAX_DELTA = false, bool NORMALIZED_FORWARD = false, bool GLOBAL_FORWARD = false, bool CACHE_LOCAL = false>
static __device__ void attention_query_tiled_body(Input q, Input k, Input v, const bool* mask,
                                            Input output, const float* lse, Input upstream,
                                            float* result, float* statistics, int heads, int queries, int keys, int radius,
                                            Strides destination) {
    constexpr int TILE_ROWS = WARPS * GROUP;
    static_assert(!CACHE_LOCAL || (!BACKWARD && NORMALIZED_FORWARD && TILE_ROWS == 16));
    __shared__ float shared_k[32][33];
    __shared__ float shared_v[32][33];
    __shared__ float cached_probability[CACHE_LOCAL ? 16 : 1][CACHE_LOCAL ? 160 : 1];
    const int lane = threadIdx.x % 32;
    const int base = blockIdx.x * TILE_ROWS + (threadIdx.x / 32) * GROUP;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    float query_value[GROUP], upstream_value[GROUP], distance[GROUP], normalizer[GROUP];
    float maximum[GROUP], total[GROUP], accumulator[GROUP];
    float deferred_weight[GROUP], deferred_value[32];
    #pragma unroll
    for (int index = 0; index < GROUP; ++index) {
        const int row = (batch * heads + head) * queries + base + index;
        const bool valid = base + index < queries;
        query_value[index] = valid ? q.read(batch, head, base + index, lane) * SCALE_ROOT : 0.0f;
        maximum[index] = -INFINITY;
        total[index] = accumulator[index] = 0.0f;
        if constexpr (NORMALIZED_FORWARD) {
            maximum[index] = valid ? lse[row * 2] : -INFINITY;
            normalizer[index] = valid ? lse[row * 2 + 1] : 0.0f;
        }
        if constexpr (BACKWARD) {
            upstream_value[index] = valid ? upstream.read(batch, head, base + index, lane) : 0.0f;
            if constexpr (STATISTICS_ONLY) distance[index] = 0.0f;
            else if (SOFTMAX_DELTA) distance[index] = valid ? statistics[row] : 0.0f;
            else distance[index] = wave_sum(upstream_value[index] * (valid ? output.read(batch, head, base + index, lane) : 0.0f));
            maximum[index] = valid ? lse[row * 2] : -INFINITY;
            normalizer[index] = valid ? lse[row * 2 + 1] : 0.0f;
            if (!STATISTICS_ONLY && !SOFTMAX_DELTA && valid && lane == 0) statistics[row] = distance[index];
        }
    }
    const int chunk_start = (base / 256) * 256;
    const int chunk_queries = min(256, queries - chunk_start);
    const int query_tiles = (chunk_queries + 15) / 16;
    const int reference_left = radius < 0 ? 0 : max(0, chunk_start - radius);
    const int reference_right = radius < 0 ? keys : min(keys, chunk_start + chunk_queries + radius);
    const int raw_first = radius < 0 ? 0 : max(0, static_cast<int>(blockIdx.x) * TILE_ROWS - radius);
    const int first = NORMALIZED_FORWARD || (BACKWARD && STATISTICS_ONLY) ? reference_left + ((raw_first - reference_left) / 32) * 32 : raw_first;
    const int last = radius < 0 ? keys : min(keys, static_cast<int>(blockIdx.x) * TILE_ROWS + TILE_ROWS + radius);
    if constexpr (NORMALIZED_FORWARD) {
        for (int start = first; start < last; start += 32) {
            for (int index = threadIdx.x; index < 1024; index += WARPS * 32) {
                const int key = start + index / 32;
                shared_k[index / 32][index % 32] = key < last && mask[batch * keys + key] ?
                    k.read(batch, head, key, index % 32) * SCALE_ROOT : 0.0f;
            }
            __syncthreads();
            float score[GROUP] = {};
            #pragma unroll 4
            for (int column = 0; column < 32; ++column) {
                const float key = shared_k[lane][column];
                #pragma unroll
                for (int index = 0; index < GROUP; ++index)
                    score[index] = fmaf(wave_read<SCALAR>(query_value[index], column), key, score[index]);
            }
            const int key = start + lane;
            const bool valid_key = key < last && mask[batch * keys + key];
            #pragma unroll
            for (int index = 0; index < GROUP; ++index) {
                const bool active = base + index < queries && valid_key && (radius < 0 || abs(base + index - key) <= radius);
                const float shifted = active && maximum[index] != -INFINITY ? expf(score[index] - maximum[index]) : 0.0f;
                total[index] += shifted;
                if constexpr (CACHE_LOCAL)
                    if (key < last) cached_probability[(threadIdx.x / 32) * GROUP + index][key - first] = shifted;
            }
            __syncthreads();
        }
        #pragma unroll
        for (int index = 0; index < GROUP; ++index) {
            normalizer[index] = wave_sum(total[index]);
            if (base + index < queries && lane == 0)
                statistics[((batch * heads + head) * queries + base + index) * 2 + 1] = normalizer[index];
        }
    }
    const int reference_group = (batch * heads + head) * query_tiles + (base - chunk_start) / 16;
    const int reference_keys = reference_right - reference_left;
    const int full_keys = (reference_keys / 32) * 32;
    int stagger_tiles = 1;
    while (stagger_tiles < 32 && stagger_tiles * 2 <= full_keys / 32) stagger_tiles *= 2;
    constexpr bool REFERENCE_ORDER = NORMALIZED_FORWARD || (BACKWARD && SOFTMAX_DELTA && !STATISTICS_ONLY);
    const int rotation = REFERENCE_ORDER ? (reference_group & (stagger_tiles - 1)) * 32 : 0;
    const int iteration_length = REFERENCE_ORDER ? reference_keys : last - first;
    const int read_last = REFERENCE_ORDER ? reference_right : last;
    const bool split_columns = REFERENCE_ORDER && !GLOBAL_FORWARD && (radius >= 0 || BACKWARD);
    const bool defer_first = split_columns && reference_keys % 32 == 0 && stagger_tiles > 1;
    const int padded_length = ((iteration_length + 31) / 32) * 32;
    for (int iteration = 0; iteration < padded_length * (split_columns && !defer_first ? 2 : 1); iteration += 32) {
        const int half = iteration / padded_length;
        const int offset = iteration % padded_length;
        const int key_rotation = split_columns ? ((reference_group * 2 + half) & (stagger_tiles - 1)) * 32 : rotation;
        const int start = REFERENCE_ORDER ? reference_left + (offset < full_keys && key_rotation ? (offset + key_rotation) % full_keys : offset) : first + offset;
        for (int index = threadIdx.x; index < 1024; index += WARPS * 32) {
            const int row = index / 32;
            const int column = index % 32;
            const int key = start + row;
            const bool valid = key < read_last && mask[batch * keys + key];
            if constexpr (!CACHE_LOCAL)
                shared_k[row][column] = valid ? k.read(batch, head, key, column) * SCALE_ROOT : 0.0f;
            if constexpr (BACKWARD || !STATISTICS_ONLY)
                shared_v[row][column] = valid ? v.read(batch, head, key, column) : 0.0f;
        }
        __syncthreads();
        float score[GROUP] = {}, value_score[GROUP] = {}, weight[GROUP];
        const bool empty_tile = REFERENCE_ORDER && radius >= 0 && (start + 31 < raw_first || start >= last);
        if (!empty_tile && !CACHE_LOCAL) {
            #pragma unroll 4
            for (int column = 0; column < 32; ++column) {
                const float key = shared_k[lane][column];
                const float value = BACKWARD ? shared_v[lane][column] : 0.0f;
                #pragma unroll
                for (int index = 0; index < GROUP; ++index) {
                    score[index] = fmaf(wave_read<SCALAR>(query_value[index], column), key, score[index]);
                    if constexpr (BACKWARD)
                        value_score[index] = fmaf(wave_read<SCALAR>(upstream_value[index], column), value, value_score[index]);
                }
            }
        }
        const int key = start + lane;
        const bool valid_key = key < read_last && mask[batch * keys + key];
        #pragma unroll
        for (int index = 0; index < GROUP; ++index) {
            const bool active = base + index < queries && valid_key && (radius < 0 || abs(base + index - key) <= radius);
            score[index] = active ? score[index] : -INFINITY;
            if constexpr (BACKWARD) {
                const float probability = active && maximum[index] != -INFINITY ? expf(score[index] - maximum[index]) / normalizer[index] : 0.0f;
                if constexpr (STATISTICS_ONLY) total[index] += probability * value_score[index];
                else weight[index] = fmaf(-probability, distance[index], probability * value_score[index]);
            } else if constexpr (NORMALIZED_FORWARD) {
                if constexpr (CACHE_LOCAL)
                    weight[index] = active && maximum[index] != -INFINITY ? cached_probability[(threadIdx.x / 32) * GROUP + index][key - first] / normalizer[index] : 0.0f;
                else weight[index] = active && maximum[index] != -INFINITY ? expf(score[index] - maximum[index]) / normalizer[index] : 0.0f;
            } else {
                const float next_maximum = fmaxf(maximum[index], wave_max(score[index]));
                const float old_weight = isfinite(maximum[index]) ? expf(maximum[index] - next_maximum) : 0.0f;
                weight[index] = active ? expf(score[index] - next_maximum) : 0.0f;
                total[index] = total[index] * old_weight + wave_sum(weight[index]);
                accumulator[index] *= old_weight;
                maximum[index] = next_maximum;
            }
        }
        if (defer_first && iteration == 0) {
            #pragma unroll
            for (int index = 0; index < GROUP; ++index)
                deferred_weight[index] = weight[index];
            #pragma unroll
            for (int row = 0; row < 32; ++row)
                deferred_value[row] = BACKWARD ? shared_k[row][lane] : shared_v[row][lane];
        }
        if constexpr (!STATISTICS_ONLY) {
            #pragma unroll 4
            for (int row = 0; row < 32; ++row) {
                const float value = BACKWARD ? shared_k[row][lane] : shared_v[row][lane];
                #pragma unroll
                for (int index = 0; index < GROUP; ++index)
                    if (!split_columns || (defer_first ? iteration != 0 || lane < 16 : lane / 16 == half))
                        accumulator[index] = fmaf(wave_read<SCALAR>(weight[index], row), value, accumulator[index]);
            }
        }
        __syncthreads();
    }
    if (defer_first && lane >= 16) {
        #pragma unroll
        for (int row = 0; row < 32; ++row) {
            const float value = deferred_value[row];
            #pragma unroll
            for (int index = 0; index < GROUP; ++index)
                accumulator[index] = fmaf(wave_read<SCALAR>(deferred_weight[index], row), value, accumulator[index]);
        }
    }
    #pragma unroll
    for (int index = 0; index < GROUP; ++index) {
        if (base + index >= queries) continue;
        const int row = (batch * heads + head) * queries + base + index;
        if constexpr (BACKWARD) {
            if constexpr (STATISTICS_ONLY) {
                const float sum = wave_sum(total[index]);
                if (lane == 0) statistics[row] = sum;
            } else result[row * 32 + lane] = accumulator[index] * SCALE_ROOT;
        } else {
            if constexpr (!STATISTICS_ONLY)
                result[batch * destination.batch + head * destination.head + (base + index) * destination.token + lane] =
                    NORMALIZED_FORWARD ? accumulator[index] :
                    total[index] > 0 ? accumulator[index] / total[index] : isnan(total[index]) ? NAN : 0.0f;
            if (!NORMALIZED_FORWARD && lane == 0) {
                statistics[row * 2] = total[index] > 0 ? maximum[index] : isnan(total[index]) ? NAN : -INFINITY;
                statistics[row * 2 + 1] = total[index] > 0 ? total[index] : isnan(total[index]) ? NAN : 0.0f;
            }
        }
    }
}

template<bool BACKWARD, int GROUP = 4, int WARPS = ROWS, bool SCALAR = true, bool STATISTICS_ONLY = false, bool SOFTMAX_DELTA = false, bool NORMALIZED_FORWARD = false, bool GLOBAL_FORWARD = false, bool CACHE_LOCAL = false>
static __global__ void attention_query_tiled(Input q, Input k, Input v, const bool* mask,
                                            Input output, const float* lse, Input upstream,
                                            float* result, float* statistics, int heads, int queries, int keys, int radius,
                                            Strides destination) {
    attention_query_tiled_body<BACKWARD, GROUP, WARPS, SCALAR, STATISTICS_ONLY, SOFTMAX_DELTA, NORMALIZED_FORWARD, GLOBAL_FORWARD, CACHE_LOCAL>(
            q, k, v, mask, output, lse, upstream, result, statistics, heads, queries, keys, radius, destination);
}

template<int GROUP = 4, int WARPS = ROWS, bool SCALAR = true, bool CHUNKED = false>
static __global__ void attention_key_value_tiled(Input q, Input k, Input v, const bool* mask,
                                                const float* lse, Input upstream, const float* delta,
                                                float* gradient_k, float* gradient_v, int heads, int queries,
                                                int keys, int radius) {
    constexpr int TILE_ROWS = WARPS * GROUP;
    __shared__ float shared_q[32][33];
    __shared__ float shared_upstream[32][33];
    const int lane = threadIdx.x % 32;
    const int base = blockIdx.x * TILE_ROWS + (threadIdx.x / 32) * GROUP;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    float key_value[GROUP], value[GROUP], accumulator_k[GROUP] = {}, accumulator_v[GROUP] = {};
    float partial_k[GROUP] = {}, partial_v[GROUP] = {};
    const bool chunked = CHUNKED && (radius < 0 || radius % 32 == 0);
    bool valid_key[GROUP];
    #pragma unroll
    for (int index = 0; index < GROUP; ++index) {
        valid_key[index] = base + index < keys && mask[batch * keys + base + index];
        key_value[index] = valid_key[index] ? k.read(batch, head, base + index, lane) * SCALE_ROOT : 0.0f;
        value[index] = valid_key[index] ? v.read(batch, head, base + index, lane) : 0.0f;
    }
    const int first = radius < 0 ? 0 : max(0, static_cast<int>(blockIdx.x) * TILE_ROWS - radius);
    const int last = radius < 0 ? queries : min(queries, static_cast<int>(blockIdx.x) * TILE_ROWS + TILE_ROWS + radius);
    const int last_chunk = (last - 1) / 256;
    const int chunk_count = last > first ? last_chunk - first / 256 + 1 : 0;
    const int iterations = chunked ? chunk_count * 256 : last - first;
    for (int offset = 0; offset < iterations; offset += 32) {
        const int start = chunked ? (last_chunk - offset / 256) * 256 + offset % 256 : first + offset;
        if (start < first || start >= last) continue;
        for (int index = threadIdx.x; index < 1024; index += WARPS * 32) {
            const int row = index / 32;
            const int column = index % 32;
            const int query = start + row;
            shared_q[row][column] = query < last ? q.read(batch, head, query, column) * SCALE_ROOT : 0.0f;
            shared_upstream[row][column] = query < last ? upstream.read(batch, head, query, column) : 0.0f;
        }
        __syncthreads();
        float score[GROUP] = {}, value_score[GROUP] = {}, probability[GROUP], derivative[GROUP];
        #pragma unroll 4
        for (int column = 0; column < 32; ++column) {
            const float query = shared_q[lane][column];
            const float gradient = shared_upstream[lane][column];
            #pragma unroll
            for (int index = 0; index < GROUP; ++index) {
                score[index] = fmaf(query, wave_read<SCALAR>(key_value[index], column), score[index]);
                value_score[index] = fmaf(gradient, wave_read<SCALAR>(value[index], column), value_score[index]);
            }
        }
        const int query = start + lane;
        const int qrow = (batch * heads + head) * queries + query;
        const float maximum = query < last ? lse[qrow * 2] : -INFINITY;
        const float normalizer = query < last ? lse[qrow * 2 + 1] : 0.0f;
        const float distance = query < last ? delta[qrow] : 0.0f;
        #pragma unroll
        for (int index = 0; index < GROUP; ++index) {
            const bool active = valid_key[index] && query < last && (radius < 0 || abs(query - base - index) <= radius);
            probability[index] = active && maximum != -INFINITY ? expf(score[index] - maximum) / normalizer : 0.0f;
            derivative[index] = fmaf(-probability[index], distance, probability[index] * value_score[index]);
        }
        #pragma unroll 4
        for (int row = 0; row < 32; ++row) {
            const float query = shared_q[row][lane];
            const float gradient = shared_upstream[row][lane];
            #pragma unroll
            for (int index = 0; index < GROUP; ++index) {
                if (chunked) {
                    partial_k[index] = fmaf(wave_read<SCALAR>(derivative[index], row), query, partial_k[index]);
                    partial_v[index] = fmaf(wave_read<SCALAR>(probability[index], row), gradient, partial_v[index]);
                } else {
                    accumulator_k[index] = fmaf(wave_read<SCALAR>(derivative[index], row), query, accumulator_k[index]);
                    accumulator_v[index] = fmaf(wave_read<SCALAR>(probability[index], row), gradient, accumulator_v[index]);
                }
            }
        }
        if (chunked && ((start + 32) % 256 == 0 || start + 32 >= last)) {
            #pragma unroll
            for (int index = 0; index < GROUP; ++index) {
                accumulator_k[index] += partial_k[index] * SCALE_ROOT;
                accumulator_v[index] += partial_v[index];
                partial_k[index] = partial_v[index] = 0.0f;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int index = 0; index < GROUP; ++index) {
        if (base + index >= keys) continue;
        const int row = ((batch * heads + head) * keys + base + index) * 32 + lane;
        gradient_k[row] = chunked ? accumulator_k[index] : accumulator_k[index] * SCALE_ROOT;
        gradient_v[row] = accumulator_v[index];
    }
}
