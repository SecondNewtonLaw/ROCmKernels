//
// RDNA2 (gfx1031) instruction microbenchmarks. Each case runs a dependent chain
// (latency) or independent chains (throughput) from one work-group and reports
// cycles per instruction from s_memtime. Throughput cases issue 32 instructions
// per iteration so the loop overhead is amortized.
//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
    constexpr int kThreads = 32;
    constexpr int kChain = 16;

    typedef float v4f32 __attribute__((ext_vector_type(4)));

    // ------------------------------------------------------------- VALU steps

    struct FmaStep {
        struct State { float a[8]; float b; float c; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = 1.0f + lane * 1.0e-3f + i;
            s.b = 1.0001f;
            s.c = 0.5f;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_fma_f32 %0, %0, %1, %2" : "+v"(s.a[0]) : "v"(s.b), "v"(s.c));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_fma_f32 %0, %0, %1, %2" : "+v"(s.a[i]) : "v"(s.b), "v"(s.c));
        }
        __device__ static float Sink(const State &s) { return s.a[0] + s.a[1] + s.a[2] + s.a[3]; }
    };

    struct FmacStep {
        struct State { float a[8]; float b; float c; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = 1.0f + lane * 1.0e-3f + i;
            s.b = 1.0001f;
            s.c = 0.5f;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(s.a[0]) : "v"(s.b), "v"(s.c));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(s.a[i]) : "v"(s.b), "v"(s.c));
        }
        __device__ static float Sink(const State &s) { return s.a[0] + s.a[1]; }
    };

#define DEFINE_FLOAT_STEP(Name, Instruction)                                                   \
    struct Name {                                                                              \
        struct State { float a[8]; float b; };                                                 \
        __device__ static void Init(State &s, const int lane) {                                \
            for (int i = 0; i < 8; ++i) s.a[i] = 1.0f + lane * 1.0e-3f + i;                    \
            s.b = 0.5f;                                                                        \
        }                                                                                      \
        __device__ static void Dependent(State &s, float *) {                                   \
            asm volatile(Instruction " %0, %0, %1" : "+v"(s.a[0]) : "v"(s.b));                 \
        }                                                                                      \
        __device__ static void Wide(State &s, float *, const int i) {                          \
            asm volatile(Instruction " %0, %0, %1" : "+v"(s.a[i]) : "v"(s.b));                 \
        }                                                                                      \
        __device__ static float Sink(const State &s) { return s.a[0]; }                        \
    };

    DEFINE_FLOAT_STEP(AddStep, "v_add_f32")
    DEFINE_FLOAT_STEP(MulStep, "v_mul_f32")

#define DEFINE_UNARY_FLOAT_STEP(Name, Instruction)                                             \
    struct Name {                                                                              \
        struct State { float a[8]; };                                                          \
        __device__ static void Init(State &s, const int lane) {                                \
            for (int i = 0; i < 8; ++i) s.a[i] = 0.5f + lane * 1.0e-3f + i * 0.125f;           \
        }                                                                                      \
        __device__ static void Dependent(State &s, float *) {                                   \
            asm volatile(Instruction " %0, %0" : "+v"(s.a[0]));                                \
        }                                                                                      \
        __device__ static void Wide(State &s, float *, const int i) {                          \
            asm volatile(Instruction " %0, %0" : "+v"(s.a[i]));                                \
        }                                                                                      \
        __device__ static float Sink(const State &s) { return s.a[0]; }                        \
    };

    DEFINE_UNARY_FLOAT_STEP(ExpStep, "v_exp_f32")
    DEFINE_UNARY_FLOAT_STEP(LogStep, "v_log_f32")
    DEFINE_UNARY_FLOAT_STEP(RcpStep, "v_rcp_f32")
    DEFINE_UNARY_FLOAT_STEP(SqrtStep, "v_sqrt_f32")
    DEFINE_UNARY_FLOAT_STEP(RsqStep, "v_rsq_f32")

    struct MovStep {
        struct State { float a[8]; float b; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = 1.0f + lane + i;
            s.b = 0.5f;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_mov_b32 %0, %1" : "=v"(s.a[0]) : "v"(s.b));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_mov_b32 %0, %1" : "=v"(s.a[i]) : "v"(s.b));
        }
        __device__ static float Sink(const State &s) { return s.a[0]; }
    };

    struct Dot2Step {
        struct State { float a[8]; unsigned b; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = lane * 0.001f + i;
            s.b = 0x3c003c00u;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_dot2_f32_f16 %0, %1, %1, %0" : "+v"(s.a[0]) : "v"(s.b));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_dot2_f32_f16 %0, %1, %1, %0" : "+v"(s.a[i]) : "v"(s.b));
        }
        __device__ static float Sink(const State &s) { return s.a[0]; }
    };

    struct Dot4Step {
        struct State { int a[8]; unsigned b; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = lane + i;
            s.b = 0x01010101u;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_dot4c_i32_i8 %0, %1, %1" : "+v"(s.a[0]) : "v"(s.b));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_dot4c_i32_i8 %0, %1, %1" : "+v"(s.a[i]) : "v"(s.b));
        }
        __device__ static float Sink(const State &s) { return static_cast<float>(s.a[0]); }
    };

    struct PkFmaStep {
        struct State { unsigned a[8]; unsigned b; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = 0x3c003c00u + lane + i;
            s.b = 0x38003800u;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_pk_fma_f16 %0, %1, %1, %0" : "+v"(s.a[0]) : "v"(s.b));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_pk_fma_f16 %0, %1, %1, %0" : "+v"(s.a[i]) : "v"(s.b));
        }
        __device__ static float Sink(const State &s) { return static_cast<float>(s.a[0]); }
    };

#define DEFINE_INT_STEP(Name, Instruction)                                                     \
    struct Name {                                                                              \
        struct State { int a[8]; int b; };                                                     \
        __device__ static void Init(State &s, const int lane) {                                \
            for (int i = 0; i < 8; ++i) s.a[i] = 1 + lane + i;                                 \
            s.b = 3;                                                                           \
        }                                                                                      \
        __device__ static void Dependent(State &s, float *) {                                   \
            asm volatile(Instruction " %0, %0, %1" : "+v"(s.a[0]) : "v"(s.b));                 \
        }                                                                                      \
        __device__ static void Wide(State &s, float *, const int i) {                          \
            asm volatile(Instruction " %0, %0, %1" : "+v"(s.a[i]) : "v"(s.b));                 \
        }                                                                                      \
        __device__ static float Sink(const State &s) { return static_cast<float>(s.a[0]); }    \
    };

    DEFINE_INT_STEP(AddU32Step, "v_add_nc_u32")
    DEFINE_INT_STEP(MulLoU32Step, "v_mul_lo_u32")
    DEFINE_INT_STEP(LshlStep, "v_lshlrev_b32")

    struct MadU24Step {
        struct State { int a[8]; int b; int c; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = 1 + lane + i;
            s.b = 3;
            s.c = 7;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_mad_u32_u24 %0, %1, %2, %0" : "+v"(s.a[0]) : "v"(s.b), "v"(s.c));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_mad_u32_u24 %0, %1, %2, %0" : "+v"(s.a[i]) : "v"(s.b), "v"(s.c));
        }
        __device__ static float Sink(const State &s) { return static_cast<float>(s.a[0]); }
    };

    struct BfeStep {
        struct State { int a[8]; int b; int c; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = 0x12345678 + lane + i;
            s.b = 4;
            s.c = 8;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_bfe_u32 %0, %0, %1, %2" : "+v"(s.a[0]) : "v"(s.b), "v"(s.c));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_bfe_u32 %0, %0, %1, %2" : "+v"(s.a[i]) : "v"(s.b), "v"(s.c));
        }
        __device__ static float Sink(const State &s) { return static_cast<float>(s.a[0]); }
    };

    struct SaluStep {
        struct State { int a[8]; };
        __device__ static void Init(State &s, const int) {
            for (int i = 0; i < 8; ++i) s.a[i] = i + 1;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("s_add_u32 %0, %0, 1" : "+s"(s.a[0]) : : "scc");
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("s_add_u32 %0, %0, 1" : "+s"(s.a[i]) : : "scc");
        }
        __device__ static float Sink(const State &s) { return static_cast<float>(s.a[0]); }
    };
    struct CmpSelectStep {
        struct State { float a[8]; float b; float c; unsigned mask; };
        __device__ static void Init(State &s, const int lane) {
            for (int i = 0; i < 8; ++i) s.a[i] = 0.25f + lane * 1.0e-3f + i;
            s.b = 0.5f;
            s.c = 2.0f;
            s.mask = 0;
        }
        __device__ static void Dependent(State &s, float *) {
            asm volatile("v_cmp_gt_f32 %0, %1, %2" : "=s"(s.mask) : "v"(s.a[0]), "v"(s.b));
            asm volatile("v_cndmask_b32 %0, %1, %2, %3" : "+v"(s.a[0]) : "v"(s.c), "v"(s.b), "s"(s.mask));
        }
        __device__ static void Wide(State &s, float *, const int i) {
            asm volatile("v_cmp_gt_f32 %0, %1, %2" : "=s"(s.mask) : "v"(s.a[i]), "v"(s.b));
            asm volatile("v_cndmask_b32 %0, %1, %2, %3" : "+v"(s.a[i]) : "v"(s.c), "v"(s.b), "s"(s.mask));
        }
        __device__ static float Sink(const State &s) { return s.a[0]; }
    };

    // ------------------------------------------------------ templated kernels

    template <typename Step>
    __global__ void latency_kernel(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[1024];
        for (int index = threadIdx.x; index < 1024; index += blockDim.x) scratch[index] = static_cast<float>(index & 31);
        __syncthreads();
        typename Step::State state;
        Step::Init(state, threadIdx.x);
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int index = 0; index < kChain; ++index) Step::Dependent(state, scratch);
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (Step::Sink(state) == 0.0f) cycles[1] = 1;
    }

    template <typename Step>
    __global__ void throughput_kernel(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[1024];
        for (int index = threadIdx.x; index < 1024; index += blockDim.x) scratch[index] = static_cast<float>(index & 31);
        __syncthreads();
        typename Step::State state;
        Step::Init(state, threadIdx.x);
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int round = 0; round < 4; ++round) {
                #pragma unroll
                for (int index = 0; index < 8; ++index) Step::Wide(state, scratch, index);
            }
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (Step::Sink(state) == 0.0f) cycles[1] = 1;
    }

    // --------------------------------------------------------- LDS latency

    __global__ void lds_b32_latency(const int iterations, unsigned long long *cycles) {
        __shared__ int indices[256];
        for (int index = threadIdx.x; index < 256; index += blockDim.x) indices[index] = ((index + 1) & 255) * 4;
        __syncthreads();
        unsigned address = (threadIdx.x & 31) * 4;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile("ds_read_b32 %0, %0" : "+v"(address));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (address == 0) cycles[1] = 1;
    }

    __global__ void lds_b64_latency(const int iterations, unsigned long long *cycles) {
        __shared__ int indices[256];
        for (int index = threadIdx.x; index < 256; index += blockDim.x) indices[index] = ((index + 2) & 255) * 4;
        __syncthreads();
        unsigned address = (threadIdx.x & 31) * 8;
        unsigned long long pair;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile("ds_read_b64 %0, %1" : "=v"(pair) : "v"(address));
            address = static_cast<unsigned>(pair);
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (address == 0) cycles[1] = 1;
    }

    __global__ void lds_b128_latency(const int iterations, unsigned long long *cycles) {
        __shared__ int indices[256];
        for (int index = threadIdx.x; index < 256; index += blockDim.x) indices[index] = ((index + 4) & 255) * 4;
        __syncthreads();
        unsigned address = (threadIdx.x & 15) * 16;
        v4f32 quad;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile("ds_read_b128 %0, %1" : "=v"(quad) : "v"(address));
            address = static_cast<unsigned>(quad.x);
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (address == 0) cycles[1] = 1;
    }

    // ------------------------------------------------------ LDS throughput

    __global__ void lds_b32_throughput(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[1024];
        for (int index = threadIdx.x; index < 1024; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * 4;
        float v0, v1, v2, v3, v4, v5, v6, v7;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_read_b32 %0, %8 offset:0\n"
                "ds_read_b32 %1, %8 offset:128\n"
                "ds_read_b32 %2, %8 offset:256\n"
                "ds_read_b32 %3, %8 offset:384\n"
                "ds_read_b32 %4, %8 offset:512\n"
                "ds_read_b32 %5, %8 offset:640\n"
                "ds_read_b32 %6, %8 offset:768\n"
                "ds_read_b32 %7, %8 offset:896\n"
                : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3), "=v"(v4), "=v"(v5), "=v"(v6), "=v"(v7)
                : "v"(address));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 == 0.0f) cycles[1] = 1;
    }

    __global__ void lds_b64_throughput(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[1024];
        for (int index = threadIdx.x; index < 1024; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * 8;
        unsigned long long v0, v1, v2, v3, v4, v5, v6, v7;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_read_b64 %0, %8 offset:0\n"
                "ds_read_b64 %1, %8 offset:256\n"
                "ds_read_b64 %2, %8 offset:512\n"
                "ds_read_b64 %3, %8 offset:768\n"
                "ds_read_b64 %4, %8 offset:1024\n"
                "ds_read_b64 %5, %8 offset:1280\n"
                "ds_read_b64 %6, %8 offset:1536\n"
                "ds_read_b64 %7, %8 offset:1792\n"
                : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3), "=v"(v4), "=v"(v5), "=v"(v6), "=v"(v7)
                : "v"(address));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if ((v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7) == 0) cycles[1] = 1;
    }

    __global__ void lds_b128_throughput(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[1024];
        for (int index = threadIdx.x; index < 1024; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * 16;
        v4f32 v0, v1, v2, v3, v4, v5, v6, v7;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_read_b128 %0, %8 offset:0\n"
                "ds_read_b128 %1, %8 offset:512\n"
                "ds_read_b128 %2, %8 offset:1024\n"
                "ds_read_b128 %3, %8 offset:1536\n"
                "ds_read_b128 %4, %8 offset:2048\n"
                "ds_read_b128 %5, %8 offset:2560\n"
                "ds_read_b128 %6, %8 offset:3072\n"
                "ds_read_b128 %7, %8 offset:3584\n"
                : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3), "=v"(v4), "=v"(v5), "=v"(v6), "=v"(v7)
                : "v"(address));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (v0.x + v1.y + v2.z + v3.w + v4.x + v5.y + v6.z + v7.w == 0.0f) cycles[1] = 1;
    }

    __global__ void lds_read2_throughput(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[1024];
        for (int index = threadIdx.x; index < 1024; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * 4;
        unsigned long long v0, v1, v2, v3;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_read2_b32 %0, %4 offset0:0 offset1:16\n"
                "ds_read2_b32 %1, %4 offset0:32 offset1:48\n"
                "ds_read2_b32 %2, %4 offset0:64 offset1:80\n"
                "ds_read2_b32 %3, %4 offset0:96 offset1:112\n"
                : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3)
                : "v"(address));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if ((v0 | v1 | v2 | v3) == 0) cycles[1] = 1;
    }

    template <int Stride>
    __global__ void lds_write_conflict(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[8192];
        for (int index = threadIdx.x; index < 8192; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * Stride * 4;
        const float value = static_cast<float>(threadIdx.x);
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_write_b32 %0, %1 offset:0\n"
                "ds_write_b32 %0, %1 offset:1024\n"
                "ds_write_b32 %0, %1 offset:2048\n"
                "ds_write_b32 %0, %1 offset:3072\n"
                "ds_write_b32 %0, %1 offset:4096\n"
                "ds_write_b32 %0, %1 offset:5120\n"
                "ds_write_b32 %0, %1 offset:6144\n"
                "ds_write_b32 %0, %1 offset:7168\n"
                "s_waitcnt lgkmcnt(0)\n"
                :
                : "v"(address), "v"(value));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (scratch[threadIdx.x] == 0.0f) cycles[1] = 1;
    }

    __global__ void lds_write_throughput(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[1024];
        for (int index = threadIdx.x; index < 1024; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * 4;
        const float value = static_cast<float>(threadIdx.x);
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_write_b32 %0, %1 offset:0\n"
                "ds_write_b32 %0, %1 offset:128\n"
                "ds_write_b32 %0, %1 offset:256\n"
                "ds_write_b32 %0, %1 offset:384\n"
                "ds_write_b32 %0, %1 offset:512\n"
                "ds_write_b32 %0, %1 offset:640\n"
                "ds_write_b32 %0, %1 offset:768\n"
                "ds_write_b32 %0, %1 offset:896\n"
                :
                : "v"(address), "v"(value));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (scratch[threadIdx.x] == 0.0f) cycles[1] = 1;
    }

    __global__ void lds_bpermute_throughput(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[1024];
        for (int index = threadIdx.x; index < 1024; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * 4;
        float v0 = 1.0f, v1 = 2.0f, v2 = 3.0f, v3 = 4.0f;
        float v4 = 5.0f, v5 = 6.0f, v6 = 7.0f, v7 = 8.0f;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_bpermute_b32 %0, %8, %0\n"
                "ds_bpermute_b32 %1, %8, %1\n"
                "ds_bpermute_b32 %2, %8, %2\n"
                "ds_bpermute_b32 %3, %8, %3\n"
                "ds_bpermute_b32 %4, %8, %4\n"
                "ds_bpermute_b32 %5, %8, %5\n"
                "ds_bpermute_b32 %6, %8, %6\n"
                "ds_bpermute_b32 %7, %8, %7\n"
                : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3), "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7)
                : "v"(address));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 == 0.0f) cycles[1] = 1;
    }

    template <int Stride>
    __global__ void lds_conflict(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[8192];
        for (int index = threadIdx.x; index < 8192; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * Stride * 4;
        float v0, v1, v2, v3, v4, v5, v6, v7;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_read_b32 %0, %8 offset:0\n"
                "ds_read_b32 %1, %8 offset:1024\n"
                "ds_read_b32 %2, %8 offset:2048\n"
                "ds_read_b32 %3, %8 offset:3072\n"
                "ds_read_b32 %4, %8 offset:4096\n"
                "ds_read_b32 %5, %8 offset:5120\n"
                "ds_read_b32 %6, %8 offset:6144\n"
                "ds_read_b32 %7, %8 offset:7168\n"
                "s_waitcnt lgkmcnt(0)\n"
                : "=v"(v0), "=v"(v1), "=v"(v2), "=v"(v3), "=v"(v4), "=v"(v5), "=v"(v6), "=v"(v7)
                : "v"(address));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 == 0.0f) cycles[1] = 1;
    }

    // ------------------------------------------------------ global latency

    __global__ void global_chase(const int iterations, const int *buffer, unsigned long long *cycles) {
        int index = 0;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            index = __ldg(buffer + index);
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (index == -1) cycles[1] = 1;
    }

    __global__ void fill_chain(int *buffer, const int size, const int stride) {
        for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x) {
            buffer[index] = (index + stride < size) ? index + stride : index + stride - size;
        }
    }

    // ------------------------------------------------------ mixed issue

    __global__ void mixed_kernel(const int iterations, unsigned long long *cycles) {
        __shared__ float scratch[512];
        for (int index = threadIdx.x; index < 512; index += blockDim.x) scratch[index] = static_cast<float>(index);
        __syncthreads();
        const unsigned address = threadIdx.x * 4;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        float v4 = 0.0f, v5 = 0.0f, v6 = 0.0f, v7 = 0.0f;
        const float b = 1.0001f;
        const float c = 0.5f;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            asm volatile(
                "ds_read_b32 %0, %11 offset:0\n"
                "ds_read_b32 %1, %11 offset:64\n"
                "ds_read_b32 %2, %11 offset:128\n"
                "ds_read_b32 %3, %11 offset:192\n"
                "ds_read_b32 %4, %11 offset:256\n"
                "ds_read_b32 %5, %11 offset:320\n"
                "ds_read_b32 %6, %11 offset:384\n"
                "ds_read_b32 %7, %11 offset:448\n"
                "s_waitcnt lgkmcnt(0)\n"
                "v_fma_f32 %0, %0, %12, %13\n"
                "v_fma_f32 %0, %0, %12, %13\n"
                "v_fma_f32 %1, %1, %12, %13\n"
                "v_fma_f32 %1, %1, %12, %13\n"
                "v_fma_f32 %2, %2, %12, %13\n"
                "v_fma_f32 %2, %2, %12, %13\n"
                "v_fma_f32 %3, %3, %12, %13\n"
                "v_fma_f32 %3, %3, %12, %13\n"
                "v_fma_f32 %4, %4, %12, %13\n"
                "v_fma_f32 %4, %4, %12, %13\n"
                "v_fma_f32 %5, %5, %12, %13\n"
                "v_fma_f32 %5, %5, %12, %13\n"
                "v_fma_f32 %6, %6, %12, %13\n"
                "v_fma_f32 %6, %6, %12, %13\n"
                "v_fma_f32 %7, %7, %12, %13\n"
                "v_fma_f32 %7, %7, %12, %13\n"
                : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3), "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7)
                : "v"(address), "v"(b), "v"(c));
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 == 0.0f) cycles[1] = 1;
    }

    __global__ void salu_valu_mix(const int iterations, unsigned long long *cycles) {
        int scalar = blockIdx.x + 1;
        float accumulators[8];
        for (int index = 0; index < 8; ++index) accumulators[index] = 1.0f + index;
        const float b = 1.0001f;
        const float c = 0.5f;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int round = 0; round < 4; ++round) {
                #pragma unroll
                for (int index = 0; index < 8; ++index) {
                    asm volatile("s_add_u32 %0, %0, 1" : "+s"(scalar) : : "scc");
                }
                #pragma unroll
                for (int index = 0; index < 8; ++index) {
                    asm volatile("v_fma_f32 %0, %0, %1, %2" : "+v"(accumulators[index]) : "v"(b), "v"(c));
                }
            }
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (accumulators[0] == 0.0f || scalar == 0) cycles[1] = 1;
    }

    // --------------------------------------------------------- measurement

    using Kernel = void (*)(int, unsigned long long *);

    double Measure(const Kernel kernel, const int iterations, const int operationsPerIteration, const int threads = kThreads) {
        unsigned long long *device = nullptr;
        if (hipMalloc(&device, 2 * sizeof(unsigned long long)) != hipSuccess) {
            std::printf("hipMalloc failed\n");
            std::exit(1);
        }

        double best = std::numeric_limits<double>::max();
        for (int attempt = 0; attempt < 4; ++attempt) {
            void *arguments[] = {const_cast<int *>(&iterations), &device};
            if (hipLaunchKernel(reinterpret_cast<const void *>(kernel), dim3(1), dim3(threads), arguments, 0, nullptr) != hipSuccess) {
                std::printf("launch failed\n");
                std::exit(1);
            }
            if (hipDeviceSynchronize() != hipSuccess) {
                std::printf("sync failed\n");
                std::exit(1);
            }
            unsigned long long host[2] = {};
            if (hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost) != hipSuccess) {
                std::printf("copy failed\n");
                std::exit(1);
            }
            if (attempt > 0) best = std::min(best, static_cast<double>(host[0]));
        }
        (void) hipFree(device);
        return best / (static_cast<double>(iterations) * operationsPerIteration);
    }

    template <typename Step>
    void ReportLatency(const char *name, const int instructionsPerStep = 1) {
        const double cycles = Measure(latency_kernel<Step>, 200000, kChain * instructionsPerStep);
        std::printf("%-34s %7.2f cycles\n", name, cycles);
    }

    template <typename Step>
    void ReportThroughput(const char *name, const int instructionsPerStep = 1) {
        const double cycles = Measure(throughput_kernel<Step>, 200000, 32 * instructionsPerStep);
        std::printf("%-34s %7.2f cycles\n", name, cycles);
    }

    void ReportKernel(const char *name, const Kernel kernel, const int operationsPerIteration, const int iterations = 200000) {
        const double cycles = Measure(kernel, iterations, operationsPerIteration);
        std::printf("%-34s %7.2f cycles\n", name, cycles);
    }

    double MeasureChase(const int *buffer, const int iterations) {
        unsigned long long *device = nullptr;
        if (hipMalloc(&device, 2 * sizeof(unsigned long long)) != hipSuccess) {
            std::printf("hipMalloc failed\n");
            std::exit(1);
        }
        const int *pointer = buffer;
        double best = std::numeric_limits<double>::max();
        for (int attempt = 0; attempt < 4; ++attempt) {
            void *arguments[] = {const_cast<int *>(&iterations), &pointer, &device};
            if (hipLaunchKernel(reinterpret_cast<const void *>(global_chase), dim3(1), dim3(kThreads), arguments, 0, nullptr) != hipSuccess) {
                std::printf("launch failed\n");
                std::exit(1);
            }
            if (hipDeviceSynchronize() != hipSuccess) {
                std::printf("sync failed\n");
                std::exit(1);
            }
            unsigned long long host[2] = {};
            if (hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost) != hipSuccess) {
                std::printf("copy failed\n");
                std::exit(1);
            }
            if (attempt > 0) best = std::min(best, static_cast<double>(host[0]));
        }
        (void) hipFree(device);
        return best / static_cast<double>(iterations);
    }

    void ReportChase(const char *name, const int kilobytes, const int iterations) {
        const int size = kilobytes * 1024 / static_cast<int>(sizeof(int));
        int *buffer = nullptr;
        if (hipMalloc(&buffer, static_cast<size_t>(kilobytes) * 1024) != hipSuccess) {
            std::printf("%s allocation failed\n", name);
            return;
        }
        fill_chain<<<1024, 256>>>(buffer, size, 32);
        if (hipDeviceSynchronize() != hipSuccess) {
            std::printf("%s fill failed\n", name);
            (void) hipFree(buffer);
            return;
        }
        const double cycles = MeasureChase(buffer, iterations);
        std::printf("%-34s %7.2f cycles\n", name, cycles);
        (void) hipFree(buffer);
    }

    void CalibrateClock() {
        const int iterations = 200000;
        unsigned long long *device = nullptr;
        if (hipMalloc(&device, 2 * sizeof(unsigned long long)) != hipSuccess) return;
        void *arguments[] = {const_cast<int *>(&iterations), &device};

        (void) hipLaunchKernel(reinterpret_cast<const void *>(latency_kernel<FmaStep>), dim3(1), dim3(kThreads), arguments, 0, nullptr);
        (void) hipDeviceSynchronize();
        const auto start = std::chrono::steady_clock::now();
        constexpr int repeats = 10;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            (void) hipLaunchKernel(reinterpret_cast<const void *>(latency_kernel<FmaStep>), dim3(1), dim3(kThreads), arguments, 0, nullptr);
        }
        (void) hipDeviceSynchronize();
        const auto stop = std::chrono::steady_clock::now();
        unsigned long long host[2] = {};
        (void) hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost);
        const double seconds = std::chrono::duration<double>(stop - start).count() / repeats;
        std::printf("%-34s %7.2f GHz\n", "s_memtime calibration", static_cast<double>(host[0]) / seconds / 1.0e9);
        (void) hipFree(device);
    }
} // namespace

int main() {
    std::printf("RDNA2 instruction microbenchmark (gfx1031)\n");
    std::printf("one wave, s_memtime cycles; latency = dependent chain, throughput = 32 independent ops\n\n");

    std::printf("-- VALU and MUFU --\n");
    ReportLatency<FmaStep>("v_fma_f32 latency");
    ReportThroughput<FmaStep>("v_fma_f32 throughput");
    ReportLatency<FmacStep>("v_fmac_f32 latency");
    ReportThroughput<FmacStep>("v_fmac_f32 throughput");
    ReportLatency<AddStep>("v_add_f32 latency");
    ReportThroughput<AddStep>("v_add_f32 throughput");
    ReportLatency<MulStep>("v_mul_f32 latency");
    ReportThroughput<MulStep>("v_mul_f32 throughput");
    ReportThroughput<MovStep>("v_mov_b32 throughput");
    ReportLatency<ExpStep>("v_exp_f32 latency");
    ReportThroughput<ExpStep>("v_exp_f32 throughput");
    ReportLatency<LogStep>("v_log_f32 latency");
    ReportThroughput<LogStep>("v_log_f32 throughput");
    ReportLatency<RcpStep>("v_rcp_f32 latency");
    ReportThroughput<RcpStep>("v_rcp_f32 throughput");
    ReportLatency<SqrtStep>("v_sqrt_f32 latency");
    ReportThroughput<SqrtStep>("v_sqrt_f32 throughput");
    ReportLatency<RsqStep>("v_rsq_f32 latency");
    ReportThroughput<RsqStep>("v_rsq_f32 throughput");

    std::printf("\n-- dot products and packed fp16 --\n");
    ReportLatency<Dot2Step>("v_dot2_f32_f16 latency");
    ReportThroughput<Dot2Step>("v_dot2_f32_f16 throughput");
    ReportLatency<Dot4Step>("v_dot4c_i32_i8 latency");
    ReportThroughput<Dot4Step>("v_dot4c_i32_i8 throughput");
    ReportLatency<PkFmaStep>("v_pk_fma_f16 latency");
    ReportThroughput<PkFmaStep>("v_pk_fma_f16 throughput");

    std::printf("\n-- integer and select --\n");
    ReportLatency<AddU32Step>("v_add_nc_u32 latency");
    ReportThroughput<AddU32Step>("v_add_nc_u32 throughput");
    ReportLatency<MulLoU32Step>("v_mul_lo_u32 latency");
    ReportThroughput<MulLoU32Step>("v_mul_lo_u32 throughput");
    ReportLatency<LshlStep>("v_lshlrev_b32 latency");
    ReportThroughput<LshlStep>("v_lshlrev_b32 throughput");
    ReportLatency<MadU24Step>("v_mad_u32_u24 latency");
    ReportThroughput<MadU24Step>("v_mad_u32_u24 throughput");
    ReportLatency<BfeStep>("v_bfe_u32 latency");
    ReportThroughput<BfeStep>("v_bfe_u32 throughput");
    ReportLatency<SaluStep>("s_add_u32 latency");
    ReportThroughput<SaluStep>("s_add_u32 throughput");
    ReportLatency<CmpSelectStep>("v_cmp_gt_f32 + v_cndmask pair", 2);
    ReportThroughput<CmpSelectStep>("v_cmp_gt_f32 + v_cndmask throughput", 2);

    std::printf("\n-- LDS --\n");
    ReportKernel("ds_read_b32 dependent chain", lds_b32_latency, 1);
    ReportKernel("ds_read_b64 dependent chain", lds_b64_latency, 1);
    ReportKernel("ds_read_b128 dependent chain", lds_b128_latency, 1);
    ReportKernel("ds_read_b32 throughput", lds_b32_throughput, 8);
    ReportKernel("ds_read_b64 throughput", lds_b64_throughput, 8);
    ReportKernel("ds_read_b128 throughput", lds_b128_throughput, 8);
    ReportKernel("ds_read2_b32 throughput", lds_read2_throughput, 8);
    ReportKernel("ds_write_b32 throughput", lds_write_throughput, 8);
    ReportKernel("ds_bpermute_b32 throughput", lds_bpermute_throughput, 8);
    ReportKernel("ds_read_b32 stride 1", lds_conflict<1>, 8);
    ReportKernel("ds_read_b32 stride 2", lds_conflict<2>, 8);
    ReportKernel("ds_read_b32 stride 4", lds_conflict<4>, 8);
    ReportKernel("ds_read_b32 stride 8", lds_conflict<8>, 8);
    ReportKernel("ds_read_b32 stride 16", lds_conflict<16>, 8);
    ReportKernel("ds_read_b32 stride 32", lds_conflict<32>, 8);
    ReportKernel("ds_write_b32 stride 1", lds_write_conflict<1>, 8);
    ReportKernel("ds_write_b32 stride 32", lds_write_conflict<32>, 8);

    std::printf("\n-- global memory (pointer chase, one 128-byte line per step) --\n");
    ReportChase("L1 chase (4 KB)", 4, 1000000);
    ReportChase("L2 chase (1 MB)", 1024, 300000);
    ReportChase("L3 chase (8 MB)", 8192, 300000);
    ReportChase("L3 chase (16 MB)", 16384, 250000);
    ReportChase("infinity cache (32 MB)", 32768, 200000);
    ReportChase("DRAM (256 MB)", 262144, 100000);

    std::printf("\n-- mixed issue --\n");
    ReportKernel("8 LDS + 16 FMA", mixed_kernel, 24);
    ReportKernel("32 FMA + 32 SALU", salu_valu_mix, 32);

    std::printf("\n-- wave scaling (v_fma_f32 issue rate) --\n");
    std::printf("%-34s %7.2f cycles\n", "1 wave", Measure(throughput_kernel<FmaStep>, 200000, 32, 32));
    std::printf("%-34s %7.2f cycles\n", "2 waves", Measure(throughput_kernel<FmaStep>, 200000, 32, 64));
    std::printf("%-34s %7.2f cycles\n", "4 waves", Measure(throughput_kernel<FmaStep>, 200000, 32, 128));

    std::printf("\n");
    CalibrateClock();
    return 0;
}








