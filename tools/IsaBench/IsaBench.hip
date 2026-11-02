//
// RDNA2 (gfx1031) instruction microbenchmarks. Each kernel runs a dependent chain
// (latency) or independent chains (throughput) and reports cycles per instruction
// from s_memtime. One wave on one CU keeps the measurement isolated.
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

    __device__ __forceinline__ void fma_dependent(float &accumulator, const float b, const float c) {
        asm volatile("v_fma_f32 %0, %0, %1, %2" : "+v"(accumulator) : "v"(b), "v"(c));
    }

    __device__ __forceinline__ void dot2_dependent(float &accumulator, const unsigned value) {
        asm volatile("v_dot2_f32_f16 %0, %1, %1, %0" : "+v"(accumulator) : "v"(value));
    }

    __device__ __forceinline__ void dot4_dependent(int &accumulator, const unsigned value) {
        asm volatile("v_dot4c_i32_i8 %0, %1, %1" : "+v"(accumulator) : "v"(value));
    }

    __device__ __forceinline__ void exp_dependent(float &accumulator) {
        asm volatile("v_exp_f32 %0, %0" : "+v"(accumulator));
    }

    __global__ void fma_latency(const int iterations, unsigned long long *cycles) {
        float accumulator = 1.0f + threadIdx.x * 1.0e-3f;
        const float b = 1.0001f;
        const float c = 0.5f;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int index = 0; index < kChain; ++index) fma_dependent(accumulator, b, c);
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (accumulator == 0.0f) cycles[1] = 1;
    }

    __global__ void fma_throughput(const int iterations, unsigned long long *cycles) {
        float accumulators[8];
        #pragma unroll
        for (int index = 0; index < 8; ++index) accumulators[index] = 1.0f + index;
        const float b = 1.0001f;
        const float c = 0.5f;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int round = 0; round < 4; ++round) {
                #pragma unroll
                for (int index = 0; index < 8; ++index) fma_dependent(accumulators[index], b, c);
            }
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (accumulators[0] == 0.0f) cycles[1] = 1;
    }

    __global__ void dot2_latency(const int iterations, unsigned long long *cycles) {
        float accumulator = 0.0f;
        const unsigned value = 0x3c003c00u;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int index = 0; index < kChain; ++index) dot2_dependent(accumulator, value);
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (accumulator == 0.0f) cycles[1] = 1;
    }

    __global__ void dot2_throughput(const int iterations, unsigned long long *cycles) {
        float accumulators[8] = {};
        const unsigned value = 0x3c003c00u;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int round = 0; round < 4; ++round) {
                #pragma unroll
                for (int index = 0; index < 8; ++index) dot2_dependent(accumulators[index], value);
            }
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (accumulators[0] == 0.0f) cycles[1] = 1;
    }

    __global__ void dot4_latency(const int iterations, unsigned long long *cycles) {
        int accumulator = 0;
        const unsigned value = 0x01010101u;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int index = 0; index < kChain; ++index) dot4_dependent(accumulator, value);
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (accumulator == 0) cycles[1] = 1;
    }

    __global__ void dot4_throughput(const int iterations, unsigned long long *cycles) {
        int accumulators[8] = {};
        const unsigned value = 0x01010101u;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int round = 0; round < 4; ++round) {
                #pragma unroll
                for (int index = 0; index < 8; ++index) dot4_dependent(accumulators[index], value);
            }
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (accumulators[0] == 0) cycles[1] = 1;
    }

    __global__ void exp_latency(const int iterations, unsigned long long *cycles) {
        float accumulator = 1.0f;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int index = 0; index < kChain; ++index) exp_dependent(accumulator);
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (accumulator == 0.0f) cycles[1] = 1;
    }

    __global__ void lds_latency(const int iterations, unsigned long long *cycles) {
        __shared__ float memory[64];
        for (int index = threadIdx.x; index < 64; index += blockDim.x) memory[index] = static_cast<float>(index);
        __syncthreads();
        int index = threadIdx.x & 31;
        float value = 0.0f;
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            value = memory[index];
            index = static_cast<int>(value) & 31;
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (value == 0.0f) cycles[1] = 1;
    }

    __global__ void lds_throughput(const int iterations, unsigned long long *cycles) {
        __shared__ float memory[256];
        for (int index = threadIdx.x; index < 256; index += blockDim.x) memory[index] = static_cast<float>(index);
        __syncthreads();
        float values[8] = {};
        const unsigned long long start = clock64();
        #pragma unroll 1
        for (int iteration = 0; iteration < iterations; ++iteration) {
            #pragma unroll
            for (int round = 0; round < 2; ++round) {
                #pragma unroll
                for (int index = 0; index < 8; ++index) values[index] += memory[threadIdx.x + index * 32];
            }
        }
        const unsigned long long stop = clock64();
        if (threadIdx.x == 0) cycles[0] = stop - start;
        if (values[0] == 0.0f) cycles[1] = 1;
    }

    using Kernel = void (*)(int, unsigned long long *);

    double Measure(const Kernel kernel, const int iterations, const int instructionsPerIteration) {
        unsigned long long *device = nullptr;
        if (hipMalloc(&device, 2 * sizeof(unsigned long long)) != hipSuccess) {
            std::printf("hipMalloc failed\n");
            std::exit(1);
        }

        double best = std::numeric_limits<double>::max();
        for (int attempt = 0; attempt < 4; ++attempt) {
            void *arguments[] = {const_cast<int *>(&iterations), &device};
            if (hipLaunchKernel(reinterpret_cast<const void *>(kernel), dim3(1), dim3(kThreads), arguments, 0, nullptr) != hipSuccess) {
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
        return best / (static_cast<double>(iterations) * instructionsPerIteration);
    }

    void Report(const char *name, const Kernel kernel, const int instructionsPerIteration) {
        const int iterations = 200000;
        const double cycles = Measure(kernel, iterations, instructionsPerIteration);
        std::printf("%-28s %8.2f cycles/instruction\n", name, cycles);
    }

    void CalibrateClock() {
        const int iterations = 200000;
        unsigned long long *device = nullptr;
        if (hipMalloc(&device, 2 * sizeof(unsigned long long)) != hipSuccess) return;
        void *arguments[] = {const_cast<int *>(&iterations), &device};

        (void) hipLaunchKernel(reinterpret_cast<const void *>(fma_latency), dim3(1), dim3(kThreads), arguments, 0, nullptr);
        (void) hipDeviceSynchronize();
        const auto start = std::chrono::steady_clock::now();
        constexpr int repeats = 10;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            (void) hipLaunchKernel(reinterpret_cast<const void *>(fma_latency), dim3(1), dim3(kThreads), arguments, 0, nullptr);
        }
        (void) hipDeviceSynchronize();
        const auto stop = std::chrono::steady_clock::now();
        unsigned long long host[2] = {};
        (void) hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost);
        const double seconds = std::chrono::duration<double>(stop - start).count() / repeats;
        std::printf("s_memtime calibration      %8.2f GHz\n", static_cast<double>(host[0]) / seconds / 1.0e9);
        (void) hipFree(device);
    }
} // namespace

int main() {
    std::printf("RDNA2 instruction microbenchmark (gfx1031, one wave, s_memtime cycles)\n");
    std::printf("%-28s %s\n", "instruction", "cost");

    Report("v_fma_f32 latency", fma_latency, kChain);
    Report("v_fma_f32 throughput", fma_throughput, 32);
    Report("v_dot2_f32_f16 latency", dot2_latency, kChain);
    Report("v_dot2_f32_f16 throughput", dot2_throughput, 32);
    Report("v_dot4c_i32_i8 latency", dot4_latency, kChain);
    Report("v_dot4c_i32_i8 throughput", dot4_throughput, 32);
    Report("v_exp_f32 latency", exp_latency, kChain);
    Report("ds_read_b32 latency", lds_latency, 1);
    Report("ds_read_b32 throughput", lds_throughput, 16);
    CalibrateClock();
    return 0;
}


