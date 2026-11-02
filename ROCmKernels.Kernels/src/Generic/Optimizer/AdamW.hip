#include <ROCmKernels/Kernels/Generic/AdamW.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {
    constexpr int kThreads = 256;
    constexpr int kMaxBlocks = 4096;
    constexpr int kVector = 4;
    // Each chunk lies inside one tensor, so a block never searches for its tensor or straddles a boundary.
    constexpr long long kChunk = kThreads * kVector * 4;

    struct DeviceTensor {
        void *parameter;
        const void *gradient;
        float *firstMoment;
        float *secondMoment;
        long long elements;
        int vectorized;
    };

    // Passed by value so a step needs no device allocation or host copy; more tensors take several launches.
    constexpr int kBatch = 32;
    struct Batch {
        DeviceTensor tensors[kBatch];
        int firstChunk[kBatch + 1];
        int count;
    };

    struct Step {
        float learningRate;
        float beta1;
        float beta2;
        float decay;
        float firstCorrection;
        float secondCorrection;
        float epsilon;
    };

    __device__ __forceinline__ float update(float &first, float &second, const float gradient, const float value, const Step &step) {
        first = step.beta1 * first + (1.0f - step.beta1) * gradient;
        second = step.beta2 * second + (1.0f - step.beta2) * gradient * gradient;
        const float firstHat = first / step.firstCorrection;
        const float secondHat = second / step.secondCorrection;
        return value - step.decay * value - step.learningRate * firstHat / (sqrtf(secondHat) + step.epsilon);
    }

    template<typename T>
    __device__ __forceinline__ float4 load4(const T *pointer) {
        if constexpr (std::is_same_v<T, __half>) {
            const float2 low = ROCmKernels::Kernels::LoadPair(pointer);
            const float2 high = ROCmKernels::Kernels::LoadPair(pointer + 2);
            return make_float4(low.x, low.y, high.x, high.y);
        } else {
            return *reinterpret_cast<const float4 *>(pointer);
        }
    }

    template<typename T>
    __device__ __forceinline__ void store4(T *pointer, const float4 value) {
        if constexpr (std::is_same_v<T, __half>) {
            ROCmKernels::Kernels::StorePair(pointer, make_float2(value.x, value.y));
            ROCmKernels::Kernels::StorePair(pointer + 2, make_float2(value.z, value.w));
        } else {
            *reinterpret_cast<float4 *>(pointer) = value;
        }
    }

    template<typename T>
    __global__ __launch_bounds__(kThreads) void adamw_kernel(const Batch batch, const Step step) {
        for (int chunkIndex = blockIdx.x; chunkIndex < batch.firstChunk[batch.count]; chunkIndex += gridDim.x) {
            int owner = 0;
            while (batch.firstChunk[owner + 1] <= chunkIndex) ++owner;
            const DeviceTensor &tensor = batch.tensors[owner];
            T *parameter = static_cast<T *>(tensor.parameter);
            const T *gradient = static_cast<const T *>(tensor.gradient);
            const long long begin = static_cast<long long>(chunkIndex - batch.firstChunk[owner]) * kChunk;
            const long long end = begin + kChunk < tensor.elements ? begin + kChunk : tensor.elements;

            long long scalarBegin = begin;
            if (tensor.vectorized) {
                scalarBegin = begin + (end - begin) / kVector * kVector;
                for (long long index = begin + threadIdx.x * kVector; index < scalarBegin; index += kThreads * kVector) {
                    const float4 g = load4(gradient + index);
                    const float4 value = load4(parameter + index);
                    float4 first = *reinterpret_cast<const float4 *>(tensor.firstMoment + index);
                    float4 second = *reinterpret_cast<const float4 *>(tensor.secondMoment + index);
                    float4 updated;
                    updated.x = update(first.x, second.x, g.x, value.x, step);
                    updated.y = update(first.y, second.y, g.y, value.y, step);
                    updated.z = update(first.z, second.z, g.z, value.z, step);
                    updated.w = update(first.w, second.w, g.w, value.w, step);
                    *reinterpret_cast<float4 *>(tensor.firstMoment + index) = first;
                    *reinterpret_cast<float4 *>(tensor.secondMoment + index) = second;
                    store4(parameter + index, updated);
                }
            }

            for (long long index = scalarBegin + threadIdx.x; index < end; index += kThreads) {
                float first = tensor.firstMoment[index];
                float second = tensor.secondMoment[index];
                const float updated = update(first, second, ROCmKernels::Kernels::LoadScalar(gradient + index),
                        ROCmKernels::Kernels::LoadScalar(parameter + index), step);
                tensor.firstMoment[index] = first;
                tensor.secondMoment[index] = second;
                ROCmKernels::Kernels::StoreScalar(parameter + index, updated);
            }
        }
    }
} // namespace

extern "C" ROCMKERNELS_EXPORT int adamw_multi(const ROCmKernels::Kernels::AdamWTensor *tensors, int tensorCount, long long step,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, ROCmKernels::Kernels::DType dtype,
        hipStream_t stream) {
    using ROCmKernels::Kernels::AdamWTensor;
    using ROCmKernels::Kernels::DType;

    if (!tensors || tensorCount <= 0 || step < 1 || !(learningRate >= 0.0f) || !(beta1 >= 0.0f && beta1 < 1.0f) ||
        !(beta2 >= 0.0f && beta2 < 1.0f) || !(epsilon > 0.0f) || !(weightDecay >= 0.0f) ||
        (dtype != DType::Float32 && dtype != DType::Float16))
        return hipErrorInvalidValue;

    const std::size_t elementBytes = dtype == DType::Float16 ? sizeof(__half) : sizeof(float);
    const auto aligned = [](const void *pointer, const std::size_t bytes) { return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0; };

    for (int index = 0; index < tensorCount; ++index) {
        const AdamWTensor &tensor = tensors[index];
        if (tensor.elements <= 0) continue;
        if (!tensor.parameter || !tensor.gradient || !tensor.firstMoment || !tensor.secondMoment) return hipErrorInvalidValue;
    }

    const Step parameters{learningRate, beta1, beta2, learningRate * weightDecay,
            static_cast<float>(1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(step))),
            static_cast<float>(1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(step))), epsilon};

    Batch batch{};
    const auto flush = [&]() {
        if (batch.count == 0) return hipSuccess;
        const int chunks = batch.firstChunk[batch.count];
        const dim3 blocks(chunks < kMaxBlocks ? chunks : kMaxBlocks);
        if (dtype == DType::Float32) hipLaunchKernelGGL((adamw_kernel<float>), blocks, dim3(kThreads), 0, stream, batch, parameters);
        else hipLaunchKernelGGL((adamw_kernel<__half>), blocks, dim3(kThreads), 0, stream, batch, parameters);
        batch.count = 0;
        return hipGetLastError();
    };

    for (int index = 0; index < tensorCount; ++index) {
        const AdamWTensor &tensor = tensors[index];
        if (tensor.elements <= 0) continue;

        const long long chunks = (tensor.elements + kChunk - 1) / kChunk;
        if (chunks > std::numeric_limits<int>::max() - batch.firstChunk[batch.count]) {
            const hipError_t status = flush();
            if (status != hipSuccess) return status;
            if (chunks > std::numeric_limits<int>::max()) return hipErrorInvalidValue;
        }

        const bool vectorized = aligned(tensor.parameter, kVector * elementBytes) && aligned(tensor.gradient, kVector * elementBytes) &&
                                aligned(tensor.firstMoment, kVector * sizeof(float)) && aligned(tensor.secondMoment, kVector * sizeof(float));
        batch.tensors[batch.count] = DeviceTensor{tensor.parameter, tensor.gradient, tensor.firstMoment, tensor.secondMoment,
                tensor.elements, vectorized ? 1 : 0};
        batch.firstChunk[batch.count + 1] = batch.firstChunk[batch.count] + static_cast<int>(chunks);
        ++batch.count;

        if (batch.count == kBatch) {
            const hipError_t status = flush();
            if (status != hipSuccess) return status;
        }
    }
    return flush();
}
