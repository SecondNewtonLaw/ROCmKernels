#include <ROCmKernels/Kernels/Generic/AdamW.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {
    constexpr int kThreads = 256;
    constexpr int kMaxBlocks = 4096;

    struct DeviceTensor {
        void *parameter;
        const void *gradient;
        float *firstMoment;
        float *secondMoment;
        long long begin;
        long long end;
    };

    template<typename T>
    __global__ void adamw_kernel(const DeviceTensor *tensors, const int tensorCount, const long long elements, const float learningRate,
            const float beta1, const float beta2, const float decay, const float firstCorrection, const float secondCorrection,
            const float epsilon) {
        const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; index < elements; index += stride) {
            // Offsets are contiguous and ordered, so the owning tensor is the last one whose begin is <= index.
            int low = 0;
            int high = tensorCount;
            while (low + 1 < high) {
                const int middle = (low + high) / 2;
                if (index < tensors[middle].begin) high = middle;
                else low = middle;
            }

            const DeviceTensor &tensor = tensors[low];
            const long long local = index - tensor.begin;
            T *parameter = static_cast<T *>(tensor.parameter);
            const T *gradient = static_cast<const T *>(tensor.gradient);

            const float gradientValue = ROCmKernels::Kernels::LoadScalar(gradient + local);
            const float first = beta1 * tensor.firstMoment[local] + (1.0f - beta1) * gradientValue;
            const float second = beta2 * tensor.secondMoment[local] + (1.0f - beta2) * gradientValue * gradientValue;
            tensor.firstMoment[local] = first;
            tensor.secondMoment[local] = second;

            const float firstHat = first / firstCorrection;
            const float secondHat = second / secondCorrection;
            const float value = ROCmKernels::Kernels::LoadScalar(parameter + local);
            const float decayed = value - decay * value;
            const float updated = decayed - learningRate * firstHat / (sqrtf(secondHat) + epsilon);
            ROCmKernels::Kernels::StoreScalar(parameter + local, updated);
        }
    }

    template<typename T>
    hipError_t launch(const DeviceTensor *tensors, const int tensorCount, const long long elements, const float learningRate, const float beta1,
            const float beta2, const float decay, const float firstCorrection, const float secondCorrection, const float epsilon,
            const hipStream_t stream) {
        const long long requestedBlocks = (elements + kThreads - 1) / kThreads;
        const unsigned int blocks = static_cast<unsigned int>(requestedBlocks < kMaxBlocks ? requestedBlocks : kMaxBlocks);
        hipLaunchKernelGGL((adamw_kernel<T>), dim3(blocks), dim3(kThreads), 0, stream, tensors, tensorCount, elements, learningRate, beta1,
                beta2, decay, firstCorrection, secondCorrection, epsilon);
        return hipGetLastError();
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

    std::vector<DeviceTensor> descriptors;
    descriptors.reserve(static_cast<std::size_t>(tensorCount));

    long long elements = 0;
    for (int index = 0; index < tensorCount; ++index) {
        const AdamWTensor &tensor = tensors[index];
        if (tensor.elements <= 0) continue;
        if (!tensor.parameter || !tensor.gradient || !tensor.firstMoment || !tensor.secondMoment) return hipErrorInvalidValue;

        // The offsets passed to the kernel are signed 64-bit, so the fused total must fit as well.
        if (tensor.elements > std::numeric_limits<long long>::max() - elements) return hipErrorInvalidValue;
        descriptors.push_back(DeviceTensor{tensor.parameter, tensor.gradient, tensor.firstMoment, tensor.secondMoment, elements,
                elements + tensor.elements});
        elements += tensor.elements;
    }

    if (descriptors.empty()) return hipSuccess;

    const float firstCorrection = static_cast<float>(1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(step)));
    const float secondCorrection = static_cast<float>(1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(step)));
    const float decay = learningRate * weightDecay;

    DeviceTensor *deviceTensors = nullptr;
    hipError_t status = hipMalloc(reinterpret_cast<void **>(&deviceTensors), descriptors.size() * sizeof(DeviceTensor));
    if (status != hipSuccess) return status;

    status = hipMemcpy(deviceTensors, descriptors.data(), descriptors.size() * sizeof(DeviceTensor), hipMemcpyHostToDevice);
    if (status == hipSuccess) {
        switch (dtype) {
            case DType::Float32:
                status = launch<float>(deviceTensors, static_cast<int>(descriptors.size()), elements, learningRate, beta1, beta2, decay,
                        firstCorrection, secondCorrection, epsilon, stream);
                break;
            case DType::Float16:
                status = launch<__half>(deviceTensors, static_cast<int>(descriptors.size()), elements, learningRate, beta1, beta2, decay,
                        firstCorrection, secondCorrection, epsilon, stream);
                break;
            default:
                status = hipErrorInvalidValue;
                break;
        }
    }

    // hipFree waits for the launch, so the descriptor array cannot be released before the kernel reads it.
    const hipError_t freeStatus = hipFree(deviceTensors);
    return static_cast<int>(status != hipSuccess ? status : freeStatus);
}
