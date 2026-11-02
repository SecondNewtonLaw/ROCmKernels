#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/AdamW.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kTensorCount = 4;
    constexpr std::int64_t kElementsPerTensor = 2'000'000;
    constexpr std::uint64_t kTotalElements = static_cast<std::uint64_t>(kTensorCount) * kElementsPerTensor;

    template<typename T> const char *TypeName() {
        if constexpr (std::is_same_v<T, __half>) return "fp16";
        else return "fp32";
    }

    template<typename T> std::vector<T> ToStorage(const std::vector<float> &values) {
        std::vector<T> storage(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            if constexpr (std::is_same_v<T, __half>) storage[index] = __float2half_rn(values[index]);
            else storage[index] = values[index];
        }
        return storage;
    }

    template<typename T> struct State {
        void Initialize() {
            const std::size_t totalElements = static_cast<std::size_t>(kTotalElements);

            this->parameter.Resize(totalElements);
            this->gradient.Resize(totalElements);
            this->first.Resize(totalElements);
            this->second.Resize(totalElements);

            const std::vector<float> hostParameter(totalElements, 0.5f);
            const std::vector<float> hostGradient(totalElements, 0.25f);
            const std::vector<float> hostFirst(totalElements, 0.5f);
            const std::vector<float> hostSecond(totalElements, 0.25f);
            const std::vector<T> storageParameter = ToStorage<T>(hostParameter);
            const std::vector<T> storageGradient = ToStorage<T>(hostGradient);

            this->parameter.CopyFromHost(storageParameter.data(), storageParameter.size());
            this->gradient.CopyFromHost(storageGradient.data(), storageGradient.size());
            this->first.CopyFromHost(hostFirst.data(), hostFirst.size());
            this->second.CopyFromHost(hostSecond.data(), hostSecond.size());

            this->tensors.resize(kTensorCount);
            for (int tensor = 0; tensor < kTensorCount; ++tensor) {
                const std::size_t offset = static_cast<std::size_t>(tensor) * kElementsPerTensor;
                this->tensors[tensor] = ROCmKernels::Kernels::AdamWTensor{
                    this->parameter.Data() + offset, this->gradient.Data() + offset, this->first.Data() + offset, this->second.Data() + offset,
                    kElementsPerTensor
                };
            }
        }

        std::vector<ROCmKernels::Kernels::AdamWTensor> tensors;
        ROCmKernels::DeviceBuffer<T> parameter;
        ROCmKernels::DeviceBuffer<T> gradient;
        ROCmKernels::DeviceBuffer<float> first;
        ROCmKernels::DeviceBuffer<float> second;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeCase() {
        const auto state = std::make_shared<State<T>>();
        const std::uint64_t bytesMoved = kTotalElements * (static_cast<std::uint64_t>(sizeof(T)) * 3 + sizeof(float) * 4);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("AdamW ") + TypeName<T>() + " 4 tensors 8M elements",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::AdamWMulti(state->tensors.data(), kTensorCount, 7, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f,
                            ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = bytesMoved,
            .operations = kTotalElements * 12,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeCase<float>());
            ROCmKernels::Benchmark::Register(MakeCase<__half>());
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
