#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/LinearActVec.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kColumns = 4096;
    constexpr int kInner = 4096;

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

    template<typename T> struct LinearActVecState {
        void Initialize() {
            const std::size_t weightElements = static_cast<std::size_t>(kColumns) * kInner;

            this->input.Resize(kInner);
            this->weight.Resize(weightElements);
            this->bias.Resize(kColumns);
            this->output.Resize(kColumns);

            const std::vector<float> hostInput(kInner, 0.5f);
            const std::vector<float> hostWeight(weightElements, 0.25f);
            const std::vector<float> hostBias(kColumns, 0.125f);
            const std::vector<T> storageInput = ToStorage<T>(hostInput);
            const std::vector<T> storageWeight = ToStorage<T>(hostWeight);
            const std::vector<T> storageBias = ToStorage<T>(hostBias);

            this->input.CopyFromHost(storageInput.data(), storageInput.size());
            this->weight.CopyFromHost(storageWeight.data(), storageWeight.size());
            this->bias.CopyFromHost(storageBias.data(), storageBias.size());
        }

        ROCmKernels::DeviceBuffer<T> input;
        ROCmKernels::DeviceBuffer<T> weight;
        ROCmKernels::DeviceBuffer<T> bias;
        ROCmKernels::DeviceBuffer<T> output;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeLinearActVecCase() {
        const auto state = std::make_shared<LinearActVecState<T>>();
        const std::uint64_t bytes =
                (static_cast<std::uint64_t>(kColumns) * kInner + kColumns + kInner) * sizeof(T);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("LinearActVec ") + TypeName<T>() + " c4096 k4096",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::LinearActVec(state->input.Data(), state->weight.Data(), state->bias.Data(), state->output.Data(),
                            kColumns, kInner, ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytes,
            .bytesTouched = bytes,
            .operations = 2 * static_cast<std::uint64_t>(kColumns) * kInner,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeLinearActVecCase<float>());
            ROCmKernels::Benchmark::Register(MakeLinearActVecCase<__half>());
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
