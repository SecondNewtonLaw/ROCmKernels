#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/LinearAct.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kRows = 4096;
    constexpr int kColumns = 4096;
    constexpr int kInner = 1024;
    constexpr int kSiLUElements = 4 * 1024 * 1024;

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

    template<typename T> struct LinearState {
        void Initialize() {
            const std::size_t inputElements = static_cast<std::size_t>(kRows) * kInner;
            const std::size_t weightElements = static_cast<std::size_t>(kInner) * kColumns;
            const std::size_t outputElements = static_cast<std::size_t>(kRows) * kColumns;

            this->input.Resize(inputElements);
            this->weight.Resize(weightElements);
            this->bias.Resize(kColumns);
            this->output.Resize(outputElements);

            const std::vector<float> hostInput(inputElements, 0.5f);
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

    template<typename T> ROCmKernels::Benchmark::Case MakeLinearCase() {
        const auto state = std::make_shared<LinearState<T>>();
        const std::uint64_t outputElements = static_cast<std::uint64_t>(kRows) * kColumns;
        const std::uint64_t bytesMoved =
                (static_cast<std::uint64_t>(kRows) * kInner + static_cast<std::uint64_t>(kInner) * kColumns + outputElements) * sizeof(T);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("LinearAct ") + TypeName<T>() + " 4096x4096x1024",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::LinearActForward(state->input.Data(), state->weight.Data(), state->bias.Data(), state->output.Data(),
                            kRows, kColumns, kInner, ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = bytesMoved,
            .operations = 2 * static_cast<std::uint64_t>(kRows) * kColumns * kInner + outputElements * 4,
        };
    }

    template<typename T> struct SiLUState {
        void Initialize() {
            this->input.Resize(kSiLUElements);
            this->output.Resize(kSiLUElements);

            const std::vector<float> hostInput(kSiLUElements, 0.5f);
            const std::vector<T> storageInput = ToStorage<T>(hostInput);
            this->input.CopyFromHost(storageInput.data(), storageInput.size());
        }

        ROCmKernels::DeviceBuffer<T> input;
        ROCmKernels::DeviceBuffer<T> output;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeSiLUCase() {
        const auto state = std::make_shared<SiLUState<T>>();
        const std::uint64_t elements = kSiLUElements;
        const std::uint64_t bytesMoved = 2 * elements * sizeof(T);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("SiLU ") + TypeName<T>() + " 4M",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::SiLU(state->input.Data(), state->output.Data(), static_cast<long long>(kSiLUElements),
                            ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = bytesMoved,
            .operations = elements * 4,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeLinearCase<float>());
            ROCmKernels::Benchmark::Register(MakeLinearCase<__half>());
            ROCmKernels::Benchmark::Register(MakeSiLUCase<float>());
            ROCmKernels::Benchmark::Register(MakeSiLUCase<__half>());
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
