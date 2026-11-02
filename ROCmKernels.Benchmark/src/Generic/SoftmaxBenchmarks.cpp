#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/Softmax.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kRows = 4096;
    constexpr int kColumns = 1024;
    constexpr float kScale = 0.03125f;

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
            const std::size_t elements = static_cast<std::size_t>(kRows) * kColumns;

            this->input.Resize(elements);
            this->output.Resize(elements);

            const std::vector<float> hostValues(elements, 0.5f);
            const std::vector<T> storageValues = ToStorage<T>(hostValues);

            this->input.CopyFromHost(storageValues.data(), storageValues.size());
        }

        ROCmKernels::DeviceBuffer<T> input;
        ROCmKernels::DeviceBuffer<T> output;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeForwardCase() {
        const auto state = std::make_shared<State<T>>();
        const std::uint64_t elements = static_cast<std::uint64_t>(kRows) * kColumns;
        const std::uint64_t bytes = elements * sizeof(T) * 2;

        return ROCmKernels::Benchmark::Case{
            .name = std::string("Softmax ") + TypeName<T>() + " r4096 c1024",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::SoftmaxForward(state->input.Data(), nullptr, state->output.Data(), kRows, kColumns, kScale,
                            ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytes,
            .bytesTouched = bytes,
            .operations = elements * 5,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeForwardCase<float>());
            ROCmKernels::Benchmark::Register(MakeForwardCase<__half>());
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
