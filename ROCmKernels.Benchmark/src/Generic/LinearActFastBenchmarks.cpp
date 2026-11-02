#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/LinearActFast.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
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

    // The training shapes from the step profile: 8192 tokens of width 128 and an
    // 8192-wide FFN, plus the qkv and output projections next to them.
    struct Shape {
        int rows;
        int columns;
        int inner;
        const char *name;
    };

    constexpr Shape kShapes[] = {
        {4096, 4096, 1024, "4096x4096x1024"},
        {8192, 8192, 128, "ffn up 8192x8192x128"},
        {8192, 128, 8192, "ffn down 8192x128x8192"},
        {8192, 384, 128, "qkv 8192x384x128"},
        {8192, 128, 128, "o 8192x128x128"},
    };

    template<typename T> struct LinearFastState {
        void Initialize(const Shape &shape) {
            const std::size_t inputElements = static_cast<std::size_t>(shape.rows) * shape.inner;
            const std::size_t weightElements = static_cast<std::size_t>(shape.inner) * shape.columns;
            const std::size_t outputElements = static_cast<std::size_t>(shape.rows) * shape.columns;

            this->input.Resize(inputElements);
            this->weight.Resize(weightElements);
            this->bias.Resize(shape.columns);
            this->output.Resize(outputElements);

            const std::vector<float> hostInput(inputElements, 0.5f);
            const std::vector<float> hostWeight(weightElements, 0.25f);
            const std::vector<float> hostBias(shape.columns, 0.125f);
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

    template<typename T> ROCmKernels::Benchmark::Case MakeLinearFastCase(const Shape &shape) {
        const auto state = std::make_shared<LinearFastState<T>>();
        const std::uint64_t outputElements = static_cast<std::uint64_t>(shape.rows) * shape.columns;
        const std::uint64_t bytesMoved = (static_cast<std::uint64_t>(shape.rows) * shape.inner +
                static_cast<std::uint64_t>(shape.inner) * shape.columns + outputElements) * sizeof(T);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("LinearActFast ") + TypeName<T>() + " " + shape.name,
            .setup = [state, shape] { state->Initialize(shape); },
            .run =
                [state, shape](hipStream_t stream) {
                    ROCmKernels::Kernels::LinearActFast(state->input.Data(), state->weight.Data(), state->bias.Data(), state->output.Data(),
                            shape.rows, shape.columns, shape.inner, ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = bytesMoved,
            .operations = 2 * static_cast<std::uint64_t>(shape.rows) * shape.columns * shape.inner,
        };
    }

    struct Registrar {
        Registrar() {
            for (const Shape &shape : kShapes) {
                ROCmKernels::Benchmark::Register(MakeLinearFastCase<float>(shape));
                ROCmKernels::Benchmark::Register(MakeLinearFastCase<__half>(shape));
            }
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
