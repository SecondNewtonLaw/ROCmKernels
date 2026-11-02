#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/RMSNorm.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kRows = 4096;
    constexpr int kColumns = 1024;
    constexpr float kEpsilon = 1e-5f;

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

            this->x.Resize(elements);
            this->residual.Resize(elements);
            this->scale.Resize(kColumns);
            this->output.Resize(elements);
            this->rstd.Resize(kRows);

            const std::vector<float> hostValues(elements, 0.5f);
            const std::vector<float> hostScale(kColumns, 1.0f);
            const std::vector<T> storageValues = ToStorage<T>(hostValues);
            const std::vector<T> storageScale = ToStorage<T>(hostScale);

            this->x.CopyFromHost(storageValues.data(), storageValues.size());
            this->residual.CopyFromHost(storageValues.data(), storageValues.size());
            this->scale.CopyFromHost(storageScale.data(), storageScale.size());
        }

        ROCmKernels::DeviceBuffer<T> x;
        ROCmKernels::DeviceBuffer<T> residual;
        ROCmKernels::DeviceBuffer<T> scale;
        ROCmKernels::DeviceBuffer<T> output;
        ROCmKernels::DeviceBuffer<float> rstd;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeForwardCase() {
        const auto state = std::make_shared<State<T>>();
        const std::uint64_t elements = static_cast<std::uint64_t>(kRows) * kColumns;
        const std::uint64_t bytesMoved = elements * sizeof(T) * 4 + static_cast<std::uint64_t>(kRows) * sizeof(float);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("RMSNorm ") + TypeName<T>() + " fwd b4096 c1024",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::RMSNormForward(state->x.Data(), state->residual.Data(), state->scale.Data(), state->output.Data(),
                            state->rstd.Data(), kRows, kColumns, kEpsilon, ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = bytesMoved,
            .operations = elements * 4,
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
