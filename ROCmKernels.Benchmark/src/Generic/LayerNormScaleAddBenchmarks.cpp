#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/LayerNormScaleAdd.hpp>

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
            const std::size_t statisticsCount = static_cast<std::size_t>(kRows) * 2;

            this->x.Resize(elements);
            this->z.Resize(elements);
            this->scale.Resize(kColumns);
            this->weight.Resize(kColumns);
            this->bias.Resize(kColumns);
            this->output.Resize(elements);
            this->statistics.Resize(statisticsCount);
            this->upstream.Resize(elements);
            this->gradientX.Resize(elements);
            this->gradientZ.Resize(elements);
            this->gradientScale.Resize(kColumns);
            this->gradientWeight.Resize(kColumns);
            this->gradientBias.Resize(kColumns);

            const std::vector<float> hostValues(elements, 0.5f);
            const std::vector<float> hostScale(kColumns, 1.0f);
            const std::vector<float> hostWeight(kColumns, 1.0f);
            const std::vector<float> hostBias(kColumns, 0.0f);
            std::vector<float> hostStatistics(statisticsCount);
            for (int row = 0; row < kRows; ++row) {
                hostStatistics[static_cast<std::size_t>(row) * 2] = 0.0f;
                hostStatistics[static_cast<std::size_t>(row) * 2 + 1] = 1.0f;
            }

            const std::vector<T> storageValues = ToStorage<T>(hostValues);
            const std::vector<T> storageScale = ToStorage<T>(hostScale);
            const std::vector<T> storageWeight = ToStorage<T>(hostWeight);
            const std::vector<T> storageBias = ToStorage<T>(hostBias);

            this->x.CopyFromHost(storageValues.data(), storageValues.size());
            this->z.CopyFromHost(storageValues.data(), storageValues.size());
            this->upstream.CopyFromHost(storageValues.data(), storageValues.size());
            this->scale.CopyFromHost(storageScale.data(), storageScale.size());
            this->weight.CopyFromHost(storageWeight.data(), storageWeight.size());
            this->bias.CopyFromHost(storageBias.data(), storageBias.size());
            this->statistics.CopyFromHost(hostStatistics.data(), hostStatistics.size());
        }

        ROCmKernels::DeviceBuffer<T> x;
        ROCmKernels::DeviceBuffer<T> z;
        ROCmKernels::DeviceBuffer<T> scale;
        ROCmKernels::DeviceBuffer<T> weight;
        ROCmKernels::DeviceBuffer<T> bias;
        ROCmKernels::DeviceBuffer<T> output;
        ROCmKernels::DeviceBuffer<float> statistics;
        ROCmKernels::DeviceBuffer<T> upstream;
        ROCmKernels::DeviceBuffer<T> gradientX;
        ROCmKernels::DeviceBuffer<T> gradientZ;
        ROCmKernels::DeviceBuffer<float> gradientScale;
        ROCmKernels::DeviceBuffer<float> gradientWeight;
        ROCmKernels::DeviceBuffer<float> gradientBias;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeForwardCase() {
        const auto state = std::make_shared<State<T>>();
        const std::uint64_t elements = static_cast<std::uint64_t>(kRows) * kColumns;
        const std::uint64_t bytesMoved = elements * sizeof(T) * 4 + static_cast<std::uint64_t>(kRows) * 2 * sizeof(float);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("LayerNormScaleAdd ") + TypeName<T>() + " forward b4096 c1024",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::LayerNormScaleAddForward(state->x.Data(), state->z.Data(), state->scale.Data(), state->weight.Data(),
                            state->bias.Data(), state->output.Data(), state->statistics.Data(), kRows, kColumns, kEpsilon,
                            ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = bytesMoved,
            .operations = elements * 8,
        };
    }

    template<typename T> ROCmKernels::Benchmark::Case MakeBackwardCase() {
        const auto state = std::make_shared<State<T>>();
        const std::uint64_t elements = static_cast<std::uint64_t>(kRows) * kColumns;
        const std::uint64_t bytesMoved = elements * sizeof(T) * 5 + static_cast<std::uint64_t>(kColumns) * 3 * sizeof(float);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("LayerNormScaleAdd ") + TypeName<T>() + " backward b4096 c1024",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::LayerNormScaleAddBackward(state->x.Data(), state->z.Data(), state->scale.Data(), state->weight.Data(),
                            state->statistics.Data(), state->upstream.Data(), state->gradientX.Data(), state->gradientZ.Data(),
                            state->gradientScale.Data(), state->gradientWeight.Data(), state->gradientBias.Data(), kRows, kColumns,
                            ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = bytesMoved,
            .operations = elements * 16,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeForwardCase<float>());
            ROCmKernels::Benchmark::Register(MakeBackwardCase<float>());
            ROCmKernels::Benchmark::Register(MakeForwardCase<__half>());
            ROCmKernels::Benchmark::Register(MakeBackwardCase<__half>());
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
