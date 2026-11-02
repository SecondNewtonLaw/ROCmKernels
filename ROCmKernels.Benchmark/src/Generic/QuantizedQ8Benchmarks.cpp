#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/QuantizedQ8.hpp>

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

    std::vector<ROCmKernels::Kernels::BlockQ8_0> MakeBlocks(const std::size_t count, const float scale) {
        using ROCmKernels::Kernels::BlockQ8_0;

        std::vector<BlockQ8_0> blocks(count);
        for (std::size_t block = 0; block < count; ++block) {
            blocks[block].scale = __float2half_rn(scale);
            for (int index = 0; index < ROCmKernels::Kernels::kQ8Block; ++index) {
                blocks[block].values[index] = static_cast<std::int8_t>((index + static_cast<int>(block)) % 7 - 3);
            }
        }

        return blocks;
    }

    template<typename T> std::vector<T> MakeVector(const std::size_t count, const float value) {
        std::vector<T> values(count);
        for (std::size_t index = 0; index < count; ++index) {
            if constexpr (std::is_same_v<T, __half>) values[index] = __float2half_rn(value);
            else values[index] = value;
        }
        return values;
    }

    template<typename T> struct GemvState {
        void Initialize() {
            const std::size_t inputBlocks = kInner / ROCmKernels::Kernels::kQ8Block;
            const std::size_t weightBlocks = static_cast<std::size_t>(kColumns) * inputBlocks;

            this->input.Resize(inputBlocks);
            this->weight.Resize(weightBlocks);
            this->bias.Resize(kColumns);
            this->output.Resize(kColumns);

            const std::vector<ROCmKernels::Kernels::BlockQ8_0> hostInput = MakeBlocks(inputBlocks, 0.25f);
            const std::vector<ROCmKernels::Kernels::BlockQ8_0> hostWeight = MakeBlocks(weightBlocks, 0.125f);
            const std::vector<T> hostBias = MakeVector<T>(kColumns, 0.125f);

            this->input.CopyFromHost(hostInput.data(), hostInput.size());
            this->weight.CopyFromHost(hostWeight.data(), hostWeight.size());
            this->bias.CopyFromHost(hostBias.data(), hostBias.size());
        }

        ROCmKernels::DeviceBuffer<ROCmKernels::Kernels::BlockQ8_0> input;
        ROCmKernels::DeviceBuffer<ROCmKernels::Kernels::BlockQ8_0> weight;
        ROCmKernels::DeviceBuffer<T> bias;
        ROCmKernels::DeviceBuffer<T> output;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeCase() {
        const auto state = std::make_shared<GemvState<T>>();

        const std::uint64_t weightBytes = static_cast<std::uint64_t>(kColumns) * kInner / ROCmKernels::Kernels::kQ8Block *
                                          sizeof(ROCmKernels::Kernels::BlockQ8_0);
        const std::uint64_t inputBytes = static_cast<std::uint64_t>(kInner) / ROCmKernels::Kernels::kQ8Block *
                                        sizeof(ROCmKernels::Kernels::BlockQ8_0);
        const std::uint64_t vectorBytes = static_cast<std::uint64_t>(kColumns) * sizeof(T);
        const std::uint64_t bytesMoved = weightBytes + inputBytes + vectorBytes;

        return ROCmKernels::Benchmark::Case{
            .name = std::string("Q8_0 GEMV ") + TypeName<T>() + " c4096 k4096",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::QuantizedGEMVQ8_0(state->input.Data(), state->weight.Data(), state->bias.Data(),
                            state->output.Data(), kColumns, kInner, ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = bytesMoved,
            .operations = 2 * static_cast<std::uint64_t>(kColumns) * kInner,
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
