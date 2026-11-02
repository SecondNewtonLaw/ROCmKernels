#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/BroadcastCopy.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kRows = 4096;
    constexpr int kColumns = 1024;
    constexpr int kSourceRowStride = kColumns + 3;
    constexpr int kDestinationRowStride = kColumns + 5;
    constexpr long long kFillElements = 4'000'000;

    template<typename T> const char *TypeName() {
        if constexpr (std::is_same_v<T, __half>) return "fp16";
        else return "fp32";
    }

    template<typename T> T ToStorage(const float value) {
        if constexpr (std::is_same_v<T, __half>) return __float2half_rn(value);
        else return value;
    }

    template<typename T> struct BroadcastState {
        void Initialize() {
            this->a.Resize(static_cast<std::size_t>(kRows) * kColumns);
            this->b.Resize(kColumns);
            this->out.Resize(static_cast<std::size_t>(kRows) * kColumns);

            const std::vector<T> hostA(static_cast<std::size_t>(kRows) * kColumns, ToStorage<T>(0.5f));
            const std::vector<T> hostB(kColumns, ToStorage<T>(0.25f));
            this->a.CopyFromHost(hostA.data(), hostA.size());
            this->b.CopyFromHost(hostB.data(), hostB.size());
        }

        ROCmKernels::DeviceBuffer<T> a;
        ROCmKernels::DeviceBuffer<T> b;
        ROCmKernels::DeviceBuffer<T> out;
    };

    template<typename T> struct CopyState {
        void Initialize() {
            this->source.Resize(static_cast<std::size_t>(kRows) * kSourceRowStride);
            this->destination.Resize(static_cast<std::size_t>(kRows) * kDestinationRowStride);

            const std::vector<T> hostSource(static_cast<std::size_t>(kRows) * kSourceRowStride, ToStorage<T>(0.5f));
            this->source.CopyFromHost(hostSource.data(), hostSource.size());
        }

        ROCmKernels::DeviceBuffer<T> source;
        ROCmKernels::DeviceBuffer<T> destination;
    };

    template<typename T> struct FillState {
        void Initialize() { this->output.Resize(static_cast<std::size_t>(kFillElements)); }

        ROCmKernels::DeviceBuffer<T> output;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeBroadcastCase() {
        const auto state = std::make_shared<BroadcastState<T>>();
        const std::uint64_t elements = static_cast<std::uint64_t>(kRows) * kColumns;

        return ROCmKernels::Benchmark::Case{
            .name = std::string("BroadcastBinary add row ") + TypeName<T>() + " 4096x1024",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::BroadcastBinary(state->a.Data(), state->b.Data(), state->out.Data(), kRows, kColumns,
                            ROCmKernels::Kernels::BinaryOp::Add, ROCmKernels::Kernels::BroadcastMode::Row,
                            ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = elements * sizeof(T) * 3,
            .bytesTouched = (elements * 2 + kColumns) * sizeof(T),
            .operations = elements,
        };
    }

    template<typename T> ROCmKernels::Benchmark::Case MakeCopyCase() {
        const auto state = std::make_shared<CopyState<T>>();
        const std::uint64_t elements = static_cast<std::uint64_t>(kRows) * kColumns;

        return ROCmKernels::Benchmark::Case{
            .name = std::string("Copy2D ") + TypeName<T>() + " 4096x1024",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::Copy2D(state->source.Data(), state->destination.Data(), kRows, kColumns, kSourceRowStride,
                            kDestinationRowStride, ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = elements * sizeof(T) * 2,
            .bytesTouched = elements * sizeof(T) * 2,
        };
    }

    template<typename T> ROCmKernels::Benchmark::Case MakeFillCase() {
        const auto state = std::make_shared<FillState<T>>();

        return ROCmKernels::Benchmark::Case{
            .name = std::string("Fill ") + TypeName<T>() + " 4M",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    ROCmKernels::Kernels::Fill(state->output.Data(), 0.5f, kFillElements, ROCmKernels::Kernels::DTypeOf<T>::value, stream);
                },
            .bytesMoved = static_cast<std::uint64_t>(kFillElements) * sizeof(T),
            .bytesTouched = static_cast<std::uint64_t>(kFillElements) * sizeof(T),
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeBroadcastCase<float>());
            ROCmKernels::Benchmark::Register(MakeBroadcastCase<__half>());
            ROCmKernels::Benchmark::Register(MakeCopyCase<float>());
            ROCmKernels::Benchmark::Register(MakeCopyCase<__half>());
            ROCmKernels::Benchmark::Register(MakeFillCase<float>());
            ROCmKernels::Benchmark::Register(MakeFillCase<__half>());
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
