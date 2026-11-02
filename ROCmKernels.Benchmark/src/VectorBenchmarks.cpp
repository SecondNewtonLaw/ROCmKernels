//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Elementwise/Copy.hpp>
#include <ROCmKernels/Kernels/Elementwise/Vector.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
    template <typename T> constexpr const char *TypeName();

    template <> constexpr const char *TypeName<float>() { return "float"; }

    template <> constexpr const char *TypeName<double>() { return "double"; }

    std::string FormatElementCount(const std::size_t elementCount) {
        if (elementCount % (1u << 20) == 0) {
            return std::to_string(elementCount >> 20) + "M";
        }
        if (elementCount % (1u << 10) == 0) {
            return std::to_string(elementCount >> 10) + "K";
        }
        return std::to_string(elementCount);
    }

    template <typename T> struct VectorAddState {
        explicit VectorAddState(const std::size_t count) : elementCount(count) {}

        void Initialize() {
            this->lhs.Resize(this->elementCount);
            this->rhs.Resize(this->elementCount);
            this->out.Resize(this->elementCount);

            const std::vector<T> hostLhs(this->elementCount, static_cast<T>(1.25));
            const std::vector<T> hostRhs(this->elementCount, static_cast<T>(2.5));
            this->lhs.CopyFromHost(hostLhs.data(), this->elementCount);
            this->rhs.CopyFromHost(hostRhs.data(), this->elementCount);
        }

        void Run(hipStream_t stream) const {
            ROCmKernels::Kernels::VectorAdd(this->lhs.Data(), this->rhs.Data(), this->out.Data(), this->elementCount, stream);
        }

        std::size_t elementCount;
        ROCmKernels::DeviceBuffer<T> lhs;
        ROCmKernels::DeviceBuffer<T> rhs;
        ROCmKernels::DeviceBuffer<T> out;
    };

    template <typename T> ROCmKernels::Benchmark::Case MakeVectorAddCase(const std::size_t elementCount) {
        const auto state = std::make_shared<VectorAddState<T>>(elementCount);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("VectorAdd<") + TypeName<T>() + "> " + FormatElementCount(elementCount) + " elements",
            .setup = [state] { state->Initialize(); },
            .run = [state](hipStream_t stream) { state->Run(stream); },
            .bytesMoved = static_cast<std::uint64_t>(elementCount) * sizeof(T) * 3,
            .operations = static_cast<std::uint64_t>(elementCount),
        };
    }

    struct CopyState {
        explicit CopyState(const std::size_t count) : elementCount(count) {}

        void Initialize() {
            this->source.Resize(this->elementCount);
            this->destination.Resize(this->elementCount);

            const std::vector<float> hostSource(this->elementCount, 0.5f);
            this->source.CopyFromHost(hostSource.data(), hostSource.size());
        }

        void Run(hipStream_t stream) const { ROCmKernels::Kernels::Copy(this->source.Data(), this->destination.Data(), this->elementCount, stream); }

        std::size_t elementCount;
        ROCmKernels::DeviceBuffer<float> source;
        ROCmKernels::DeviceBuffer<float> destination;
    };

    ROCmKernels::Benchmark::Case MakeCopyCase(const std::size_t elementCount) {
        const auto state = std::make_shared<CopyState>(elementCount);

        return ROCmKernels::Benchmark::Case{
            .name = "Copy " + FormatElementCount(elementCount) + " elements",
            .setup = [state] { state->Initialize(); },
            .run = [state](hipStream_t stream) { state->Run(stream); },
            .bytesMoved = static_cast<std::uint64_t>(elementCount) * sizeof(float) * 2,
            .operations = 0,
        };
    }

    struct Registrar {
        Registrar() {
            using ROCmKernels::Benchmark::Register;

            Register(MakeVectorAddCase<float>(1u << 18));  // 1 MiB per buffer
            Register(MakeVectorAddCase<float>(1u << 22));  // 16 MiB per buffer
            Register(MakeVectorAddCase<float>(1u << 25));  // 128 MiB per buffer
            Register(MakeVectorAddCase<float>(1u << 27));  // 512 MiB per buffer
            Register(MakeVectorAddCase<double>(1u << 23)); // 64 MiB per buffer
            Register(MakeCopyCase(1u << 26));              // 256 MiB
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
