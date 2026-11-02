#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/TopK.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kRows = 512;
    constexpr int kClasses = 55765;
    constexpr int kK = 8;

    template<typename T> const char *TypeName() {
        if constexpr (std::is_same_v<T, __half>) return "fp16";
        else return "fp32";
    }

    template<typename T> struct State {
        void Initialize() {
            const std::size_t elementCount = static_cast<std::size_t>(kRows) * kClasses;

            this->scores.Resize(elementCount);
            this->values.Resize(static_cast<std::size_t>(kRows) * kK);
            this->indices.Resize(static_cast<std::size_t>(kRows) * kK);

            // A cheap deterministic spread keeps the per-thread insertion rate realistic.
            std::vector<T> host(elementCount);
            std::uint32_t state = 0x9E3779B9u;
            for (std::size_t index = 0; index < elementCount; ++index) {
                state = state * 1664525u + 1013904223u;
                const float value = static_cast<float>(state >> 8) * (1.0f / 16777216.0f) - 0.5f;
                if constexpr (std::is_same_v<T, __half>) host[index] = __float2half_rn(value);
                else host[index] = value;
            }

            this->scores.CopyFromHost(host.data(), host.size());
        }

        void Forward(const hipStream_t stream) {
            ROCmKernels::Kernels::TopKForward(this->scores.Data(), this->values.Data(), this->indices.Data(), kRows, kClasses, kK,
                    ROCmKernels::Kernels::DTypeOf<T>::value, stream);
        }

        ROCmKernels::DeviceBuffer<T> scores;
        ROCmKernels::DeviceBuffer<float> values;
        ROCmKernels::DeviceBuffer<int> indices;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeCase() {
        const auto state = std::make_shared<State<T>>();

        const std::uint64_t scoreBytes = static_cast<std::uint64_t>(kRows) * kClasses * sizeof(T);
        const std::uint64_t outputBytes = static_cast<std::uint64_t>(kRows) * kK * (sizeof(float) + sizeof(int));

        return ROCmKernels::Benchmark::Case{
            .name = std::string("TopK ") + TypeName<T>() + " r512 c55765 k8",
            .setup = [state] { state->Initialize(); },
            .run = [state](hipStream_t stream) { state->Forward(stream); },
            .bytesMoved = scoreBytes + outputBytes,
            .bytesTouched = scoreBytes + outputBytes,
            .operations = static_cast<std::uint64_t>(kRows) * kClasses * 2,
        };
    }

    struct Registrar {
        Registrar() {
            using ROCmKernels::Benchmark::Register;

            Register(MakeCase<float>());
            Register(MakeCase<__half>());
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
