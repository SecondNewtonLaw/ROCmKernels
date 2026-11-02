#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/AttentionGeneric.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kBatch = 2;
    constexpr int kHeads = 8;
    constexpr int kQueries = 512;
    constexpr int kKeys = 512;
    constexpr int kRadius = -1;

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

    template<typename T, int HeadWidth> struct State {
        void Initialize() {
            const std::size_t qElements = static_cast<std::size_t>(kBatch) * kHeads * kQueries * HeadWidth;
            const std::size_t kElements = static_cast<std::size_t>(kBatch) * kHeads * kKeys * HeadWidth;

            this->q.Resize(qElements);
            this->k.Resize(kElements);
            this->v.Resize(kElements);
            this->mask.Resize(static_cast<std::size_t>(kBatch) * kKeys);
            this->output.Resize(qElements);
            this->lse.Resize(static_cast<std::size_t>(kBatch) * kHeads * kQueries * 2);

            const std::vector<T> hostValues = ToStorage<T>(std::vector<float>(qElements, 0.5f));
            const std::vector<T> hostKeyValues = ToStorage<T>(std::vector<float>(kElements, 0.25f));
            const std::vector<std::uint8_t> hostMask(static_cast<std::size_t>(kBatch) * kKeys, 1);

            this->q.CopyFromHost(hostValues.data(), hostValues.size());
            this->k.CopyFromHost(hostKeyValues.data(), hostKeyValues.size());
            this->v.CopyFromHost(hostKeyValues.data(), hostKeyValues.size());
            this->mask.CopyFromHost(hostMask.data(), hostMask.size());
        }

        void Forward(const hipStream_t stream) {
            ROCmKernels::Kernels::AttentionGenericForward(this->q.Data(), this->k.Data(), this->v.Data(),
                    reinterpret_cast<const bool *>(this->mask.Data()), this->output.Data(), this->lse.Data(), kBatch, kHeads, kQueries, kKeys,
                    HeadWidth, kRadius, this->qStrides(), this->kStrides(), this->kStrides(), this->qStrides(),
                    ROCmKernels::Kernels::DTypeOf<T>::value, stream);
        }

        [[nodiscard]] ROCmKernels::Kernels::Strides qStrides() const {
            return ROCmKernels::Kernels::Strides{kHeads * kQueries * HeadWidth, kQueries * HeadWidth, HeadWidth};
        }

        [[nodiscard]] ROCmKernels::Kernels::Strides kStrides() const {
            return ROCmKernels::Kernels::Strides{kHeads * kKeys * HeadWidth, kKeys * HeadWidth, HeadWidth};
        }

        ROCmKernels::DeviceBuffer<T> q;
        ROCmKernels::DeviceBuffer<T> k;
        ROCmKernels::DeviceBuffer<T> v;
        ROCmKernels::DeviceBuffer<std::uint8_t> mask;
        ROCmKernels::DeviceBuffer<T> output;
        ROCmKernels::DeviceBuffer<float> lse;
    };

    template<typename T, int HeadWidth> ROCmKernels::Benchmark::Case MakeCase() {
        const auto state = std::make_shared<State<T, HeadWidth>>();

        const std::uint64_t qBytes = static_cast<std::uint64_t>(kBatch) * kHeads * kQueries * HeadWidth * sizeof(T);
        const std::uint64_t kBytes = static_cast<std::uint64_t>(kBatch) * kHeads * kKeys * HeadWidth * sizeof(T);
        const std::uint64_t queryBlocks = static_cast<std::uint64_t>(kBatch) * kHeads * ((kQueries + 7) / 8);
        const std::uint64_t keyTraffic = queryBlocks * static_cast<std::uint64_t>(kKeys) * 2 * HeadWidth * sizeof(T);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("AttentionGeneric") + std::to_string(HeadWidth) + " " + TypeName<T>() + " b2 h8 l512",
            .setup = [state] { state->Initialize(); },
            .run =
                [state](hipStream_t stream) {
                    state->Forward(stream);
                },
            .bytesMoved = keyTraffic + 2 * qBytes,
            .bytesTouched = 2 * qBytes + 2 * kBytes,
            .operations = 4 * static_cast<std::uint64_t>(kBatch) * kHeads * kQueries * kKeys * HeadWidth,
        };
    }

    struct Registrar {
        Registrar() {
            using ROCmKernels::Benchmark::Register;

            Register(MakeCase<float, 16>());
            Register(MakeCase<float, 32>());
            Register(MakeCase<float, 64>());
            Register(MakeCase<__half, 16>());
            Register(MakeCase<__half, 32>());
            Register(MakeCase<__half, 64>());
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
