//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Attention/Attention.hpp>
#include <ROCmKernels/Kernels/Attention/Attention64.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
    template <int HeadWidth> struct State {
        State(const int batchSize, const int heads, const int queries, const int keys)
            : batchSize(batchSize), heads(heads), queries(queries), keys(keys), qElements(static_cast<std::size_t>(batchSize) * heads * queries * HeadWidth),
              kElements(static_cast<std::size_t>(batchSize) * heads * keys * HeadWidth) {}

        void Initialize(const bool backward, const bool tiled) {
            this->q.Resize(this->qElements);
            this->k.Resize(this->kElements);
            this->v.Resize(this->kElements);
            this->mask.Resize(static_cast<std::size_t>(this->batchSize) * this->keys);
            this->upstream.Resize(this->qElements);
            this->output.Resize(this->qElements);
            this->lse.Resize(static_cast<std::size_t>(this->batchSize) * this->heads * this->queries * 2);
            this->gradientQ.Resize(this->qElements);
            this->gradientK.Resize(this->kElements);
            this->gradientV.Resize(this->kElements);
            this->delta.Resize(static_cast<std::size_t>(this->batchSize) * this->heads * this->queries);

            const std::vector<float> hostValues(this->qElements, 0.5f);
            const std::vector<float> hostKeyValues(this->kElements, 0.25f);
            const std::vector<std::uint8_t> hostMask(static_cast<std::size_t>(this->batchSize) * this->keys, 1);
            this->q.CopyFromHost(hostValues.data(), hostValues.size());
            this->k.CopyFromHost(hostKeyValues.data(), hostKeyValues.size());
            this->v.CopyFromHost(hostKeyValues.data(), hostKeyValues.size());
            this->mask.CopyFromHost(hostMask.data(), hostMask.size());
            this->upstream.CopyFromHost(hostValues.data(), hostValues.size());

            if (backward) {
                this->Forward(tiled, nullptr);
            }
        }

        void Forward(const bool tiled, hipStream_t stream) {
            const auto boolMask = reinterpret_cast<const bool *>(this->mask.Data());
            if constexpr (HeadWidth == 32) {
                if (tiled) {
                    ROCmKernels::Kernels::AttentionForwardTiled(
                        this->q.Data(), this->k.Data(), this->v.Data(), boolMask, this->output.Data(), this->lse.Data(), this->batchSize, this->heads,
                        this->queries, this->keys, -1, this->qStrides(), this->kStrides(), this->kStrides(), this->qStrides(), stream
                    );
                } else {
                    ROCmKernels::Kernels::AttentionForward(
                        this->q.Data(), this->k.Data(), this->v.Data(), boolMask, this->output.Data(), this->lse.Data(), this->batchSize, this->heads,
                        this->queries, this->keys, -1, this->qStrides(), this->kStrides(), this->kStrides(), this->qStrides(), stream
                    );
                }
            } else {
                ROCmKernels::Kernels::Attention64Forward(
                    this->q.Data(), this->k.Data(), this->v.Data(), boolMask, this->output.Data(), this->lse.Data(), this->batchSize, this->heads,
                    this->queries, this->keys, -1, this->qStrides(), this->kStrides(), this->kStrides(), this->qStrides(), stream
                );
            }
        }

        void Backward(const bool tiled, hipStream_t stream) {
            const auto boolMask = reinterpret_cast<const bool *>(this->mask.Data());
            if constexpr (HeadWidth == 32) {
                if (tiled) {
                    ROCmKernels::Kernels::AttentionBackwardTiled(
                        this->q.Data(), this->k.Data(), this->v.Data(), boolMask, this->output.Data(), this->lse.Data(), this->upstream.Data(),
                        this->gradientQ.Data(), this->gradientK.Data(), this->gradientV.Data(), this->delta.Data(), this->batchSize, this->heads, this->queries,
                        this->keys, -1, this->qStrides(), this->kStrides(), this->kStrides(), this->qStrides(), this->qStrides(), stream
                    );
                } else {
                    ROCmKernels::Kernels::AttentionBackward(
                        this->q.Data(), this->k.Data(), this->v.Data(), boolMask, this->output.Data(), this->lse.Data(), this->upstream.Data(),
                        this->gradientQ.Data(), this->gradientK.Data(), this->gradientV.Data(), this->delta.Data(), this->batchSize, this->heads, this->queries,
                        this->keys, -1, this->qStrides(), this->kStrides(), this->kStrides(), this->qStrides(), this->qStrides(), stream
                    );
                }
            } else {
                ROCmKernels::Kernels::Attention64Backward(
                    this->q.Data(), this->k.Data(), this->v.Data(), boolMask, this->output.Data(), this->lse.Data(), this->upstream.Data(),
                    this->gradientQ.Data(), this->gradientK.Data(), this->gradientV.Data(), this->delta.Data(), this->batchSize, this->heads, this->queries,
                    this->keys, -1, this->qStrides(), this->kStrides(), this->kStrides(), this->qStrides(), this->qStrides(), stream
                );
            }
        }

        [[nodiscard]] ROCmKernels::Kernels::Strides qStrides() const {
            return ROCmKernels::Kernels::Strides{this->heads * this->queries * HeadWidth, this->queries * HeadWidth, HeadWidth};
        }

        [[nodiscard]] ROCmKernels::Kernels::Strides kStrides() const {
            return ROCmKernels::Kernels::Strides{this->heads * this->keys * HeadWidth, this->keys * HeadWidth, HeadWidth};
        }

        int batchSize;
        int heads;
        int queries;
        int keys;
        std::size_t qElements;
        std::size_t kElements;
        ROCmKernels::DeviceBuffer<float> q;
        ROCmKernels::DeviceBuffer<float> k;
        ROCmKernels::DeviceBuffer<float> v;
        ROCmKernels::DeviceBuffer<std::uint8_t> mask;
        ROCmKernels::DeviceBuffer<float> upstream;
        ROCmKernels::DeviceBuffer<float> output;
        ROCmKernels::DeviceBuffer<float> lse;
        ROCmKernels::DeviceBuffer<float> gradientQ;
        ROCmKernels::DeviceBuffer<float> gradientK;
        ROCmKernels::DeviceBuffer<float> gradientV;
        ROCmKernels::DeviceBuffer<float> delta;
    };

    template <int HeadWidth>
    ROCmKernels::Benchmark::Case
    MakeCase(const std::string &name, const int batchSize, const int heads, const int queries, const int keys, const bool tiled, const bool backward) {
        const auto state = std::make_shared<State<HeadWidth>>(batchSize, heads, queries, keys);

        const std::uint64_t qElements = static_cast<std::uint64_t>(batchSize) * heads * queries * HeadWidth;
        const std::uint64_t kElements = static_cast<std::uint64_t>(batchSize) * heads * keys * HeadWidth;
        const std::uint64_t qBytes = qElements * sizeof(float);
        const std::uint64_t kBytes = kElements * sizeof(float);
        const std::uint64_t uniqueBytes = backward ? 4 * qBytes + 4 * kBytes : 2 * qBytes + 2 * kBytes;

        const std::uint64_t blockRows = HeadWidth == 32 ? (tiled ? 16 : 4) : 16;
        const std::uint64_t queryBlockCount = static_cast<std::uint64_t>(batchSize) * heads * ((queries + blockRows - 1) / blockRows);
        const std::uint64_t keyBlockCount = static_cast<std::uint64_t>(batchSize) * heads * ((keys + blockRows - 1) / blockRows);
        const std::uint64_t kvTraffic = queryBlockCount * static_cast<std::uint64_t>(keys) * 2 * HeadWidth * sizeof(float);
        const std::uint64_t qTraffic = keyBlockCount * static_cast<std::uint64_t>(queries) * 2 * HeadWidth * sizeof(float);

        const std::uint64_t pairProducts = static_cast<std::uint64_t>(batchSize) * heads * queries * keys * HeadWidth;
        const std::uint64_t operations = backward ? 8 * pairProducts : 4 * pairProducts;

        std::uint64_t bytesMoved = kvTraffic + 2 * qBytes;
        if (backward) {
            bytesMoved = kvTraffic + qTraffic + 3 * qBytes + 4 * kBytes;
        }

        return ROCmKernels::Benchmark::Case{
            .name = name,
            .setup = [state, backward, tiled] { state->Initialize(backward, tiled); },
            .run =
                [state, backward, tiled](hipStream_t stream) {
                    if (backward) {
                        state->Backward(tiled, stream);
                    } else {
                        state->Forward(tiled, stream);
                    }
                },
            .bytesMoved = bytesMoved,
            .bytesTouched = uniqueBytes,
            .operations = operations,
        };
    }

    ROCmKernels::Benchmark::Case MakeFlashCase(
        const std::string &name, const int batchSize, const int heads, const int queries, const int keys, const int variant, const std::uint64_t blockRows
    ) {
        const auto state = std::make_shared<State<32>>(batchSize, heads, queries, keys);

        const std::uint64_t qBytes = static_cast<std::uint64_t>(batchSize) * heads * queries * 32 * sizeof(float);
        const std::uint64_t kBytes = static_cast<std::uint64_t>(batchSize) * heads * keys * 32 * sizeof(float);
        const std::uint64_t queryBlockCount = static_cast<std::uint64_t>(batchSize) * heads * ((queries + blockRows - 1) / blockRows);
        const std::uint64_t kvTraffic = queryBlockCount * static_cast<std::uint64_t>(keys) * 2 * 32 * sizeof(float);

        return ROCmKernels::Benchmark::Case{
            .name = name,
            .setup = [state] { state->Initialize(false, false); },
            .run =
                [state, variant](hipStream_t stream) {
                    if (variant < 0) {
                        ROCmKernels::Kernels::AttentionForwardFlash(
                            state->q.Data(), state->k.Data(), state->v.Data(), reinterpret_cast<const bool *>(state->mask.Data()),
                            state->output.Data(), state->lse.Data(), state->batchSize, state->heads, state->queries, state->keys, -1,
                            state->qStrides(), state->kStrides(), state->kStrides(), state->qStrides(), stream
                        );
                        return;
                    }
                    ROCmKernels::Kernels::AttentionForwardFlashTuned(
                        state->q.Data(), state->k.Data(), state->v.Data(), reinterpret_cast<const bool *>(state->mask.Data()),
                        state->output.Data(), state->lse.Data(), state->batchSize, state->heads, state->queries, state->keys, -1, variant,
                        state->qStrides(), state->kStrides(), state->kStrides(), state->qStrides(), stream
                    );
                },
            .bytesMoved = kvTraffic + 2 * qBytes,
            .bytesTouched = 2 * qBytes + 2 * kBytes,
            .operations = 4 * static_cast<std::uint64_t>(batchSize) * heads * queries * keys * 32,
        };
    }

    struct Registrar {
        Registrar() {
            using ROCmKernels::Benchmark::Register;

            Register(MakeFlashCase("Attention32 flash forward b2 h12 l1024", 2, 12, 1024, 1024, -1, 64));
            Register(MakeCase<32>("Attention32 forward b2 h12 l1024", 2, 12, 1024, 1024, false, false));
            Register(MakeCase<32>("Attention32 backward b2 h12 l1024", 2, 12, 1024, 1024, false, true));
            Register(MakeCase<32>("Attention32 tiled forward b2 h12 l1024", 2, 12, 1024, 1024, true, false));
            Register(MakeCase<32>("Attention32 tiled backward b2 h12 l1024", 2, 12, 1024, 1024, true, true));
            Register(MakeCase<64>("Attention64 forward b1 h4 l512", 1, 4, 512, 512, false, false));
            Register(MakeCase<64>("Attention64 backward b1 h4 l512", 1, 4, 512, 512, false, true));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
