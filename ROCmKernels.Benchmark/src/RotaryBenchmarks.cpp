//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Positional/Rotary.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
    constexpr int kHeadWidth = 32;

    struct State {
        State(const int batchSize, const int heads, const int length)
            : batchSize(batchSize), heads(heads), length(length), elements(batchSize * heads * length * kHeadWidth),
              elementsPerTable(batchSize * length * kHeadWidth) {}

        void Initialize() {
            this->q.Resize(this->elements);
            this->k.Resize(this->elements);
            this->cosine.Resize(this->elementsPerTable);
            this->sine.Resize(this->elementsPerTable);
            this->outQ.Resize(this->elements);
            this->outK.Resize(this->elements);

            const std::vector<float> hostValues(this->elements, 0.5f);
            const std::vector<float> hostTable(this->elementsPerTable, 0.5f);
            this->q.CopyFromHost(hostValues.data(), hostValues.size());
            this->k.CopyFromHost(hostValues.data(), hostValues.size());
            this->cosine.CopyFromHost(hostTable.data(), hostTable.size());
            this->sine.CopyFromHost(hostTable.data(), hostTable.size());
        }

        int batchSize;
        int heads;
        int length;
        int elements;
        int elementsPerTable;
        ROCmKernels::DeviceBuffer<float> q;
        ROCmKernels::DeviceBuffer<float> k;
        ROCmKernels::DeviceBuffer<float> cosine;
        ROCmKernels::DeviceBuffer<float> sine;
        ROCmKernels::DeviceBuffer<float> outQ;
        ROCmKernels::DeviceBuffer<float> outK;
    };

    ROCmKernels::Benchmark::Case MakeCase(const std::string &name, const int batchSize, const int heads, const int length, const bool backward) {
        const auto state = std::make_shared<State>(batchSize, heads, length);
        const ROCmKernels::Kernels::Strides qkStrides{length * heads * kHeadWidth, kHeadWidth, heads * kHeadWidth};
        const ROCmKernels::Kernels::Strides tableStrides{length * kHeadWidth, 0, kHeadWidth};

        return ROCmKernels::Benchmark::Case{
            .name = name,
            .setup = [state] { state->Initialize(); },
            .run =
                [state, qkStrides, tableStrides, backward](hipStream_t stream) {
                    ROCmKernels::Kernels::Rotary(
                        state->q.Data(), state->k.Data(), state->cosine.Data(), state->sine.Data(), state->outQ.Data(), state->outK.Data(), state->batchSize,
                        state->heads, state->length, backward, qkStrides, qkStrides, tableStrides, tableStrides, stream
                    );
                },
            .bytesMoved = static_cast<std::uint64_t>(state->elements) * sizeof(float) * 4,
            .operations = 0,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeCase("Rotary forward b2 h12 l1024", 2, 12, 1024, false));
            ROCmKernels::Benchmark::Register(MakeCase("Rotary backward b2 h12 l1024", 2, 12, 1024, true));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
