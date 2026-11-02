//
// Created by Dottik on 2/11/2026.
//

#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Embedding/EmbeddingGrad.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {
    struct State {
        State(const int nnz, const int rows, const int columns) : nnz(nnz), rows(rows), columns(columns) {}

        void Initialize() {
            this->keys.Resize(this->nnz);
            this->order.Resize(this->nnz);
            this->values.Resize(static_cast<std::size_t>(this->nnz) * this->columns);
            this->dense.Resize(static_cast<std::size_t>(this->rows) * this->columns);

            std::vector<std::int64_t> hostKeys(this->nnz);
            std::vector<std::int64_t> hostOrder(this->nnz);
            for (int index = 0; index < this->nnz; ++index) {
                hostKeys[index] = static_cast<std::int64_t>(index) * this->rows / this->nnz;
                hostOrder[index] = index;
            }

            const std::vector<float> hostValues(static_cast<std::size_t>(this->nnz) * this->columns, 0.5f);
            this->keys.CopyFromHost(hostKeys.data(), hostKeys.size());
            this->order.CopyFromHost(hostOrder.data(), hostOrder.size());
            this->values.CopyFromHost(hostValues.data(), hostValues.size());
        }

        int nnz;
        int rows;
        int columns;
        ROCmKernels::DeviceBuffer<std::int64_t> keys;
        ROCmKernels::DeviceBuffer<std::int64_t> order;
        ROCmKernels::DeviceBuffer<float> values;
        ROCmKernels::DeviceBuffer<float> dense;
    };

    ROCmKernels::Benchmark::Case MakeCase(const int nnz, const int rows, const int columns, const bool vectorized) {
        const auto state = std::make_shared<State>(nnz, rows, columns);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("EmbeddingGrad ") + (vectorized ? "vectorized" : "scalar") + " nnz=" + std::to_string(nnz) + " rows=" + std::to_string(rows) +
                    " columns=" + std::to_string(columns),
            .setup = [state] { state->Initialize(); },
            .run =
                [state, vectorized](hipStream_t stream) {
                    ROCmKernels::Kernels::EmbeddingGradTuned(
                        state->keys.Data(), state->order.Data(), state->values.Data(), state->dense.Data(), state->nnz, state->rows, state->columns, vectorized,
                        stream
                    );
                },
            .bytesMoved = (static_cast<std::uint64_t>(nnz) + rows) * columns * sizeof(float),
            .operations = 0,
        };
    }

    struct Registrar {
        Registrar() {
            ROCmKernels::Benchmark::Register(MakeCase(4096, 512, 384, false));
            ROCmKernels::Benchmark::Register(MakeCase(4096, 512, 384, true));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
