#include <ROCmKernels/Benchmark/Harness.hpp>

#include <ROCmKernels/Common/DeviceBuffer.hpp>
#include <ROCmKernels/Kernels/Generic/RowGather.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    constexpr int kTableRows = 55765;
    constexpr int kRows = 512;
    constexpr int kColumns = 256;

    enum class Operation : unsigned char { Get, Set, Add };

    template<typename T> const char *TypeName() {
        if constexpr (std::is_same_v<T, __half>) return "fp16";
        else return "fp32";
    }

    const char *OperationName(const Operation operation) {
        switch (operation) {
            case Operation::Get: return "get";
            case Operation::Set: return "set";
            case Operation::Add: return "add";
        }
        return "unknown";
    }

    template<typename T> T Narrow(const float value) {
        if constexpr (std::is_same_v<T, __half>) return __float2half_rn(value);
        else return value;
    }

    template<typename T> struct State {
        explicit State(const Operation operation) : operation(operation) {}

        void Initialize() {
            const std::size_t tableCount = static_cast<std::size_t>(kTableRows) * kColumns;
            const std::size_t rowCount = static_cast<std::size_t>(kRows) * kColumns;

            this->table.Resize(tableCount);
            this->input.Resize(rowCount);
            this->output.Resize(rowCount);
            this->indices.Resize(kRows);

            std::vector<T> hostTable(tableCount);
            std::vector<T> hostInput(rowCount);
            std::uint32_t state = 0x9E3779B9u;
            for (std::size_t index = 0; index < tableCount; ++index) {
                state = state * 1664525u + 1013904223u;
                const float value = static_cast<float>(state >> 8) * (1.0f / 16777216.0f) - 0.5f;
                hostTable[index] = this->operation == Operation::Add ? Narrow<T>(0.0f) : Narrow<T>(value);
            }
            for (std::size_t index = 0; index < rowCount; ++index) {
                state = state * 1664525u + 1013904223u;
                hostInput[index] = Narrow<T>(static_cast<float>(state >> 8) * (1.0f / 16777216.0f) - 0.5f);
            }

            std::vector<int> hostIndices(kRows);
            for (int row = 0; row < kRows; ++row) hostIndices[row] = (row * 109 + 3) % kTableRows;

            this->table.CopyFromHost(hostTable.data(), hostTable.size());
            this->input.CopyFromHost(hostInput.data(), hostInput.size());
            this->indices.CopyFromHost(hostIndices.data(), hostIndices.size());
        }

        void Run(const hipStream_t stream) {
            switch (this->operation) {
                case Operation::Get:
                    ROCmKernels::Kernels::GetRowsForward(
                        this->table.Data(), this->indices.Data(), this->output.Data(), kRows, kColumns, ROCmKernels::Kernels::DTypeOf<T>::value, stream
                    );
                    break;
                case Operation::Set:
                    ROCmKernels::Kernels::SetRowsForward(
                        this->input.Data(), this->indices.Data(), this->table.Data(), kRows, kColumns, ROCmKernels::Kernels::DTypeOf<T>::value, stream
                    );
                    break;
                case Operation::Add:
                    ROCmKernels::Kernels::AddRowsForward(
                        this->input.Data(), this->indices.Data(), this->table.Data(), kRows, kColumns, ROCmKernels::Kernels::DTypeOf<T>::value, stream
                    );
                    break;
            }
        }

        Operation operation;
        ROCmKernels::DeviceBuffer<T> table;
        ROCmKernels::DeviceBuffer<T> input;
        ROCmKernels::DeviceBuffer<T> output;
        ROCmKernels::DeviceBuffer<int> indices;
    };

    template<typename T> ROCmKernels::Benchmark::Case MakeCase(const Operation operation) {
        const auto state = std::make_shared<State<T>>(operation);

        return ROCmKernels::Benchmark::Case{
            .name = std::string("RowGather ") + OperationName(operation) + " " + TypeName<T>() + " table 55765x256 rows 512",
            .setup = [state] { state->Initialize(); },
            .run = [state](hipStream_t stream) { state->Run(stream); },
            .bytesMoved = static_cast<std::uint64_t>(kRows) * kColumns * sizeof(T) * 2,
            .operations = 0,
        };
    }

    struct Registrar {
        Registrar() {
            using ROCmKernels::Benchmark::Register;

            Register(MakeCase<float>(Operation::Get));
            Register(MakeCase<__half>(Operation::Get));
            Register(MakeCase<float>(Operation::Set));
            Register(MakeCase<__half>(Operation::Set));
            Register(MakeCase<float>(Operation::Add));
        }
    };

    [[maybe_unused]] const Registrar kRegistrar;
} // namespace
