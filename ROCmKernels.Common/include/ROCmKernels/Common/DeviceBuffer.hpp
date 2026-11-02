//
// Created by Dottik on 2/11/2026.
//

#pragma once

#include <ROCmKernels/Common/Error.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ROCmKernels {
    /// Owns a HIP device allocation and releases it on destruction.
    template <typename T> class DeviceBuffer {
        T *m_pData = nullptr;
        std::size_t m_elementCount = 0;

        void ValidateRange(std::size_t elementCount) const {
            if (elementCount > this->m_elementCount) {
                throw std::out_of_range("DeviceBuffer transfer exceeds the allocation.");
            }
        }

      public:
        DeviceBuffer() = default;

        explicit DeviceBuffer(std::size_t elementCount) { this->Resize(elementCount); }

        ~DeviceBuffer() { this->Release(); }

        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        DeviceBuffer(DeviceBuffer &&other) noexcept : m_pData(std::exchange(other.m_pData, nullptr)), m_elementCount(std::exchange(other.m_elementCount, 0)) {}

        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
            if (this != &other) {
                this->Release();
                this->m_pData = std::exchange(other.m_pData, nullptr);
                this->m_elementCount = std::exchange(other.m_elementCount, 0);
            }

            return *this;
        }

        /// Frees the current allocation and reserves room for elementCount elements.
        void Resize(std::size_t elementCount) {
            this->Release();

            if (elementCount == 0) {
                return;
            }

            ROCMKERNELS_HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&this->m_pData), elementCount * sizeof(T)));
            this->m_elementCount = elementCount;
        }

        void Release() {
            if (this->m_pData != nullptr) {
                ROCMKERNELS_HIP_CHECK(hipFree(this->m_pData));
                this->m_pData = nullptr;
            }

            this->m_elementCount = 0;
        }

        void CopyFromHost(const T *source, std::size_t elementCount, hipStream_t stream = nullptr) {
            this->ValidateRange(elementCount);

            if (elementCount == 0) {
                return;
            }

            const std::size_t byteCount = elementCount * sizeof(T);
            if (stream == nullptr) {
                ROCMKERNELS_HIP_CHECK(hipMemcpy(this->m_pData, source, byteCount, hipMemcpyHostToDevice));
            } else {
                ROCMKERNELS_HIP_CHECK(hipMemcpyAsync(this->m_pData, source, byteCount, hipMemcpyHostToDevice, stream));
            }
        }

        void CopyToHost(T *destination, std::size_t elementCount, hipStream_t stream = nullptr) const {
            this->ValidateRange(elementCount);

            if (elementCount == 0) {
                return;
            }

            const std::size_t byteCount = elementCount * sizeof(T);
            if (stream == nullptr) {
                ROCMKERNELS_HIP_CHECK(hipMemcpy(destination, this->m_pData, byteCount, hipMemcpyDeviceToHost));
            } else {
                ROCMKERNELS_HIP_CHECK(hipMemcpyAsync(destination, this->m_pData, byteCount, hipMemcpyDeviceToHost, stream));
            }
        }

        [[nodiscard]] T *Data() const noexcept { return this->m_pData; }
        [[nodiscard]] std::size_t Count() const noexcept { return this->m_elementCount; }
        [[nodiscard]] std::size_t ByteSize() const noexcept { return this->m_elementCount * sizeof(T); }
    };
} // namespace ROCmKernels
