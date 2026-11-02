#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <type_traits>

namespace ROCmKernels::Kernels {
    /// Storage type of a kernel argument. Accumulation is always fp32.
    enum class DType : unsigned char { Float32, Float16 };

    template<typename T> struct DTypeOf;
    template<> struct DTypeOf<float> {
        static constexpr DType value = DType::Float32;
    };
    template<> struct DTypeOf<__half> {
        static constexpr DType value = DType::Float16;
    };

    template<typename T> __device__ __forceinline__ float LoadScalar(const T *pointer) {
        if constexpr (std::is_same_v<T, __half>) return __half2float(*pointer);
        else return *pointer;
    }

    template<typename T> __device__ __forceinline__ void StoreScalar(T *pointer, const float value) {
        if constexpr (std::is_same_v<T, __half>) *pointer = __float2half_rn(value);
        else *pointer = value;
    }

    template<typename T> __device__ __forceinline__ float2 LoadPair(const T *pointer) {
        if constexpr (std::is_same_v<T, __half>) {
            const __half2 value = *reinterpret_cast<const __half2 *>(pointer);
            return __half22float2(value);
        } else {
            return *reinterpret_cast<const float2 *>(pointer);
        }
    }

    template<typename T> __device__ __forceinline__ void StorePair(T *pointer, const float2 value) {
        if constexpr (std::is_same_v<T, __half>) *reinterpret_cast<__half2 *>(pointer) = __floats2half2_rn(value.x, value.y);
        else *reinterpret_cast<float2 *>(pointer) = value;
    }
} // namespace ROCmKernels::Kernels
