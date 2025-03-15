#ifndef CUDA_UTILS_MINIMAL_CUH
#define CUDA_UTILS_MINIMAL_CUH

#include "cuda_common.h"
#include <cuda_runtime.h>
#if defined(ENABLE_FP16)
#include <cuda_fp16.h>
#elif defined(ENABLE_BF16)
#include <cuda_bf16.h>
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure for 128-bit loads/stores
template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

// Short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// Load/store functions for Packed128
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}

// Coalesced load for better memory access patterns
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return load128(address);
}

template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}

// Stochastic rounding function (only needed if using BF16 or FP16)
#if !defined(ENABLE_FP32)
// Simplified version - replace with appropriate implementation for your needs
__device__ inline void stochastic_rounding(float value, floatX* output, unsigned int seed) {
    *output = (floatX)value;
}
#endif

#endif // CUDA_UTILS_MINIMAL_CUH
