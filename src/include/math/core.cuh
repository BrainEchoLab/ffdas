#ifndef FFDAS_MATH_CORE_H_
#define FFDAS_MATH_CORE_H_

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// Host/device bit manipulation functions
#if defined(__GNUC__) || defined(__clang__)
static int ffs(unsigned int x) {
    return __builtin_ffs(x); // GCC/Clang built-in
}
#elif defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
static int ffs(unsigned int x) {
    unsigned long index;
    if (_BitScanForward(&index, x)) {
        return static_cast<int>(index + 1); // _BitScanForward returns 0-based index
    }
    return 0; // If x is 0
}
#else
#error "Unsupported compiler"
#endif

#if defined(__GNUC__) || defined(__clang__)
static int popc(unsigned int x) {
    return __builtin_popcount(x);  // GCC/Clang built-in
}
#elif defined(_MSC_VER)
#include <nmmintrin.h> // For _mm_popcnt_u32
static int popc(unsigned int x) {
    return static_cast<int>(_mm_popcnt_u32(x));
}
#else
#error "Unsupported compiler"
#endif


#if defined(__GNUC__) || defined(__clang__)
static int ctz(unsigned int x) {
    return __builtin_ctz(x);  // GCC/Clang built-in
}
#elif defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
static int ctz(unsigned int x) {
    unsigned long index;
    _BitScanForward(&index, x);
    return static_cast<int>(index);
}
#else
#error "Unsupported compiler"
#endif


#if defined(__GNUC__) || defined(__clang__)
static int clz(unsigned int x) {
    return __builtin_clz(x);  // GCC/Clang built-in
}
#elif defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
static int clz(unsigned int x) {
    unsigned long index;
    if (_BitScanReverse(&index, x)) {
        return static_cast<int>(31 - index); // _BitScanReverse returns 0-based index
    }
    return 32; // If x is 0, all bits are leading zeros
}
#else
#error "Unsupported compiler"
#endif

// Generic multiplication functions
template<typename type>
__inline__ __device__ type mul(type x, type y) {
    return x * y;
}

template<>
__inline__ __device__ float2 mul(float2 x, float2 y) {
    return cuCmulf(x, y);
}

template<>
__inline__ __device__ __half mul(__half x, __half y) {
    return __hmul(x, y);
}

template<>
__inline__ __device__ __half2 mul(__half2 x, __half2 y) {
    __half2 prod;
    prod = make_half2(__hsub(__hmul(x.x, y.x),
                      __hmul(x.y, y.y)),
                      __hadd(__hmul(x.x, y.y),
                      __hmul(x.y, x.x)));
    return prod;
}

template<>
__inline__ __device__ double2 mul(double2 x, double2 y) {
    return make_double2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}

// Generic conjugation functions
template<typename type>
__device__ __inline__ type conj(type x) {
    return x;
}

template<>
__device__ __inline__ short2 conj(short2 x) {
    return make_short2(x.x, -x.y);
}

template<>
__device__ __inline__ float2 conj(float2 x) {
    return make_float2(x.x, -x.y);
}

template<>
__device__ __inline__ __half2 conj(__half2 x) {
    return make_half2(x.x, -x.y);
}

template<>
__device__ __inline__ double2 conj(double2 x) {
    return make_double2(x.x, -x.y);
}

// Generic addition functions
template<typename type>
__inline__ __device__ type add(type x, type y) {
    return x + y;
}

template<>
__inline__ __device__ float2 add(float2 x, float2 y) {
    return cuCaddf(x, y);
}

template<>
__inline__ __device__ __half add(__half x, __half y) {
    return __hadd(x, y);
}

template<>
__inline__ __device__ __half2 add(__half2 x, __half2 y) {
    return __hadd2(x, y);
}

template<>
__inline__ __device__ double2 add(double2 x, double2 y) {
    return make_double2(x.x + y.x, x.y + y.y);
}

// Generic (fused) multiply-add functions
template<typename xy_type, typename z_type>
__device__ __inline__ z_type muladd(xy_type x, xy_type y, z_type z);

template<>
__device__ __inline__ float muladd(float x, float y, float z) {
    return __fmaf_rn(x, y, z);
}

template<>
__device__ __inline__ float muladd(__half x, __half y, float z) {
    return __half2float(__hmul(x, y)) + z;  
}

template<>
__device__ __inline__ __half muladd(__half x, __half y, __half z) {
    return __hfma(x, y, z);
}

template<>
__device__ __inline__ float2 muladd(float2 x, float2 y, float2 z) {
    return cuCfmaf(x, y, z);
}

template<>
__device__ __inline__ float2 muladd(__half2 x, __half2 y, float2 z) {
    return cuCaddf(__half22float2(__hmul2(x, y)), z);
}

template<>
__device__ __inline__ __half2 muladd(__half2 x, __half2 y, __half2 z) {
    return __hfma2(x, y, z);
}

template<>
__device__ __inline__ double muladd(double x, double y, double z) {
    return fma(x, y, z);
}

template<>
__device__ __inline__ double2 muladd(double2 x, double2 y, double2 z) {
    return make_double2(
        fma(x.x, y.x, z.x) - x.y * y.y,
        fma(x.x, y.y, z.y) + x.y * y.x
    );
}

#endif  // FFDAS_MATH_CORE_H_