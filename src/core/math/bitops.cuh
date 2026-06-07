#ifndef FFDAS_MATH_BITOPS_H_
#define FFDAS_MATH_BITOPS_H_

#include <cuda_runtime.h>
#include "core.cuh"

// Check whether an integer is an exact power of two
template<typename type>
__device__ __inline__ bool __ispow2(type n) {
    return n > 0 && __popc(n) == 1;
}

template<typename type>
inline bool ispow2(type n) {
    return n > 0 && popc(n) == 1;
}

// Compute log2(n) of an integer power of two
template<typename type>
__device__ __inline__ int __log2pow2(type n) {
    return __ffs(n) - 1;
}

template<typename type>
inline int log2pow2(type n) {
    return ffs(n) - 1;
}

// Find the next power of two
template<typename type>
__device__ __inline__ type __nextpow2(type n) {
    if (n == 0)
        return 1;
    if (__ispow2(n))
        return n;
    
    return 1 << (32 - __clz(n));
}

template<typename type>
inline type nextpow2(type n) {
    if (n == 0)
        return 1;
    if (ispow2(n))
        return n;
    
    return 1 << (32 - clz(n));
}

// Multiply n by a power of two efficiently using bit shifting
template<typename type>
__host__ __device__ __inline__ type mulpow2(type n, int pow) {
    return n << pow;
}

template<>
__host__ __device__ __inline__ uint2 mulpow2(uint2 n, int pow) {
    return make_uint2(
        mulpow2<unsigned int>(n.x, pow),
        mulpow2<unsigned int>(n.y, pow)
    );
}

template<>
__host__ __device__ __inline__ uint3 mulpow2(uint3 n, int pow) {
    return make_uint3(
        mulpow2<unsigned int>(n.x, pow),
        mulpow2<unsigned int>(n.y, pow),
        mulpow2<unsigned int>(n.z, pow)
    );
}

// Ceiling division of n by a power of two
template<typename type>
__host__ __device__ __inline__ type ceildivpow2(type n, int pow) {
    return (n + (1 << pow) - 1) >> pow;
}

template<>
__host__ __device__ __inline__ uint2 ceildivpow2(uint2 n, int pow) {
    return make_uint2(
        ceildivpow2<unsigned int>(n.x, pow),
        ceildivpow2<unsigned int>(n.y, pow)
    );
}

template<>
__host__ __device__ __inline__ uint3 ceildivpow2(uint3 n, int pow) {
    return make_uint3(
        ceildivpow2<unsigned int>(n.x, pow),
        ceildivpow2<unsigned int>(n.y, pow),
        ceildivpow2<unsigned int>(n.z, pow)
    );
}

// Efficiently compute n / (2^pow) and n % (2^pow) using bitwise operations
template<typename type>
__host__ __device__ __inline__ void divmodpow2(type n, int pow, type &d, type &m) {
    d = n >> pow;
    m = n & ((1 << pow) - 1);
}

template<>
__host__ __device__ __inline__ void divmodpow2(uint2 n, int pow, uint2 &d, uint2 &m) {
    divmodpow2<unsigned int>(n.x, pow, d.x, m.x);
    divmodpow2<unsigned int>(n.y, pow, d.y, m.y);
}

template<>
__host__ __device__ __inline__ void divmodpow2(uint3 n, int pow, uint3 &d, uint3 &m) {
    divmodpow2<unsigned int>(n.x, pow, d.x, m.x);
    divmodpow2<unsigned int>(n.y, pow, d.y, m.y);
    divmodpow2<unsigned int>(n.z, pow, d.z, m.z);
}

#endif  // FFDAS_MATH_BITOPS_H_