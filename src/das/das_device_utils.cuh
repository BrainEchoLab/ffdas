#pragma once

#include "type_utils.h"
#include "ffdas_math.cuh"


__device__ __forceinline__
float modff_int(float x, int *ipart) {
    *ipart = __float2int_rd(x);
    return x - __int2float_rn(*ipart);
}

__device__ __forceinline__
float clampf(float x, float amin, float amax) {
    return fmaxf(amin, fminf(amax, x));
}

__device__ __forceinline__
float distf(float3 x, float3 y, float3 *delta, float *rinv) {
    delta->x = x.x - y.x;
    delta->y = x.y - y.y;
    delta->z = x.z - y.z;
    float r_sq = delta->x * delta->x + delta->y * delta->y + delta->z * delta->z;
    *rinv = rsqrtf(r_sq);
    return r_sq * (*rinv);
}


template<typename T>
static __device__ __forceinline__ 
void accumulate_inplace(T* y, T acc, T beta) {
    if (builtin_traits<T>::is_zero(beta)) {
        *y = acc;
    } else {
        *y = cmadd(beta, *y, acc);
    }
}


template <typename T, int N> struct vector_load_type;
template <> struct vector_load_type<float, 1> { using type = float; };
template <> struct vector_load_type<float, 2> { using type = float2; };
template <> struct vector_load_type<float, 4> { using type = float4; };
template <> struct vector_load_type<__half, 1> { using type = __half; };
template <> struct vector_load_type<__half, 2> { using type = __half2; };
template <> struct vector_load_type<__half, 4> { using type = float2; };
template <> struct vector_load_type<__half, 8> { using type = float4; };
template <> struct vector_load_type<float2, 1> { using type = float2; };
template <> struct vector_load_type<float2, 2> { using type = float4; };
template <> struct vector_load_type<__half2, 1> { using type = __half2; };
template <> struct vector_load_type<__half2, 2> { using type = float2; };
template <> struct vector_load_type<__half2, 4> { using type = float4; };


template <typename T, int N>
__device__ __forceinline__ void vector_load(
    const T* __restrict__ x,
    T *out
) {
    using vector_type = typename vector_load_type<T, N>::type;
    vector_type tmp = *reinterpret_cast<const vector_type*>(x);
    memcpy(out, &tmp, sizeof(tmp));
}
