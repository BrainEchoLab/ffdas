#ifndef FFDAS_MATH_VECTOR_H_
#define FFDAS_MATH_VECTOR_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Generic linear interpolation functions
template<typename type>
__device__ inline type lerp(const type &s0, const type &s1, const float &t);

template<typename type>
__device__ inline type lerp(const type &s0, const type &s1, const __half &t);

template<>
__device__ inline float lerp(const float &s0, const float &s1, const float &t) {
    return __fmaf_rn(t, s1 - s0, s0);
}

template<>
__device__ inline float lerp(const float &s0, const float &s1, const __half &t) {
    return __fmaf_rn(__half2float(t), s1 - s0, s0);
}

template<>
__device__ inline float2 lerp(const float2 &s0, const float2 &s1, const float &t) {
    float2 result;
    result.x = __fmaf_rn(t, s1.x - s0.x, s0.x);
    result.y = __fmaf_rn(t, s1.y - s0.y, s0.y);
    return result;
}

template<>
__device__ inline float2 lerp(const float2 &s0, const float2 &s1, const __half &t) {
    float t_flt = __half2float(t);
    float2 result;
    result.x = __fmaf_rn(t_flt, s1.x - s0.x, s0.x);
    result.y = __fmaf_rn(t_flt, s1.y - s0.y, s0.y);
    return result;
}

template<>
__device__ inline __half lerp(const __half &s0, const __half &s1, const float &t) {
    return __hfma(__float2half_rn(t), __hsub(s1, s0), s0);
}

template<>
__device__ inline __half lerp(const __half &s0, const __half &s1, const __half &t) {
    return __hfma(t, __hsub(s1, s0), s0);
}

template<>
__device__ inline __half2 lerp(const __half2 &s0, const __half2 &s1, const float &t) {
    __half2 t_hlf = __float2half2_rn(t);
    return __hfma2(t_hlf, __hsub2(s1, s0), s0);
}

template<>
__device__ inline __half2 lerp(const __half2 &s0, const __half2 &s1, const __half &t) {
    return __hfma2(__half2half2(t), __hsub2(s1, s0), s0);
}

template<>
__device__ inline double lerp(const double &s0, const double &s1, const float &t) {
    return fma((double)t, s1 - s0, s0);
}

template<>
__device__ inline double2 lerp(const double2 &s0, const double2 &s1, const float &t) {
    double td = (double)t;
    return make_double2(fma(td, s1.x - s0.x, s0.x), fma(td, s1.y - s0.y, s0.y));
}

// Generic min/max functions
template<typename type>
__inline__ __device__ type minimum(type x, type y) {
    return min(x, y);
}

template<>
__inline__ __device__ int3 minimum(int3 x, int3 y) {
    return make_int3(min(x.x, y.x), min(x.y, y.y), min(x.z, y.z));
}

template<>
__inline__ __device__ float3 minimum(float3 x, float3 y) {
    return make_float3(fminf(x.x, y.x), fminf(x.y, y.y), fminf(x.z, y.z));
}

template<typename type>
__inline__ __device__ type maximum(type x, type y) {
    return max(x, y);
}

template<>
__inline__ __device__ float2 maximum(float2 x, float2 y) {
    return make_float2(fmaxf(x.x, y.x), fmaxf(x.y, y.y));
}

template<>
__inline__ __device__ int3 maximum(int3 x, int3 y) {
    return make_int3(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z));
}

template<>
__inline__ __device__ float3 maximum(float3 x, float3 y) {
    return make_float3(fmaxf(x.x, y.x), fmaxf(x.y, y.y), fmaxf(x.z, y.z));
}

__device__ __inline__ float3 cross(float3 &a, float3 &b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

template<typename type>
__inline__ __device__ bool isequal(type x, type y) { return (x == y); }

template<> __inline__ __device__ bool isequal(float2 x, float2 y) { return (x.x == y.x) && (x.y == y.y); }
template<> __inline__ __device__ bool isequal(float3 x, float3 y) { return (x.x == y.x) && (x.y == y.y) && (x.z == y.z); }
template<> __inline__ __device__ bool isequal(int2 x, int2 y) { return (x.x == y.x) && (x.y == y.y); }
template<> __inline__ __device__ bool isequal(int3 x, int3 y) { return (x.x == y.x) && (x.y == y.y) && (x.z == y.z); }


// Complex multiply-add: z += x * y (complex multiplication for complex types)
template<typename Tx, typename Tz>
__device__ __inline__ Tz cmadd(const Tx &x, const Tx &y, const Tz &z);

template<> __device__ __inline__ __half cmadd(const __half &x, const __half &y, const __half &z) { return __hfma(x, y, z); }
template<> __device__ __inline__ float cmadd(const __half &x, const __half &y, const float &z) { return __half2float(__hmul(x, y)) + z; }
template<> __device__ __inline__ float cmadd(const float &x, const float &y, const float &z) { return __fmaf_rn(x, y, z); }
template<> __device__ __inline__ double cmadd(const double &x, const double &y, const double &z) { return fma(x, y, z); }

template<>
__device__ __inline__ __half2 cmadd(const __half2 &x, const __half2 &y, const __half2 &z) {
    return __hcmadd(x, y, z);
}

template<>
__device__ __inline__ float2 cmadd(const __half2 &x, const __half2 &y, const float2 &z) {
    __half2 ac_bd = __hmul2(x, y);
    __half2 b_swapped = __lowhigh2highlow(y);
    __half2 ad_bc = __hmul2(x, b_swapped);
    float real = __low2float(ac_bd) - __high2float(ac_bd);
    float imag = __low2float(ad_bc) + __high2float(ad_bc);
    return make_float2(z.x + real, z.y + imag);
}

template<>
__device__ __inline__ float2 cmadd(const float2 &x, const float2 &y, const float2 &z) {
    return cuCfmaf(x, y, z);
}

template<>
__device__ __inline__ double2 cmadd(const double2 &x, const double2 &y, const double2 &z) {
    return make_double2(
        fma(x.x, y.x, z.x) - x.y * y.y,
        fma(x.x, y.y, z.y) + x.y * y.x
    );
}


// In-place complex multiply-add
template<typename Tx, typename Tz>
__device__ __inline__ void cmadd_inplace(const Tx &x, const Tx &y, Tz &z);

template<> __device__ __inline__ void cmadd_inplace(const __half &x, const __half &y, __half &z) { z += __hmul(x, y); }
template<> __device__ __inline__ void cmadd_inplace(const __half &x, const __half &y, float &z) { z += __half2float(__hmul(x, y)); }
template<> __device__ __inline__ void cmadd_inplace(const float &x, const float &y, float &z) { z += x * y; }
template<> __device__ __inline__ void cmadd_inplace(const double &x, const double &y, double &z) { z += x * y; }

template<>
__device__ __inline__ void cmadd_inplace(const __half2 &x, const __half2 &y, __half2 &z) {
    z = __hcmadd(x, y, z);
}

template<>
__device__ __inline__ void cmadd_inplace(const __half2 &x, const __half2 &y, float2 &z) {
    __half2 ac_bd = __hmul2(x, y);
    z.x += __half2float(ac_bd.x - ac_bd.y);
    __half2 ad_bc = __hmul2(x, __lowhigh2highlow(y));
    z.y += __half2float(ad_bc.x + ad_bc.y);
}

template<>
__device__ __inline__ void cmadd_inplace(const float2 &x, const float2 &y, float2 &z) {
    z = cuCfmaf(x, y, z);
}

template<>
__device__ __inline__ void cmadd_inplace(const double2 &x, const double2 &y, double2 &z) {
    double re = x.x * y.x - x.y * y.y;
    double im = x.x * y.y + x.y * y.x;
    z.x += re;
    z.y += im;
}

#endif  // FFDAS_MATH_VECTOR_H_
