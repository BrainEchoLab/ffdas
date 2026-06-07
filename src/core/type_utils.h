#ifndef FFDAS_TYPE_UTILS_H_
#define FFDAS_TYPE_UTILS_H_

#include "ffdas.h"

#include <type_traits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "error_checking.h"


template <typename T> struct builtin_traits;

template <> struct builtin_traits<__half> {
    using scalar_type = __half;
    using complex_type = __half2;

    typedef __half T;

    static __host__ __device__ T zero() { return __float2half(0.0f); }
    static __host__ __device__ T half() { return __float2half(0.5f); }
    static __host__ __device__ T one() { return __float2half(1.0f); }

    static __host__ __device__ bool is_zero(const __half &v) {
        return __half_as_ushort(v) == 0;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_R_16F;
    static constexpr cudaDataType cuda_datatype = CUDA_R_16F;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_16F;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<float> {
    using scalar_type = float;
    using complex_type = float2;

    typedef float T;

    static T zero() { return 0.0f; }
    static __host__ __device__ T half() { return 0.5f; }
    static __host__ __device__ T one() { return 1.0f; }

    static __host__ __device__ bool is_zero(const float &v) {
        return v == 0.0f;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_R_32F;
    static constexpr cudaDataType cuda_datatype = CUDA_R_32F;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_32F;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<double> {
    using scalar_type = double;
    using complex_type = double2;

    typedef double T;

    static __host__ __device__ T zero() { return 0.0; }
    static __host__ __device__ T half() { return 0.5; }
    static __host__ __device__ T one() { return 1.0; }

    static __host__ __device__ bool is_zero(const double &v) {
        return v == 0.0;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_R_64F;
    static constexpr cudaDataType cuda_datatype = CUDA_R_64F;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_64F;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<int> {
    using scalar_type = int;
    using complex_type = int2;

    typedef int T;

    static __host__ __device__ T zero() { return 0; }
    static __host__ __device__ T one() { return 1; }

    static __host__ __device__ bool is_zero(const int &v) {
        return v == 0;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_R_32I;
    static constexpr cudaDataType cuda_datatype = CUDA_R_32I;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_32I;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<short> {
    using scalar_type = short;
    using complex_type = short2;

    typedef short T;

    static __host__ __device__ T zero() { return 0; }
    static __host__ __device__ T one() { return 1; }

    static __host__ __device__ bool is_zero(const short &v) {
        return v == 0;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_R_16I;
    static constexpr cudaDataType cuda_datatype = CUDA_R_16I;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_16I;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<__half2> {
    using scalar_type = __half;
    using complex_type = __half2;

    typedef __half2 T;

    static __host__ __device__ T zero() { return __floats2half2_rn(0.0f, 0.0f); }
    static __host__ __device__ T half() { return __floats2half2_rn(0.5f, 0.0f); }
    static __host__ __device__ T one() { return __floats2half2_rn(1.0f, 0.0f); }

    static __host__ __device__ bool is_zero(const __half2 &v) {
        return __half_as_ushort(v.x) == 0 && __half_as_ushort(v.y) == 0;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_C_16F;
    static constexpr cudaDataType cuda_datatype = CUDA_C_16F;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_16F;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<float2> {
    using scalar_type = float;
    using complex_type = float2;

    typedef float2 T;

    static __host__ __device__ T zero() { return {0.0f, 0.0f}; }
    static __host__ __device__ T half() { return {0.5f, 0.0f}; }
    static __host__ __device__ T one() { return {1.0f, 0.0f}; }

    static __host__ __device__ bool is_zero(const float2 &v) {
        return v.x == 0.0f && v.y == 0.0f;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_C_32F;
    static constexpr cudaDataType cuda_datatype = CUDA_C_32F;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_32F;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<double2> {
    using scalar_type = double;
    using complex_type = double2;

    typedef double2 T;

    static __host__ __device__ T zero() { return {0.0, 0.0}; }
    static __host__ __device__ T half() { return {0.5, 0.0}; }
    static __host__ __device__ T one() { return {1.0, 0.0}; }

    static __host__ __device__ bool is_zero(const double2 &v) {
        return v.x == 0.0 && v.y == 0.0;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_C_64F;
    static constexpr cudaDataType cuda_datatype = CUDA_C_64F;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_64F;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<int2> {
    using scalar_type = int;
    using complex_type = int2;

    typedef int2 T;

    static __host__ __device__ T zero() { return {0, 0}; }
    static __host__ __device__ T one() { return {1, 0}; }

    static __host__ __device__ bool is_zero(const int2 &v) {
        return v.x == 0 && v.y == 0;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_C_32I;
    static constexpr cudaDataType cuda_datatype = CUDA_C_32I;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_32I;
    static constexpr size_t size = sizeof(T);
};

template <> struct builtin_traits<short2> {
    using scalar_type = short;
    using complex_type = short2;

    typedef short2 T;

    static __host__ __device__ T zero() { return {0, 0}; }
    static __host__ __device__ T one() { return {1, 0}; }

    static __host__ __device__ bool is_zero(const short2 &v) {
        return v.x == 0 && v.y == 0;
    }

    static constexpr ffdas_datatype_t ffdas_datatype = FFDAS_C_16I;
    static constexpr cudaDataType cuda_datatype = CUDA_C_16I;
    static constexpr ffdas_datatype_t complex_ffdas_datatype = FFDAS_C_16I;
    static constexpr size_t size = sizeof(T);
};

// const-qualified forwarding
template <> struct builtin_traits<const __half> : builtin_traits<__half> { using T = const __half; };
template <> struct builtin_traits<const float> : builtin_traits<float> { using T = const float; };
template <> struct builtin_traits<const double> : builtin_traits<double> { using T = const double; };
template <> struct builtin_traits<const short> : builtin_traits<short> { using T = const short; };
template <> struct builtin_traits<const int> : builtin_traits<int> { using T = const int; };
template <> struct builtin_traits<const __half2> : builtin_traits<__half2> { using T = const __half2; };
template <> struct builtin_traits<const float2> : builtin_traits<float2> { using T = const float2; };
template <> struct builtin_traits<const double2> : builtin_traits<double2> { using T = const double2; };
template <> struct builtin_traits<const int2> : builtin_traits<int2> { using T = const int2; };
template <> struct builtin_traits<const short2> : builtin_traits<short2> { using T = const short2; };


// Map ffdas_datatype_t constants to C++ types
template<ffdas_datatype_t> struct ffdas_traits;

template<> struct ffdas_traits<FFDAS_R_16F> { static constexpr bool complex = false; using type = __half; };
template<> struct ffdas_traits<FFDAS_C_16F> { static constexpr bool complex = true;  using type = __half2; };
template<> struct ffdas_traits<FFDAS_R_32F> { static constexpr bool complex = false; using type = float; };
template<> struct ffdas_traits<FFDAS_C_32F> { static constexpr bool complex = true;  using type = float2; };
template<> struct ffdas_traits<FFDAS_R_64F> { static constexpr bool complex = false; using type = double; };
template<> struct ffdas_traits<FFDAS_C_64F> { static constexpr bool complex = true;  using type = double2; };
template<> struct ffdas_traits<FFDAS_R_32I> { static constexpr bool complex = false; using type = int; };
template<> struct ffdas_traits<FFDAS_C_32I> { static constexpr bool complex = true;  using type = int2; };
template<> struct ffdas_traits<FFDAS_R_16I> { static constexpr bool complex = false; using type = short; };
template<> struct ffdas_traits<FFDAS_C_16I> { static constexpr bool complex = true;  using type = short2; };


// Type conversion utilities
template<typename T, typename S>
__host__ __device__ __inline__ S cast(const T &v) {
    return static_cast<S>(v);
}

template<> __host__ __device__ __inline__ __half cast(const short &v) { return __short2half_rn(v); }
template<> __host__ __device__ __inline__ __half2 cast(const __half &v) { return make_half2(v, __float2half(0.0f)); }
template<> __host__ __device__ __inline__ __half2 cast(const short2 &v) { return make_half2(__short2half_rn(v.x), __short2half_rn(v.y)); }
template<> __host__ __device__ __inline__ __half2 cast(const float2 &v) { return __floats2half2_rn(v.x, v.y); }
template<> __host__ __device__ __inline__ float2 cast(const float &v) { return make_float2(v, 0.0f); }
template<> __host__ __device__ __inline__ float2 cast(const short2 &v) { return make_float2(static_cast<float>(v.x), static_cast<float>(v.y)); }
template<> __host__ __device__ __inline__ float2 cast(const __half2 &v) { return __half22float2(v); }

// double casts
template<> __host__ __device__ __inline__ double2 cast(const float &v) { return make_double2(static_cast<double>(v), 0.0); }
template<> __host__ __device__ __inline__ double2 cast(const double &v) { return make_double2(v, 0.0); }
template<> __host__ __device__ __inline__ double2 cast(const float2 &v) { return make_double2(static_cast<double>(v.x), static_cast<double>(v.y)); }
template<> __host__ __device__ __inline__ float2 cast(const double2 &v) { return make_float2(static_cast<float>(v.x), static_cast<float>(v.y)); }

#endif  // FFDAS_TYPE_UTILS_H_
