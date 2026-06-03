#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>


template<typename T>
struct SampleWeight {
    const float power;
    const float mult;

    __device__ T
    operator()(float rad, float ct, float scl) const;
};


template<>
struct SampleWeight<float> {
    const float power;
    const float mult;

    __device__ float
    operator()(float rad, float ct, float scl) const {
        return (power != 1.0f) ? __powf(fabsf(ct), power)*scl : fabsf(ct)*scl;
    }
};


template<>
struct SampleWeight<__half> {
    const float power;
    const float mult;

    __device__ __half
    operator()(float rad, float ct, float scl) const {
        return __float2half_rn((power != 1.0f) ? __powf(fabsf(ct), power)*scl : fabsf(ct)*scl);
    }
};


template<>
struct SampleWeight<float2> {
    const float power;
    const float mult;

    __device__ float2
    operator()(float rad, float ct, float scl) const {
        float mag = (power != 1.0f) ? __powf(fabsf(ct), power)*scl : fabsf(ct)*scl;
        float s, c;
        __sincosf(mult * rad, &s, &c);
        return make_float2(c * mag, s * mag);
    }
};


template<>
struct SampleWeight<__half2> {
    const float power;
    const float mult;

    __device__ __half2
    operator()(float rad, float ct, float scl) const {
        float mag = (power != 1.0f) ? __powf(fabsf(ct), power)*scl : fabsf(ct)*scl;
        float s, c;
        __sincosf(mult * rad, &s, &c);
        return __floats2half2_rn(c*mag, s*mag);
    }
};
