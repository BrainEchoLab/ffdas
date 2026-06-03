#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <vector_types.h>

#include "ffdas.h"
#include "error_checking.h"

// Helper functions for vector arithmetic
__device__ __forceinline__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ short2 operator+(const short2 &a, const short2 &b) {
    return make_short2(a.x + b.x, a.y + b.y);
}

// Helper functions for scalar multiplication
__device__ __forceinline__ float scalar_mul(float a, float b) {
    return a * b;
}

__device__ __forceinline__ float2 scalar_mul(float a, float2 b) {
    return make_float2(a * b.x, a * b.y);
}

__device__ __forceinline__ short scalar_mul(float a, short b) {
    return __float2int_rn(a * (float)b);
}

__device__ __forceinline__ short2 scalar_mul(float a, short2 b) {
    return make_short2(
        __float2int_rn(a * (float)b.x),
        __float2int_rn(a * (float)b.y)
    );
}

__device__ __forceinline__ double scalar_mul(float a, double b) {
    return (double)a * b;
}

__device__ __forceinline__ double2 scalar_mul(float a, double2 b) {
    return make_double2((double)a * b.x, (double)a * b.y);
}

__device__ __forceinline__ double2 operator+(const double2 &a, const double2 &b) {
    return make_double2(a.x + b.x, a.y + b.y);
}


template<typename T>
__global__ void interp_nearest_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int num_indices,
    const int* __restrict__ indices,
    int batch_size,
    int batch_stride,
    T fill_value
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y;

    if (tid >= num_indices || b >= batch_size)
        return;

    int xidx = indices[tid];
    int yidx = tid + b * num_indices;

    if (xidx == -1) {
        y[yidx] = fill_value;
        return;
    }

    y[yidx] = x[xidx + b * batch_stride];
}


template<typename T>
__global__ void interp_linear_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int num_indices,
    const int4* __restrict__ indices,
    const float4* __restrict__ weights,
    int batch_size,
    int batch_stride,
    T fill_value
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y;

    if (tid >= num_indices || b >= batch_size)
        return;

    int4 xidx = indices[tid];
    int yidx = tid + b * num_indices;

    if (xidx.x == -1) {
        y[yidx] = fill_value;
        return;
    }

    float4 wt = weights[tid];
    
    // Handle scalar multiplication properly for all types
    T val0 = x[xidx.x + b * batch_stride];
    T val1 = x[xidx.y + b * batch_stride];
    T val2 = x[xidx.z + b * batch_stride];
    T val3 = x[xidx.w + b * batch_stride];
    
    // Use custom scalar multiplication function
    y[yidx] = scalar_mul(wt.x, val0) + scalar_mul(wt.y, val1) + 
              scalar_mul(wt.z, val2) + scalar_mul(wt.w, val3);
}
