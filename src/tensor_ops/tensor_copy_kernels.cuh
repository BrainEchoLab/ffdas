#pragma once

#include <cuda_runtime.h>

#include "type_utils.h"
#include "error_checking.h"
#include "tensor.cuh"

/*
 * Given an nd array with possibly non-contiguous strides, force a copy with new
 * strides in each dimension. This kernel does not check for any overlapping 
 * memory.
 */
template<typename Tx, typename Ty>
__global__ void strided_copy_kernel(
    int size,
    int ndims, 
    const int *dims, 
    const int *x_strides,
    const Tx *x, 
    const int *y_strides,  // new strides for each dim (possibly non-contiguous)
    Ty *y
) {
    if (ndims < 1) 
        return;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size)
        return;

    int xidx = 0;
    int yidx = 0;

    int b = tid;

    for (int i = ndims-1; i >= 0; i--) {
        int xidx_multi = b % dims[i];
        xidx += xidx_multi * x_strides[i];

        yidx += xidx_multi * y_strides[i];

        b = b / dims[i];
    }

    y[yidx] = cast<Tx, Ty>(x[xidx]);
}
