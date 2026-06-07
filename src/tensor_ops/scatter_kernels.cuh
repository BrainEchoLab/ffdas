#pragma once

#include <cuda_runtime.h>

#include "type_utils.h"
#include "error_checking.h"
#include "tensor.cuh"
#include "math/magic_division.cuh"


__constant__ uint32_t scatter_radix[FFDAS_MAX_DIMS];  // for input
__constant__ magic_pair scatter_radix_magic[FFDAS_MAX_DIMS];  // magic for dividing by scatter_radix[i]
__constant__ int32_t scatter_strides[FFDAS_MAX_DIMS];  // for output


template<typename Tx, typename Ty>
__global__ void scatter_copy_kernel(
    int n,  // total number of elements in the input
    int ndims, 
    int axis,
    const int * __restrict__ index,
    const Tx* __restrict__ x, 
    Ty* __restrict__ y
) {
    // As the output may have a different shape than the input, we work from
    // the output index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n)
        return;

    // b is the linear input index, idx_dst is the offset in y
    uint32_t b = (uint32_t)tid;
    size_t idx_dst = 0;
    uint32_t radix, coord, q;

    // peel off each coordinate in row-major order
    for(int i = 0; i < ndims-1; i++) {  // the last index hax radix 1
        q = magic_divide_unsigned(b, scatter_radix_magic[i]);
        radix = scatter_radix[i];

        b -= q * radix;
        coord = (i == axis ? index[q] : q);

        // accumulate into the source offset
        idx_dst += size_t(coord) * size_t(scatter_strides[i]);
    }

    // for the last dim, we have q = b, so we use b to index
    coord = ((ndims-1) == axis ? index[b] : b);
    idx_dst += size_t(coord) * size_t(scatter_strides[ndims-1]);

    y[idx_dst] = cast<Tx, Ty>(x[tid]);
}
