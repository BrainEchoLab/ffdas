#pragma once

#include <cuda_runtime.h>

#include "type_utils.h"
#include "error_checking.h"
#include "tensor.cuh"
#include "math/magic_division.cuh"


__constant__ uint32_t gather_radix[FFDAS_MAX_DIMS];  // for output
__constant__ magic_pair gather_radix_magic[FFDAS_MAX_DIMS];  // magic for dividing by gather_radix[i]
__constant__ int32_t gather_strides[FFDAS_MAX_DIMS];  // for input


template<typename Tx, typename Ty>
__global__ void gather_copy_kernel(
    int n,  // total number of elements in the output
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

    // b is the linear output index, idx_src is the offset in x
    uint32_t b = (uint32_t)tid;
    size_t idx_src = 0;
    uint32_t radix, coord, q;

    // peel off each coordinate in row-major order
    for(int i = 0; i < ndims-1; i++) {  // the last index hax radix 1
        q = magic_divide_unsigned(b, gather_radix_magic[i]);
        radix = gather_radix[i];

        b -= q * radix;
        coord = (i == axis ? index[q] : q);

        // accumulate into the source offset
        idx_src += size_t(coord) * size_t(gather_strides[i]);
    }

    // for the last dim, we have q = b, so we use b to index
    coord = ((ndims-1) == axis ? index[b] : b);
    idx_src += size_t(coord) * size_t(gather_strides[ndims-1]);

    y[tid] = cast<Tx, Ty>(__ldg(x + idx_src));
}
