#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "ffdas.h"
#include "context.cuh"
#include "type_utils.h"
#include "tensor.cuh"
#include "error_checking.h"
#include "math/magic_division.cuh"

#include "scatter_kernels.cuh"


namespace ffdas::detail {

template<typename Tx, typename Ty>
ffdas_error_t scatter_impl(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc, const Tx* x,
    const ffdas_tensor_desc &out_desc, Ty* out,
    int mode,
    const int *indices
) {
    size_t ndim = x_desc.ndim();

    if (out_desc.ndim() != ndim)
        return FFDAS_ERROR_INVALID_DIMS;
    if (mode < 0 || mode >= ndim)
        return FFDAS_ERROR_INVALID_ARGUMENT;
    if (!out_desc.is_contiguous())
        return FFDAS_ERROR_INVALID_DIMS;
    if (!can_use_int32_indexing(x_desc) || !can_use_int32_indexing(out_desc))
        return FFDAS_ERROR_INVALID_DIMS;

    std::vector<int> h_dims(ndim);
    std::vector<int> h_strides(ndim);
    for(int i = 0; i < ndim; i++){
        h_dims[i] = static_cast<int>(x_desc.dims[i]);
        h_strides[i] = static_cast<int>(out_desc.strides[i]);
    }

    // Decompose the input's linear index into nd coordinates on the GPU
    // using integer-free division. radix[i] is the product of dims[i+1..ndim-1],
    // so dividing the linear index by radix[i] peels off coordinate i.
    std::vector<unsigned> radix(ndim);
    radix[ndim-1] = 1;

    for(int i = ndim-2; i >= 0; i--) {
        radix[i] = radix[i+1] * h_dims[i+1];
    }

    std::vector<magic_pair> radix_magic(ndim);
    for(int i = 0; i < ndim; i++){
        radix_magic[i] = compute_magic_pair_unsigned(radix[i]);
    }

    CUDA_CHECK(cudaMemcpyToSymbolAsync(scatter_strides, h_strides.data(), ndim*sizeof(int), 0, cudaMemcpyHostToDevice, handle.stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(scatter_radix, radix.data(), ndim*sizeof(unsigned), 0, cudaMemcpyHostToDevice, handle.stream));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(scatter_radix_magic, radix_magic.data(), ndim*sizeof(magic_pair), 0, cudaMemcpyHostToDevice, handle.stream));

    dim3 block_dim(256);
    dim3 grid_dim((x_desc.numel() + block_dim.x - 1) / block_dim.x);

    scatter_copy_kernel<Tx, Ty><<<grid_dim, block_dim, 0, handle.stream>>>((int)x_desc.numel(), (int)ndim, mode, indices, x, out);

    CUDA_LAUNCH_CHECK();

    return FFDAS_SUCCESS;
}


template<ffdas_datatype_t Tx_t, ffdas_datatype_t Ty_t>
ffdas_error_t ffdas_scatter_dispatch(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc, const void* x,
    const ffdas_tensor_desc &out_desc, void* out,
    int mode,
    const int *indices
) {
    using Tx = typename ffdas_traits<Tx_t>::type;
    using Ty = typename ffdas_traits<Ty_t>::type;

    return scatter_impl<Tx, Ty>(handle, x_desc, static_cast<const Tx*>(x), out_desc, static_cast<Ty*>(out), mode, indices);
}

}  // namespace ffdas::detail
