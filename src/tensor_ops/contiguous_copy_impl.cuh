#pragma once

#include <cuda_runtime.h>

#include "ffdas.h"
#include "context.cuh"
#include "type_utils.h"
#include "tensor.cuh"
#include "error_checking.h"

#include "tensor_copy_kernels.cuh"


namespace ffdas::detail {

template<typename Tx, typename Ty>
ffdas_error_t contiguous_copy_impl(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc, const Tx* x,
    Ty* out
) {
    if (!ffdas::detail::can_use_int32_indexing(x_desc))
        return FFDAS_ERROR_INVALID_DIMS;

    ffdas::detail::nvtx_range nvtx_cast(handle, "contiguous_copy_impl");

    size_t ndim = x_desc.ndim();
    std::vector<int64_t> strides = make_contiguous_strides(x_desc.dims);

    std::vector<int> h_dims(ndim), h_x_strides(ndim), h_y_strides(ndim);
    for (size_t i = 0; i < ndim; i++) {
        h_dims[i] = static_cast<int>(x_desc.dims[i]);
        h_x_strides[i] = static_cast<int>(x_desc.strides[i]);
        h_y_strides[i] = static_cast<int>(strides[i]);
    }
    
    ffdas::detail::device_ptr<int> d_dims(handle);
    ffdas::detail::device_ptr<int> d_x_strides(handle);
    ffdas::detail::device_ptr<int> d_y_strides(handle);

    FFDAS_CHECK(d_dims.alloc(ndim * sizeof(int)));
    FFDAS_CHECK(d_x_strides.alloc(ndim * sizeof(int)));
    FFDAS_CHECK(d_y_strides.alloc(ndim * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_dims.get(), h_dims.data(), ndim * sizeof(int), cudaMemcpyHostToDevice, handle.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x_strides.get(), h_x_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice, handle.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y_strides.get(), h_y_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice, handle.stream));

    dim3 block_dim(512);
    dim3 grid_dim((x_desc.numel() + block_dim.x - 1) / block_dim.x);

    strided_copy_kernel<Tx, Ty><<<grid_dim, block_dim, 0, handle.stream>>>((int)x_desc.numel(), (int)ndim, d_dims.get(), d_x_strides.get(), x, d_y_strides.get(), out);

    CUDA_LAUNCH_CHECK();

    return FFDAS_SUCCESS;
}


template<ffdas_datatype_t Tx_t, ffdas_datatype_t Ty_t>
ffdas_error_t contiguous_copy_dispatch(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc, const void* x,
    const ffdas_tensor_desc &out_desc, void* out
) {
    using Tx = typename ffdas_traits<Tx_t>::type;
    using Ty = typename ffdas_traits<Ty_t>::type;

    return contiguous_copy_impl<Tx, Ty>(handle, x_desc, static_cast<const Tx*>(x), static_cast<Ty*>(out));
}


template ffdas_error_t contiguous_copy_impl<short, short>(ffdas_context&, const ffdas_tensor_desc&, const short*, short*);
template ffdas_error_t contiguous_copy_impl<short, __half>(ffdas_context&, const ffdas_tensor_desc&, const short*, __half*);
template ffdas_error_t contiguous_copy_impl<short, float>(ffdas_context&, const ffdas_tensor_desc&, const short*, float*);
template ffdas_error_t contiguous_copy_impl<__half, __half>(ffdas_context&, const ffdas_tensor_desc&, const __half*, __half*);
template ffdas_error_t contiguous_copy_impl<__half, float>(ffdas_context&, const ffdas_tensor_desc&, const __half*, float*);
template ffdas_error_t contiguous_copy_impl<float, __half>(ffdas_context&, const ffdas_tensor_desc&, const float*, __half*);
template ffdas_error_t contiguous_copy_impl<float, float>(ffdas_context&, const ffdas_tensor_desc&, const float*, float*);
template ffdas_error_t contiguous_copy_impl<short2, short2>(ffdas_context&, const ffdas_tensor_desc&, const short2*, short2*);
template ffdas_error_t contiguous_copy_impl<short2, __half2>(ffdas_context&, const ffdas_tensor_desc&, const short2*, __half2*);
template ffdas_error_t contiguous_copy_impl<short2, float2>(ffdas_context&, const ffdas_tensor_desc&, const short2*, float2*);
template ffdas_error_t contiguous_copy_impl<__half2, __half2>(ffdas_context&, const ffdas_tensor_desc&, const __half2*, __half2*);
template ffdas_error_t contiguous_copy_impl<__half2, float2>(ffdas_context&, const ffdas_tensor_desc&, const __half2*, float2*);
template ffdas_error_t contiguous_copy_impl<float2, __half2>(ffdas_context&, const ffdas_tensor_desc&, const float2*, __half2*);
template ffdas_error_t contiguous_copy_impl<float2, float2>(ffdas_context&, const ffdas_tensor_desc&, const float2*, float2*);
template ffdas_error_t contiguous_copy_impl<double, double>(ffdas_context&, const ffdas_tensor_desc&, const double*, double*);
template ffdas_error_t contiguous_copy_impl<double, float>(ffdas_context&, const ffdas_tensor_desc&, const double*, float*);
template ffdas_error_t contiguous_copy_impl<float, double>(ffdas_context&, const ffdas_tensor_desc&, const float*, double*);
template ffdas_error_t contiguous_copy_impl<double2, double2>(ffdas_context&, const ffdas_tensor_desc&, const double2*, double2*);
template ffdas_error_t contiguous_copy_impl<double2, float2>(ffdas_context&, const ffdas_tensor_desc&, const double2*, float2*);
template ffdas_error_t contiguous_copy_impl<float2, double2>(ffdas_context&, const ffdas_tensor_desc&, const float2*, double2*);

}  // namespace ffdas::detail
