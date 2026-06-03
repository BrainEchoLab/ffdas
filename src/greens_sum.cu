#include "greens_sum_impl.cuh"

#include <cuda_runtime.h>
#include <climits>

#include "greens_kernels.cuh"
#include "contiguous_copy_impl.cuh"
#include "tensor.cuh"
#include "error_checking.h"
#include "context.cuh"


namespace ffdas::detail {


// The kernel writes y in (samples, ny, batch_size) layout. After the kernel,
// a contiguous copy transposes this into the user's (batch_size, ny, samples)
// layout.
template<>
ffdas_error_t greens_sum_impl<half2, float2>(
    ffdas_context &handle,
    const float3 *xpos,
    const float *wavenums,
    const ffdas_tensor_desc &x_desc,
    const half2* x,
    const float3 *ypos,
    const ffdas_tensor_desc &y_desc,
    float2* y
) {
    if (x_desc.ndim() != 3)
        return FFDAS_ERROR_INVALID_DIMS;

    int64_t batch_size64 = x_desc.dims[0];
    int64_t channels64   = x_desc.dims[1];
    int64_t samples64    = x_desc.dims[2];

    if (batch_size64 > INT_MAX || channels64 > INT_MAX || samples64 > INT_MAX)
        return FFDAS_ERROR_INVALID_DIMS;

    int batch_size = static_cast<int>(batch_size64);
    int channels   = static_cast<int>(channels64);
    int samples    = static_cast<int>(samples64);

    if (y_desc.ndim() != 3 || y_desc.dims[0] != batch_size64 || y_desc.dims[2] != samples64)
        return FFDAS_ERROR_INVALID_DIMS;

    int ny = static_cast<int>(y_desc.dims[1]);

    device_ptr<float2> work(handle);
    FFDAS_CHECK(work.alloc(y_desc.nbytes()));

    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    constexpr int block_size = 128;
    constexpr int warps_per_block = block_size / 32;

    dim3 block_dim(block_size);
    dim3 grid_dim(
        (ny + (M * warps_per_block) - 1) / (M * warps_per_block),
        (batch_size + N - 1) / N,
        samples
    );

    greens_sum_kernel<warps_per_block, M, N, K><<<grid_dim, block_dim, 0, handle.stream>>>(
        samples, channels,
        xpos, wavenums, x,
        ny, ypos, work.get(),
        batch_size
    );

    CUDA_LAUNCH_CHECK();

    // kernel output is (batch, ny, samples) with strides (1, batch_size, batch_size*ny)
    ffdas_tensor_desc out_desc(
        {batch_size64, (int64_t)ny, samples64},
        {1LL, batch_size64, batch_size64 * (int64_t)ny},
        builtin_traits<float2>::ffdas_datatype
    );

    return contiguous_copy_impl<float2, float2>(handle, out_desc, work.get(), y);
}


// float2 input: downcast to half2, then call the half2 specialization.
template<>
ffdas_error_t greens_sum_impl<float2, float2>(
    ffdas_context &handle,
    const float3 *xpos,
    const float *wavenums,
    const ffdas_tensor_desc &x_desc,
    const float2* x,
    const float3 *ypos,
    const ffdas_tensor_desc &y_desc,
    float2* y
) {
    device_ptr<void> x_half(handle);
    FFDAS_CHECK(x_half.alloc(x_desc.nbytes() / 2));

    ffdas_error_t err = contiguous_copy_impl<float2, half2>(
        handle, x_desc, x, static_cast<half2*>(x_half.get()));

    if (err == FFDAS_SUCCESS) {
        err = greens_sum_impl<half2, float2>(
            handle, xpos, wavenums, x_desc,
            static_cast<half2*>(x_half.get()),
            ypos, y_desc, y);
    }

    return err;
}


}  // namespace ffdas::detail


ffdas_error_t ffdas_greens_sum(
    ffdas_handle_t handle,
    const float3 *xpos,
    const float *wavenums,
    ffdas_tensor_desc_t x_desc,
    const void* x,
    const float3 *ypos,
    ffdas_tensor_desc_t y_desc,
    void* y
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(y_desc);
    FFDAS_CHECK(handle->check_device());

    ffdas::detail::nvtx_range nvtx(*handle, "greens");

    const ffdas_tensor_desc &x_tensor = *x_desc;
    const ffdas_tensor_desc &y_tensor = *y_desc;

    switch (x_tensor.dtype) {
    case FFDAS_C_16F:
        switch (y_tensor.dtype) {
        case FFDAS_C_32F:
            return ffdas::detail::greens_sum_dispatch<FFDAS_C_16F, FFDAS_C_32F>(
                *handle, xpos, wavenums, x_tensor, x, ypos, y_tensor, y);
        default: break;
        }
        break;
    case FFDAS_C_32F:
        switch (y_tensor.dtype) {
        case FFDAS_C_32F:
            return ffdas::detail::greens_sum_dispatch<FFDAS_C_32F, FFDAS_C_32F>(
                *handle, xpos, wavenums, x_tensor, x, ypos, y_tensor, y);
        default: break;
        }
        break;
    default:
        break;
    }

    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}
