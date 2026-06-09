#include "greens_impl.cuh"

#include <climits>
#include <vector>

#include <cuda_runtime.h>

#include "ffdas.h"
#include "greens_kernels.cuh"
#include "contiguous_copy_impl.cuh"
#include "tensor.cuh"
#include "error_checking.h"
#include "context.cuh"


namespace ffdas::detail {

template<typename Tx, typename Ty>
ffdas_error_t greens_launch_sm53(
    ffdas_context &handle,
    int64_t samples,
    int64_t channels,
    const float3 *srcpos,
    const float *wavenums,
    const Tx *x,
    int64_t ndst, 
    const float3 *dstpos,
    Ty *out,
    int64_t batch_size
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}

template<typename Tx, typename Ty>
ffdas_error_t greens_launch_sm70(
    ffdas_context &handle,
    int64_t samples,
    int64_t channels,
    const float3 *srcpos,
    const float *wavenums,
    const Tx *x,
    int64_t ndst, 
    const float3 *dstpos,
    Ty *out,
    int64_t batch_size
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}


template<>
ffdas_error_t greens_launch_sm53<float2, float2>(
    ffdas_context &handle,
    int64_t samples,
    int64_t channels,
    const float3 *srcpos,
    const float *wavenums,
    const float2 *x,
    int64_t ndst, 
    const float3 *dstpos,
    float2 *out,
    int64_t batch_size
) {
    device_ptr<float2> work(handle);
    FFDAS_CHECK(work.alloc(batch_size * ndst * samples * sizeof(float2)));

    constexpr int TILE_DST = 4;
    constexpr int TILE_BATCH = 32;

    dim3 block_dim(TILE_DST * TILE_BATCH);
    dim3 grid_dim(
        (ndst + TILE_DST - 1) / TILE_DST,
        (batch_size + TILE_BATCH - 1) / TILE_BATCH,
        samples
    );

    greens_kernel_sm53<float2, TILE_DST, TILE_BATCH><<<grid_dim, block_dim, 0, handle.stream>>>(
        static_cast<int>(samples), 
        static_cast<int>(channels),
        srcpos, 
        wavenums, 
        x,
        static_cast<int>(ndst), 
        dstpos, 
        work.get(),
        batch_size
    );

    CUDA_LAUNCH_CHECK();

    ffdas_tensor_desc work_desc(
        {batch_size, ndst, samples},
        {1LL, batch_size, batch_size * ndst},
        builtin_traits<float2>::ffdas_datatype
    );

    return contiguous_copy_impl<float2, float2>(handle, work_desc, work.get(), out);
}


template<>
ffdas_error_t greens_launch_sm53<half2, float2>(
    ffdas_context &handle,
    int64_t samples,
    int64_t channels,
    const float3 *srcpos,
    const float *wavenums,
    const half2 *x,
    int64_t ndst, 
    const float3 *dstpos,
    float2 *out,
    int64_t batch_size
) {
    device_ptr<float2> work(handle);
    FFDAS_CHECK(work.alloc(batch_size * ndst * samples * sizeof(float2)));

    constexpr int TILE_DST = 4;
    constexpr int TILE_BATCH = 32;

    dim3 block_dim(TILE_DST * TILE_BATCH);
    dim3 grid_dim(
        (ndst + TILE_DST - 1) / TILE_DST,
        (batch_size + TILE_BATCH - 1) / TILE_BATCH,
        samples
    );

    greens_kernel_sm53<half2, TILE_DST, TILE_BATCH><<<grid_dim, block_dim, 0, handle.stream>>>(
        static_cast<int>(samples), 
        static_cast<int>(channels),
        srcpos, 
        wavenums, 
        x,
        static_cast<int>(ndst), 
        dstpos, 
        work.get(),
        batch_size
    );

    CUDA_LAUNCH_CHECK();

    ffdas_tensor_desc work_desc(
        {batch_size, ndst, samples},
        {1LL, batch_size, batch_size * ndst},
        builtin_traits<float2>::ffdas_datatype
    );

    return contiguous_copy_impl<float2, float2>(handle, work_desc, work.get(), out);
}

// The kernel writes out in (samples, ndst, batch_size) layout. After the kernel,
// a contiguous copy transposes this into the user's (batch_size, ndst, samples)
// layout.
template<>
ffdas_error_t greens_launch_sm70<half2, float2>(
    ffdas_context &handle,
    int64_t samples,
    int64_t channels,
    const float3 *srcpos,
    const float *wavenums,
    const half2 *x,
    int64_t ndst, 
    const float3 *dstpos,
    float2 *out,
    int64_t batch_size
) {
    device_ptr<float2> work(handle);
    FFDAS_CHECK(work.alloc(out_desc.nbytes()));

    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    constexpr int block_size = 128;
    constexpr int warps_per_block = block_size / 32;

    dim3 block_dim(block_size);
    dim3 grid_dim(
        (ndst + (M * warps_per_block) - 1) / (M * warps_per_block),
        (batch_size + N - 1) / N,
        samples
    );

    greens_kernel<warps_per_block, M, N, K><<<grid_dim, block_dim, 0, handle.stream>>>(
        static_cast<int>(samples), 
        static_cast<int>(channels),
        srcpos, 
        wavenums, 
        x,
        static_cast<int>(ndst), 
        dstpos, 
        work.get(),
        static_cast<int>(batch_size)
    );

    CUDA_LAUNCH_CHECK();

    // kernel output is (batch, ndst, samples) with strides (1, batch_size, batch_size*ndst)
    ffdas_tensor_desc work_desc(
        {batch_size, ndst, samples},
        {1LL, batch_size, batch_size * ndst},
        builtin_traits<float2>::ffdas_datatype
    );

    return contiguous_copy_impl<float2, float2>(handle, work_desc, work.get(), out);
}


// float2 input: on SM 70+ downcast to half2 for WMMA, otherwise use the
// scalar fallback directly (avoids a pointless float2->half2->float2 round-trip).
template<>
ffdas_error_t greens_launch_sm70<float2, float2>(
    ffdas_context &handle,
    int64_t samples,
    int64_t channels,
    const float3 *srcpos,
    const float *wavenums,
    const half2 *x,
    int64_t ndst, 
    const float3 *dstpos,
    float2 *out,
    int64_t batch_size
) {
    // reconstruct tensor descriptor for x to pass to contiguous_copy
    std::vector<int64_t> dims = {
        batch_size,
        channels,
        samples
    };
    std::vector<int64_t> strides = {
        channels * samples,
        samples,
        1
    };
    ffdas_tensor_desc x_desc(dims, strides, builtin_traits<float2>::ffdas_datatype);

    device_ptr<half2*> x_half(handle);
    FFDAS_CHECK(x_half.alloc(x_desc.nbytes() / 2));

    ffdas_error_t err = contiguous_copy_impl<float2, half2>(
        handle, 
        x_desc, 
        x, 
        x_half.get()
    );

    if (err != FFDAS_SUCCESS)
        return err;

    return greens_launch_sm70<half2, float2>(
        handle, 
        samples,
        channels,
        srcpos, 
        wavenums, 
        x_half.get(),
        ndst,
        dstpos, 
        out,
        batch_size
    );
}

template<typename Tx, typename Ty>
ffdas_error_t greens_impl(
    ffdas_context &handle,
    const float3 *srcpos,
    const float *wavenums,
    const ffdas_tensor_desc &x_desc,
    const Tx* x,
    const float3 *dstpos,
    const ffdas_tensor_desc &out_desc,
    Ty* out
) {
    if (x_desc.ndim() != 3)
        return FFDAS_ERROR_INVALID_DIMS;

    int64_t batch_size = x_desc.dims[0];
    int64_t channels = x_desc.dims[1];
    int64_t samples = x_desc.dims[2];
    int64_t ndst = out_desc.dims[1];

    if (out_desc.ndim() != 3 || out_desc.dims[0] != batch_size || out_desc.dims[2] != samples)
        return FFDAS_ERROR_INVALID_DIMS;

    if (batch_size > INT_MAX || channels > INT_MAX || samples > INT_MAX || ndst > INT_MAX)
        return FFDAS_ERROR_DIMS_TOO_LARGE;

    if (!x_desc.is_contiguous() || !out_desc.is_contiguous)
        return FFDAS_ERROR_NON_CONTIGUOUS;

    if (handle.arch_code < 700) {
        return greens_launch_sm53<Tx, Ty>(
            handle, 
            samples,
            channels, 
            srcpos, 
            wavenums,
            x, 
            ndst, 
            dstpos, 
            out,
            batch_size
        );
    }

    return greens_launch_sm70<Tx, Ty>(
        handle, 
        samples,
        channels, 
        srcpos, 
        wavenums,
        x, 
        ndst, 
        dstpos, 
        out,
        batch_size
    );
}

}  // namespace ffdas::detail


ffdas_error_t ffdas_greens(
    ffdas_handle_t handle,
    const float *srcpos,
    const float *wavenums,
    ffdas_tensor_desc_t x_desc,
    const void* x,
    const float *dstpos,
    ffdas_tensor_desc_t out_desc,
    void* out
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(out_desc);
    FFDAS_CHECK(handle->check_device());

    const float3 *srcpos_ = reinterpret_cast<const float3*>(srcpos);
    const float3 *dstpos_ = reinterpret_cast<const float3*>(dstpos);

    ffdas::detail::nvtx_range nvtx(*handle, "greens");

    const ffdas_tensor_desc &x_tensor = *x_desc;
    const ffdas_tensor_desc &out_tensor = *out_desc;

    switch (x_tensor.dtype) {
    case FFDAS_C_16F:
        switch (out_tensor.dtype) {
        case FFDAS_C_32F:
            return ffdas::detail::greens_dispatch<FFDAS_C_16F, FFDAS_C_32F>(
                *handle, srcpos_, wavenums, x_tensor, x, dstpos_, out_tensor, out);
        default: break;
        }
        break;
    case FFDAS_C_32F:
        switch (out_tensor.dtype) {
        case FFDAS_C_32F:
            return ffdas::detail::greens_dispatch<FFDAS_C_32F, FFDAS_C_32F>(
                *handle, srcpos_, wavenums, x_tensor, x, dstpos_, out_tensor, out);
        default: break;
        }
        break;
    default:
        break;
    }

    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}
