#include "das/das_alg4.h"
#include "das/das_alg4_kernels.cuh"
#include "das/das_common.cuh"
#include "das/das_utils.h"

#include "type_utils.h"
#include "math/core.cuh"
#include "tensor.cuh"
#include "error_checking.h"
#include "env_utils.h"

#include <vector>
#include <limits>
#include <cstdio>

namespace ffdas::detail {

template<typename Tx, typename Tcompute, int vec_size>
ffdas_error_t das_alg4_launch(
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    float2 beta, 
    float2* out
) {
    constexpr int block = std::is_same<Tcompute, float2>::value ? 256 : 512;
    constexpr int warps_per_block = block / 32;
    constexpr int N = 8 * vec_size;
    constexpr int M = 8;

    if (params.batch_size % vec_size)
        return FFDAS_ERROR_INVALID_ARGUMENT;

    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        alg4_channel_pos, 
        srcpos, 
        params.channels * sizeof(float3),
        0,
        cudaMemcpyDeviceToDevice,
        handle.stream
    ));


    std::vector<int64_t> dims = {
        params.channels,
        params.seqlen,
        params.samples,
        params.batch_size
    };
    std::vector<int64_t> strides = {
        params.channel_stride,
        params.seq_stride,
        params.sample_stride,
        params.batch_stride
    };

    device_ptr<Tcompute> x_ld_batch_ptr(handle);
    ffdas_tensor_desc x_ld_batch_desc(dims, strides, builtin_traits<Tcompute>::ffdas_datatype);

    if (!can_use_int32_indexing(x_ld_batch_desc))
        return FFDAS_ERROR_INVALID_DIMS;

    FFDAS_CHECK(x_ld_batch_ptr.alloc(x_ld_batch_desc.nbytes()));

    ffdas_error_t err = contiguous_copy_impl<Tx, Tcompute>(handle, x_ld_batch_desc, x, x_ld_batch_ptr.get());
    if (err != FFDAS_SUCCESS)
        return err;

    bool dir_check = (srcdir != NULL);

    dim3 block_dim(block);
    dim3 grid_dim(
        (params.ndst + (warps_per_block * M) - 1) / (warps_per_block * M),
        (params.batch_size + N - 1) / N
    );

    if (dir_check) {
        CUDA_CHECK(cudaFuncSetCacheConfig(das_alg4_kernel<Tcompute, float2, warps_per_block, vec_size, M, N, true>, cudaFuncCachePreferL1));
        das_alg4_kernel<Tcompute, float2, warps_per_block, vec_size, M, N, true><<<grid_dim, block_dim, 0, handle.stream>>>(
            params.samples, 
            params.seqlen, 
            params.channels,
            srcdir,
            wavenum, 
            x_ld_batch_ptr.get(), 
            params.ndst, 
            params.outstride,
            dstpos, 
            offsets, 
            weights, 
            beta, 
            out, 
            params.batch_size
        );
    } else {
        CUDA_CHECK(cudaFuncSetCacheConfig(das_alg4_kernel<Tcompute, float2, warps_per_block, vec_size, M, N, false>, cudaFuncCachePreferL1));
        das_alg4_kernel<Tcompute, float2, warps_per_block, vec_size, M, N, false><<<grid_dim, block_dim, 0, handle.stream>>>(
            params.samples, 
            params.seqlen, 
            params.channels,
            NULL,
            wavenum, 
            x_ld_batch_ptr.get(), 
            params.ndst, 
            params.outstride,
            dstpos, 
            offsets, 
            weights, 
            beta, 
            out, 
            params.batch_size
        );
    }

    CUDA_LAUNCH_CHECK();
    return FFDAS_SUCCESS;
}


template<typename Tx, typename Tcompute>
ffdas_error_t das_alg4_dispatch_vec_size(
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    float2 beta, 
    float2* out
) {
    constexpr int dtype_max = 16 / sizeof(Tcompute);
    constexpr int load_max = 4;
    int batch_max = 1U << ctz(params.batch_size);

    int vec_size = min(load_max, min(dtype_max, batch_max));

    switch (vec_size)
    {
    case 1:
        return das_alg4_launch<Tx, Tcompute, 1>(
            handle,
            params,
            srcpos,
            srcdir,
            wavenum,
            x,
            dstpos,
            offsets,
            weights,
            beta,
            out
        );
    case 2:
        return das_alg4_launch<Tx, Tcompute, 2>(
            handle,
            params,
            srcpos,
            srcdir,
            wavenum,
            x,
            dstpos,
            offsets,
            weights,
            beta,
            out
        );
    case 4:
        if constexpr (sizeof(Tcompute) <= 4) {
            return das_alg4_launch<Tx, Tcompute, 4>(
                handle,
                params,
                srcpos,
                srcdir,
                wavenum,
                x,
                dstpos,
                offsets,
                weights,
                beta,
                out
            );
        }
        break;
    default:
        break;
    }
    return FFDAS_ERROR_INVALID_ARGUMENT;
}


template<typename Tx>
ffdas_error_t das_alg4_dispatch_compute(
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    float2 beta, 
    float2* out,
    ffdas_compute_type_t compute_type
) {
    bool use_half_precision = (compute_type == FFDAS_COMPUTE_16F);

    if (use_half_precision) {
        if constexpr (std::is_same_v<Tx, float2>) {
            return das_alg4_dispatch_vec_size<Tx, __half2>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, beta, out);
        } else if constexpr (std::is_same_v<Tx, __half2>) {
            return das_alg4_dispatch_vec_size<Tx, Tx>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, beta, out);
        }
        return FFDAS_ERROR_UNSUPPORTED_TYPE;
    }

    return das_alg4_dispatch_vec_size<Tx, Tx>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, beta, out);
}


template<typename Tx, typename Ty>
ffdas_error_t das_alg4_strided(
    ffdas_context &handle,
    const float3 *srcpos, 
    const float4 *srcdir, 
    float wavenum,
    const ffdas_tensor_desc &x_desc, 
    const Tx* x,
    const float3 *dstpos, 
    const float *offsets, 
    const float *weights, 
    Ty beta, 
    const ffdas_tensor_desc &out_desc, 
    Ty* out,
    ffdas_compute_type_t compute_type
) {
    if (handle.arch_code < 800) {
        return FFDAS_ERROR_INSUFFICIENT_COMPUTE_CAPABILITY;
    }
    if constexpr (!std::is_same<Ty, float2>::value) {
        return FFDAS_ERROR_UNSUPPORTED_TYPE;
    }

    das_problem_params params;
    FFDAS_CHECK(get_das_problem_params(x_desc, out_desc, params));

    if (params.channels > DAS_ALG4_MAX_CHANNELS)
        return FFDAS_ERROR_INVALID_DIMS;

    if constexpr (std::is_same<Tx, __half2>::value || std::is_same<Tx, float2>::value) {
        return das_alg4_dispatch_compute<Tx>(
            handle,
            params,
            srcpos,
            srcdir,
            wavenum,
            x,
            dstpos,
            offsets,
            weights,
            beta,
            out,
            compute_type
        );
    }

    return FFDAS_ERROR_INVALID_ARGUMENT;
}


// template instantiations
template ffdas_error_t das_alg4_strided<__half, float>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const __half*, 
    const float3 *, const float *, const float *, float, const ffdas_tensor_desc&, float*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg4_strided<float, float>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const float*, 
    const float3 *, const float *, const float *, float, const ffdas_tensor_desc&, float*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg4_strided<__half2, float2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const __half2*, 
    const float3 *, const float *, const float *, float2, const ffdas_tensor_desc&, float2*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg4_strided<float2, float2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const float2*, 
    const float3 *, const float *, const float *, float2, const ffdas_tensor_desc&, float2*, 
    ffdas_compute_type_t
);

template<>
ffdas_error_t das_alg4_strided<double, double>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const double*, 
    const float3 *, const float *, const float *, double, const ffdas_tensor_desc&, double*, 
    ffdas_compute_type_t
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}

template<>
ffdas_error_t das_alg4_strided<double2, double2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const double2*, 
    const float3 *, const float *, const float *, double2, const ffdas_tensor_desc&, double2*, 
    ffdas_compute_type_t
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}

}  // namespace ffdas::detail
