#include "das_alg2.h"
#include "das_alg2_kernels.cuh"
// #include "das_common.cuh"
#include "das_utils.h"

#include "type_utils.h"
#include "math/core.cuh"
#include "tensor.cuh"
#include "error_checking.h"
#include "env_utils.h"

#include <vector>
#include <limits>
#include <cstdio>

namespace ffdas::detail {

template<typename Tx, typename Ty, typename Tcompute, int vec_size, int M, int N>
ffdas_error_t das_alg2_launch(
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    Ty beta, 
    Ty* out
) {
    static_assert(((M * N) % 32) == 0, "M*N must be a multiple of 32");

    if (!params.have_batch || params.batch_size % vec_size)
        return FFDAS_ERROR_INVALID_DIMS;

    CUDA_CHECK(cudaMemcpyToSymbolAsync(
        alg2_channel_pos, 
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

    constexpr int block = 1024;
    constexpr int warps_per_block = block / 32;

    bool dir_check = (srcdir != NULL);

    dim3 block_dim(block);
    dim3 grid_dim(
        (params.ndst + (M * warps_per_block) - 1) / (M * warps_per_block),
        (params.batch_size + N - 1) / N
    );

    if (dir_check) {
        CUDA_CHECK(cudaFuncSetCacheConfig(das_alg2_kernel<Tcompute, Ty, warps_per_block, vec_size, M, N, true>, cudaFuncCachePreferL1));
        das_alg2_kernel<Tcompute, Ty, warps_per_block, vec_size, M, N, true><<<grid_dim, block_dim, 0, handle.stream>>>(
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
        CUDA_CHECK(cudaFuncSetCacheConfig(das_alg2_kernel<Tcompute, Ty, warps_per_block, vec_size, M, N, false>, cudaFuncCachePreferL1));
        das_alg2_kernel<Tcompute, Ty, warps_per_block, vec_size, M, N, false><<<grid_dim, block_dim, 0, handle.stream>>>(
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


template<typename Tx, typename Ty, typename Tcompute, int M, int N>
ffdas_error_t das_alg2_dispatch_vec_size(
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    Ty beta, 
    Ty* out
) {
    constexpr int dtype_max = 16 / sizeof(Tcompute);
    constexpr int load_max = 4;
    int batch_max = 1U << ctz(params.batch_size);

    int vec_size = min(load_max, min(dtype_max, batch_max));

    switch (vec_size)
    {
    case 1:
        return das_alg2_launch<Tx, Ty, Tcompute, 1, M, N>(
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
        return das_alg2_launch<Tx, Ty, Tcompute, 2, M, N>(
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
            return das_alg2_launch<Tx, Ty, Tcompute, 4, M, N>(
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


template<typename Tx, typename Ty, int M, int N>
ffdas_error_t das_alg2_dispatch_compute(
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    Ty beta, 
    Ty* out,
    ffdas_compute_type_t compute_type
) {
    bool use_half_precision = (compute_type == FFDAS_COMPUTE_16F);

    if (use_half_precision) {
        if constexpr (std::is_same_v<Tx, float>) {
            return das_alg2_dispatch_vec_size<Tx, Ty, __half, M, N>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, beta, out);
        } else if constexpr (std::is_same_v<Tx, float2>) {
            return das_alg2_dispatch_vec_size<Tx, Ty, __half2, M, N>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, beta, out);
        }
        return FFDAS_ERROR_UNSUPPORTED_TYPE;
    }

    return das_alg2_dispatch_vec_size<Tx, Ty, Tx, M, N>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, beta, out);
}


template<typename Tx, typename Ty, int M>
ffdas_error_t das_alg2_dispatch_n(
    int n,
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    Ty beta, 
    Ty* out,
    ffdas_compute_type_t compute_type
) {
    switch (n)
    {
    case 1:
        if constexpr (M < 32) {
            return FFDAS_ERROR_INVALID_ARGUMENT;
        } else {
            return das_alg2_dispatch_compute<Tx, Ty, M, 1>(
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
    case 2:
        if constexpr (M < 16) {
            return FFDAS_ERROR_INVALID_ARGUMENT;
        } else {
            return das_alg2_dispatch_compute<Tx, Ty, M, 2>(
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
    case 4:
        if constexpr (M < 8) {
            return FFDAS_ERROR_INVALID_ARGUMENT;
        } else {
            return das_alg2_dispatch_compute<Tx, Ty, M, 4>(
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
    case 8:
        if constexpr (M < 4) {
            return FFDAS_ERROR_INVALID_ARGUMENT;
        } else {
            return das_alg2_dispatch_compute<Tx, Ty, M, 8>(
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
    case 16:
        if constexpr (M < 2) {
            return FFDAS_ERROR_INVALID_ARGUMENT;
        } else {
            return das_alg2_dispatch_compute<Tx, Ty, M, 16>(
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
    case 32:
        return das_alg2_dispatch_compute<Tx, Ty, M, 32>(
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
    case 64:
        return das_alg2_dispatch_compute<Tx, Ty, M, 64>(
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
    default:
        break;
    }

    return FFDAS_ERROR_INVALID_ARGUMENT;
}


template<typename Tx, typename Ty>
ffdas_error_t das_alg2_strided(
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
    das_problem_params params;
    FFDAS_CHECK(get_das_problem_params(x_desc, out_desc, params));

    if (params.channels > DAS_ALG2_MAX_CHANNELS)
        return FFDAS_ERROR_INVALID_DIMS;
    if (params.samples > INT16_MAX)
        return FFDAS_ERROR_INVALID_DIMS;

    int n_per_warp = getenv_int("FFDAS_ALG2_M", 2);
    int batch_per_warp = getenv_int("FFDAS_ALG2_N", 64);

    batch_per_warp = min(((params.batch_size + 15) / 16) * 16, batch_per_warp);
    n_per_warp = max(n_per_warp, 32 / batch_per_warp);

    if (((n_per_warp & (n_per_warp-1)) != 0))
        return FFDAS_ERROR_INVALID_ARGUMENT;
    if (((batch_per_warp & (batch_per_warp-1)) != 0))
        return FFDAS_ERROR_INVALID_ARGUMENT;
    if ((n_per_warp * batch_per_warp) % 32 != 0)
        return FFDAS_ERROR_INVALID_ARGUMENT;

    switch (n_per_warp)
    {
    case 1:
        return das_alg2_dispatch_n<Tx, Ty, 1>(
            batch_per_warp,
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
    case 2:
        return das_alg2_dispatch_n<Tx, Ty, 2>(
            batch_per_warp,
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
    case 4:
        return das_alg2_dispatch_n<Tx, Ty, 4>(
            batch_per_warp,
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
    default:
        break;
    }

    return FFDAS_ERROR_INVALID_ARGUMENT;
}


// template instantiations
template ffdas_error_t das_alg2_strided<__half, float>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const __half*, 
    const float3 *, const float *, const float *, float, const ffdas_tensor_desc&, float*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg2_strided<float, float>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const float*, 
    const float3 *, const float *, const float *, float, const ffdas_tensor_desc&, float*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg2_strided<__half2, float2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const __half2*, 
    const float3 *, const float *, const float *, float2, const ffdas_tensor_desc&, float2*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg2_strided<float2, float2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const float2*, 
    const float3 *, const float *, const float *, float2, const ffdas_tensor_desc&, float2*, 
    ffdas_compute_type_t
);

template<>
ffdas_error_t das_alg2_strided<double, double>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const double*, 
    const float3 *, const float *, const float *, double, const ffdas_tensor_desc&, double*, 
    ffdas_compute_type_t
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}

template<>
ffdas_error_t das_alg2_strided<double2, double2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const double2*, 
    const float3 *, const float *, const float *, double2, const ffdas_tensor_desc&, double2*, 
    ffdas_compute_type_t
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}


}  // namespace ffdas::detail
