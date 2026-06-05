#include <cstdio>

#include "das/das_alg1.h"
#include "das/das_alg1_kernels.cuh"
#include "das/das_common.cuh"
#include "das/das_utils.h"

#include "context.cuh"
#include "type_utils.h"
#include "tensor.cuh"
#include "error_checking.h"

namespace ffdas::detail {

template<typename Tx, typename Ty, typename Tcompute, int tile_width, bool is_sparse>
ffdas_error_t das_alg1_launch(
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    const int *sparse_indices, 
    Ty beta, 
    Ty* out
) {
    const void *input_ptr = x;
    device_ptr<Tcompute> xcomp_ptr;

    ffdas::detail::nvtx_range nvtx_main(handle, "das_alg1_launch");

    if constexpr (!std::is_same<Tx, Tcompute>::value) {
        std::vector<int64_t> dims = {
            params.batch_size, 
            params.channels,
            params.seqlen,
            params.samples
        };
        std::vector<int64_t> strides = {
            params.batch_stride, 
            params.channel_stride,
            params.seq_stride,
            params.sample_stride
        };

        ffdas_tensor_desc xcomp_desc(dims, strides, builtin_traits<Tcompute>::ffdas_datatype);
        if (!can_use_int32_indexing(xcomp_desc))
            return FFDAS_ERROR_INVALID_DIMS;

        FFDAS_CHECK(xcomp_ptr.alloc(handle, xcomp_desc.nbytes()));

        ffdas_error_t err = contiguous_copy_impl<Tx, Tcompute>(handle, xcomp_desc, x, xcomp_ptr.get());

        if (err != FFDAS_SUCCESS)
            return err;

        input_ptr = xcomp_ptr.get();
    } else {
        int64_t exp_seq = params.samples;
        int64_t exp_ch = (int64_t)params.seqlen * params.samples;
        int64_t exp_batch = (int64_t)params.channels * exp_ch;
        bool is_contig = (params.seqlen <= 1 || params.seq_stride == exp_seq) &&
                         (params.channels <= 1 || params.channel_stride == exp_ch) &&
                         (!params.have_batch || params.batch_size <= 1 || params.batch_stride == exp_batch);

        if (!is_contig) {
            std::vector<int64_t> dims = {
                params.batch_size,
                params.channels,
                params.seqlen,
                params.samples
            };
            std::vector<int64_t> strides = {
                params.batch_stride,
                params.channel_stride,
                params.seq_stride,
                params.sample_stride
            };

            ffdas_tensor_desc xcomp_desc(dims, strides, builtin_traits<Tcompute>::ffdas_datatype);
            if (!can_use_int32_indexing(xcomp_desc))
                return FFDAS_ERROR_INVALID_DIMS;

            FFDAS_CHECK(xcomp_ptr.alloc(handle, xcomp_desc.nbytes()));

            ffdas_error_t err = contiguous_copy_impl<Tx, Tcompute>(handle, xcomp_desc, x, xcomp_ptr.get());

            if (err != FFDAS_SUCCESS)
                return err;

            input_ptr = xcomp_ptr.get();
        }
    }

    bool dir_check = (srcdir != NULL);

    dim3 block_dim(128);
    dim3 grid_dim((params.ndst + block_dim.x - 1) / block_dim.x,
                  (params.batch_size + tile_width - 1) / tile_width);

    if (dir_check) {
        CUDA_CHECK(cudaFuncSetCacheConfig(das_alg1_kernel<Tcompute, Ty, tile_width, is_sparse, true>, cudaFuncCachePreferL1));
        ffdas::detail::nvtx_range nvtx_kernel(handle, "das_alg1_kernel");
        das_alg1_kernel<Tcompute, Ty, tile_width, is_sparse, true><<<grid_dim, block_dim, 0, handle.stream>>>(
            params.samples, 
            params.seqlen, 
            params.channels,
            srcpos,
            srcdir,
            wavenum, 
            static_cast<const Tcompute*>(input_ptr), 
            params.ndst, 
            params.outstride,
            dstpos, 
            offsets, 
            weights, 
            sparse_indices, 
            beta, 
            out, 
            params.batch_size
        );
    } else {
        CUDA_CHECK(cudaFuncSetCacheConfig(das_alg1_kernel<Tcompute, Ty, tile_width, is_sparse, false>, cudaFuncCachePreferL1));
        ffdas::detail::nvtx_range nvtx_kernel(handle, "das_alg1_kernel");
        das_alg1_kernel<Tcompute, Ty, tile_width, is_sparse, false><<<grid_dim, block_dim, 0, handle.stream>>>(
            params.samples, 
            params.seqlen, 
            params.channels,
            srcpos,
            NULL,
            wavenum, 
            static_cast<const Tcompute*>(input_ptr), 
            params.ndst, 
            params.outstride,
            dstpos, 
            offsets, 
            weights, 
            sparse_indices, 
            beta, 
            out, 
            params.batch_size
        );
    }

    CUDA_LAUNCH_CHECK();

    return FFDAS_SUCCESS;
}


template<typename Tx, typename Ty, int tile_width, bool is_sparse>
ffdas_error_t das_alg1_dispatch_compute(
    ffdas_context &handle,
    const das_problem_params& params,
    const float3 *srcpos,
    const float4 *srcdir,
    float wavenum,
    const Tx* x, 
    const float3 *dstpos, 
    const float* offsets, 
    const float* weights,
    const int *sparse_indices, 
    Ty beta, 
    Ty* out,
    ffdas_compute_type_t compute_type
) {
    if (compute_type == FFDAS_COMPUTE_16F) {
        if constexpr (std::is_same_v<Tx, float>) {
            return das_alg1_launch<Tx, Ty, __half, tile_width, is_sparse>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out);
        } else if constexpr (std::is_same_v<Tx, float2>) {
            return das_alg1_launch<Tx, Ty, __half2, tile_width, is_sparse>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out);
        }
        return FFDAS_ERROR_UNSUPPORTED_TYPE;
    } else if (compute_type == FFDAS_COMPUTE_32F) {
        if constexpr (std::is_same_v<Tx, double>) {
            return das_alg1_launch<Tx, Ty, float, tile_width, is_sparse>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out);
        } else if constexpr (std::is_same_v<Tx, double2>) {
            return das_alg1_launch<Tx, Ty, float2, tile_width, is_sparse>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out);
        }
    }

    return das_alg1_launch<Tx, Ty, Tx, tile_width, is_sparse>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out);
}

template<typename Tx, typename Ty>
ffdas_error_t das_alg1_strided(
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

    int tile_width;
    const char* env = std::getenv("FFDAS_ALG1_TILE_SIZE");
    if (env) {
        if (std::sscanf(env, "%d", &tile_width) != 1)
            return FFDAS_ERROR_INVALID_ARGUMENT;
    } else {
        // compute tile width as the largest power of two that results in
        // less than or equal to 128 bytes of register memory per thread
        // (reasonable heuristic w.r.t. number of registers needed)
        int log2n = (int)floor(log2(params.batch_size));
        tile_width = (int)min((int)pow(2, log2n), (int)(128 / builtin_traits<Ty>::size));
    }

    // tile_width must be power of two
    if (((tile_width & (tile_width-1)) != 0)) {
        return FFDAS_ERROR_INVALID_ARGUMENT;
    }

    switch (tile_width) {
        case 1:
            return das_alg1_dispatch_compute<Tx, Ty, 1, false>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, NULL, beta, out, compute_type);
        case 2:
            return das_alg1_dispatch_compute<Tx, Ty, 2, false>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, NULL, beta, out, compute_type);
        case 4:
            return das_alg1_dispatch_compute<Tx, Ty, 4, false>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, NULL, beta, out, compute_type);
        case 8:
            return das_alg1_dispatch_compute<Tx, Ty, 8, false>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, NULL, beta, out, compute_type);
        case 16:
            return das_alg1_dispatch_compute<Tx, Ty, 16, false>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, NULL, beta, out, compute_type);
        default:
            break;
    }

    return FFDAS_ERROR_INVALID_DIMS;
}


template<typename Tx, typename Ty>
ffdas_error_t das_alg1_sparse(
    ffdas_context &handle,
    const float3 *srcpos, 
    const float4 *srcdir, 
    float wavenum,
    const ffdas_tensor_desc &x_desc, 
    const Tx* x,
    const float3 *dstpos, 
    const float *offsets, 
    const float *weights, 
    int sparse_count,
    const int *sparse_indices, 
    Ty beta, 
    const ffdas_tensor_desc &out_desc, 
    Ty* out,
    ffdas_compute_type_t compute_type
) {
    das_problem_params params;
    FFDAS_CHECK(get_das_problem_params(x_desc, out_desc, params));

    if (sparse_count <= 0)
        return FFDAS_ERROR_INVALID_ARGUMENT;

    // the actual number of observations to be compounded is passed by the user
    params.seqlen = sparse_count;

    int tile_width;
    const char* env = std::getenv("FFDAS_ALG1_TILE_SIZE");
    if (env) {
        if (std::sscanf(env, "%d", &tile_width) != 1)
            return FFDAS_ERROR_INVALID_ARGUMENT;
    } else {
        // compute tile width as the largest power of two less than or equal to 16
        // (reasonable heuristic w.r.t. number of registers needed)
        int log2n = (int)floor(log2(params.batch_size));
        tile_width = (int)min((int)pow(2, log2n), 8);
    }

    // tile_width must be power of two
    if (((tile_width & (tile_width-1)) != 0)) {
        return FFDAS_ERROR_INVALID_ARGUMENT;
    }

    switch (tile_width) {
        case 1:
            return das_alg1_dispatch_compute<Tx, Ty, 1, true>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out, compute_type);
        case 2:
            return das_alg1_dispatch_compute<Tx, Ty, 2, true>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out, compute_type);
        case 4:
            return das_alg1_dispatch_compute<Tx, Ty, 4, true>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out, compute_type);
        case 8:
            return das_alg1_dispatch_compute<Tx, Ty, 8, true>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out, compute_type);
        case 16:
            return das_alg1_dispatch_compute<Tx, Ty, 16, true>(handle, params, srcpos, srcdir, wavenum, x, dstpos, offsets, weights, sparse_indices, beta, out, compute_type);
        default:
            break;
    }

    return FFDAS_ERROR_INVALID_DIMS;
}

// explicit template instantiations
template ffdas_error_t das_alg1_strided<__half, float>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const __half*, 
    const float3 *, const float *, const float *, float, const ffdas_tensor_desc&, float*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_strided<float, float>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const float*, 
    const float3 *, const float *, const float *, float, const ffdas_tensor_desc&, float*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_strided<__half2, float2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const __half2*, 
    const float3 *, const float *, const float *, float2, const ffdas_tensor_desc&, float2*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_strided<float2, float2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const float2*, 
    const float3 *, const float *, const float *, float2, const ffdas_tensor_desc&, float2*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_sparse<__half, float>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const __half*, 
    const float3 *, const float *, const float *, int, const int *, float, const ffdas_tensor_desc&, float*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_sparse<float, float>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const float*, 
    const float3 *, const float *, const float *, int, const int *, float, const ffdas_tensor_desc&, float*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_sparse<__half2, float2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const __half2*, 
    const float3 *, const float *, const float *, int, const int *, float2, const ffdas_tensor_desc&, float2*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_sparse<float2, float2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const float2*, 
    const float3 *, const float *, const float *, int, const int *, float2, const ffdas_tensor_desc&, float2*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_strided<double, double>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const double*, 
    const float3 *, const float *, const float *, double, const ffdas_tensor_desc&, double*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_strided<double2, double2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const double2*, 
    const float3 *, const float *, const float *, double2, const ffdas_tensor_desc&, double2*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_sparse<double, double>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const double*, 
    const float3 *, const float *, const float *, int, const int *, double, const ffdas_tensor_desc&, double*, 
    ffdas_compute_type_t
);

template ffdas_error_t das_alg1_sparse<double2, double2>(
    ffdas_context&, const float3 *, const float4 *, float, const ffdas_tensor_desc&, const double2*, 
    const float3 *, const float *, const float *, int, const int *, double2, const ffdas_tensor_desc&, double2*, 
    ffdas_compute_type_t
);

}  // namespace ffdas::detail
