#include <cstdio>

#include "tensor.cuh"
#include "context.cuh"
#include "das_impl.cuh"
#include "error_checking.h"
#include "das_directivity.h"


ffdas_error_t ffdas_das_ex(
    ffdas_handle_t handle,
    const float *srcpos, 
    ffdas_directivity_desc_t directivity, 
    float wavenum,
    ffdas_tensor_desc_t x_desc, 
    const void* x,
    const float *dstpos, 
    const float *offsets, 
    const float *weights, 
    const void *beta, 
    ffdas_tensor_desc_t out_desc, 
    void* out,
    ffdas_compute_type_t compute_type,
    ffdas_alg_t alg
) {
    CHECK_HANDLE(handle);
    FFDAS_CHECK(handle->check_device());
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(out_desc);

    ffdas_tensor_desc x_tensor = *x_desc;
    ffdas_tensor_desc out_tensor = *out_desc;

    const float3 *srcpos_ = reinterpret_cast<const float3*>(srcpos);
    const float3 *dstpos_ = reinterpret_cast<const float3*>(dstpos);

    ffdas::detail::nvtx_range nvtx(*handle, "das");

    // Dispatch based on both input and output tensor types
    switch (x_tensor.dtype)
    {
    case FFDAS_R_16F:
        switch (out_tensor.dtype) {
            case FFDAS_R_32F: return ffdas::detail::ffdas_das_dispatch<FFDAS_R_16F, FFDAS_R_32F>(*handle, srcpos_, directivity, wavenum, x_tensor, x, dstpos_, offsets, weights, beta, out_tensor, out, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_R_32F:
        switch (out_tensor.dtype) {
            case FFDAS_R_32F: return ffdas::detail::ffdas_das_dispatch<FFDAS_R_32F, FFDAS_R_32F>(*handle, srcpos_, directivity, wavenum, x_tensor, x, dstpos_, offsets, weights, beta, out_tensor, out, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_C_16F:
        switch (out_tensor.dtype) {
            case FFDAS_C_32F: return ffdas::detail::ffdas_das_dispatch<FFDAS_C_16F, FFDAS_C_32F>(*handle, srcpos_, directivity, wavenum, x_tensor, x, dstpos_, offsets, weights, beta, out_tensor, out, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_C_32F:
        switch (out_tensor.dtype) {
            case FFDAS_C_32F: return ffdas::detail::ffdas_das_dispatch<FFDAS_C_32F, FFDAS_C_32F>(*handle, srcpos_, directivity, wavenum, x_tensor, x, dstpos_, offsets, weights, beta, out_tensor, out, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_R_64F:
        switch (out_tensor.dtype) {
            case FFDAS_R_64F: return ffdas::detail::ffdas_das_dispatch<FFDAS_R_64F, FFDAS_R_64F>(*handle, srcpos_, directivity, wavenum, x_tensor, x, dstpos_, offsets, weights, beta, out_tensor, out, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_C_64F:
        switch (out_tensor.dtype) {
            case FFDAS_C_64F: return ffdas::detail::ffdas_das_dispatch<FFDAS_C_64F, FFDAS_C_64F>(*handle, srcpos_, directivity, wavenum, x_tensor, x, dstpos_, offsets, weights, beta, out_tensor, out, compute_type, alg);
            default: break;
        }
        break;
    default:
        break;
    }
    
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}

// srcdir: one (look direction, cos half-angle) float4 per channel, the same for every event
ffdas_error_t ffdas_das(
    ffdas_handle_t handle,
    const float *srcpos, 
    const float *srcdir, 
    float wavenum,
    ffdas_tensor_desc_t x_desc, 
    const void* x,
    const float *dstpos, 
    const float *offsets, 
    const float *weights, 
    const void *beta, 
    ffdas_tensor_desc_t out_desc, 
    void* out,
    ffdas_compute_type_t compute_type,
    ffdas_alg_t alg
) {
    ffdas_directivity_desc cone{0, 0, srcdir};
    return ffdas_das_ex(handle, srcpos, srcdir ? &cone : nullptr, wavenum, x_desc, x, dstpos, offsets, weights, beta, out_desc, out, compute_type, alg);
}
