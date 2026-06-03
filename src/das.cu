#include <cstdio>

#include "das_impl.cuh"
#include "context.cuh"
#include "error_checking.h"


ffdas_error_t ffdas_das(
    ffdas_handle_t handle,
    const float3 *xpos, 
    const float4 *xdir, 
    float wavenum,
    ffdas_tensor_desc_t x_desc, 
    const void* x,
    const float3 *ypos, 
    const float *offsets, 
    const float *weights, 
    const void *beta, 
    ffdas_tensor_desc_t y_desc, 
    void* y,
    ffdas_compute_type_t compute_type,
    ffdas_alg_t alg
) {
    CHECK_HANDLE(handle);
    FFDAS_CHECK(handle->check_device());
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(y_desc);

    ffdas_tensor_desc x_tensor = *x_desc;
    ffdas_tensor_desc y_tensor = *y_desc;

    ffdas::detail::nvtx_range nvtx(*handle, "das");

    // Dispatch based on both input and output tensor types
    switch (x_tensor.dtype)
    {
    case FFDAS_R_16F:
        switch (y_tensor.dtype) {
            case FFDAS_R_32F: return ffdas::detail::ffdas_das_dispatch<FFDAS_R_16F, FFDAS_R_32F>(*handle, xpos, xdir, wavenum, x_tensor, x, ypos, offsets, weights, beta, y_tensor, y, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_R_32F:
        switch (y_tensor.dtype) {
            case FFDAS_R_32F: return ffdas::detail::ffdas_das_dispatch<FFDAS_R_32F, FFDAS_R_32F>(*handle, xpos, xdir, wavenum, x_tensor, x, ypos, offsets, weights, beta, y_tensor, y, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_C_16F:
        switch (y_tensor.dtype) {
            case FFDAS_C_32F: return ffdas::detail::ffdas_das_dispatch<FFDAS_C_16F, FFDAS_C_32F>(*handle, xpos, xdir, wavenum, x_tensor, x, ypos, offsets, weights, beta, y_tensor, y, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_C_32F:
        switch (y_tensor.dtype) {
            case FFDAS_C_32F: return ffdas::detail::ffdas_das_dispatch<FFDAS_C_32F, FFDAS_C_32F>(*handle, xpos, xdir, wavenum, x_tensor, x, ypos, offsets, weights, beta, y_tensor, y, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_R_64F:
        switch (y_tensor.dtype) {
            case FFDAS_R_64F: return ffdas::detail::ffdas_das_dispatch<FFDAS_R_64F, FFDAS_R_64F>(*handle, xpos, xdir, wavenum, x_tensor, x, ypos, offsets, weights, beta, y_tensor, y, compute_type, alg);
            default: break;
        }
        break;
    case FFDAS_C_64F:
        switch (y_tensor.dtype) {
            case FFDAS_C_64F: return ffdas::detail::ffdas_das_dispatch<FFDAS_C_64F, FFDAS_C_64F>(*handle, xpos, xdir, wavenum, x_tensor, x, ypos, offsets, weights, beta, y_tensor, y, compute_type, alg);
            default: break;
        }
        break;
    default:
        break;
    }
    
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}
