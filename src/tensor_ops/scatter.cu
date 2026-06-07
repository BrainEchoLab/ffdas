#include "scatter_impl.cuh"
#include "context.cuh"
#include "error_checking.h"


ffdas_error_t ffdas_scatter(
    ffdas_handle_t handle,
    ffdas_tensor_desc_t x_desc,
    const void *x,
    ffdas_tensor_desc_t out_desc,
    void *out,
    int64_t mode,
    const int *indices
) {
    CHECK_HANDLE(handle);
    FFDAS_CHECK(handle->check_device());
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(out_desc);

    ffdas_tensor_desc x_tensor = *x_desc;
    ffdas_tensor_desc out_tensor = *out_desc;

    ffdas::detail::nvtx_range nvtx(*handle, "scatter");

    switch (x_tensor.dtype) {
    case FFDAS_R_16I:
        switch (out_tensor.dtype) {
        case FFDAS_R_16I:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_R_16I, FFDAS_R_16I>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        case FFDAS_R_32F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_R_16I, FFDAS_R_32F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        default:
            break;
        }
        break;
    case FFDAS_R_16F:
        switch (out_tensor.dtype) {
        case FFDAS_R_16F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_R_16F, FFDAS_R_16F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        case FFDAS_R_32F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_R_16F, FFDAS_R_32F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        default:
            break;
        }
        break;
    case FFDAS_R_32F:
        switch (out_tensor.dtype) {
        case FFDAS_R_16F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_R_32F, FFDAS_R_16F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        case FFDAS_R_32F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_R_32F, FFDAS_R_32F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        default:
            break;
        }
        break;
    case FFDAS_C_16I:
        switch (out_tensor.dtype) {
        case FFDAS_C_16I:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_C_16I, FFDAS_C_16I>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        case FFDAS_C_32F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_C_16I, FFDAS_C_32F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        default:
            break;
        }
        break;
    case FFDAS_C_16F:
        switch (out_tensor.dtype) {
        case FFDAS_C_16F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_C_16F, FFDAS_C_16F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        case FFDAS_C_32F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_C_16F, FFDAS_C_32F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        default:
            break;
        }
        break;
    case FFDAS_C_32F:
        switch (out_tensor.dtype) {
        case FFDAS_C_16F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_C_32F, FFDAS_C_16F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        case FFDAS_C_32F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_C_32F, FFDAS_C_32F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        default:
            break;
        }
        break;
    case FFDAS_R_64F:
        switch (out_tensor.dtype) {
        case FFDAS_R_64F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_R_64F, FFDAS_R_64F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        default:
            break;
        }
        break;
    case FFDAS_C_64F:
        switch (out_tensor.dtype) {
        case FFDAS_C_64F:
            return ffdas::detail::ffdas_scatter_dispatch<FFDAS_C_64F, FFDAS_C_64F>(*handle, x_tensor, x, out_tensor, out, mode, indices);
        default:
            break;
        }
        break;
    default:
        break;
    }
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}
