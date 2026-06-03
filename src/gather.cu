#include "gather_impl.cuh"
#include "context.cuh"
#include "error_checking.h"


ffdas_error_t ffdas_gather(
    ffdas_handle_t handle,
    ffdas_tensor_desc_t x_desc,
    const void *x,
    ffdas_tensor_desc_t y_desc,
    void *y,
    int64_t mode,
    size_t num_indices,
    const int *indices
) {
    CHECK_HANDLE(handle);
    FFDAS_CHECK(handle->check_device());
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(y_desc);

    ffdas_tensor_desc x_tensor = *x_desc;
    ffdas_tensor_desc y_tensor = *y_desc;

    ffdas::detail::nvtx_range nvtx(*handle, "gather");

    switch (x_tensor.dtype) {
    case FFDAS_R_16I:
        switch (y_tensor.dtype) {
        case FFDAS_R_16I:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_R_16I, FFDAS_R_16I>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        case FFDAS_R_32F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_R_16I, FFDAS_R_32F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        default:
            break;
        }
        break;
    case FFDAS_R_16F:
        switch (y_tensor.dtype) {
        case FFDAS_R_16F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_R_16F, FFDAS_R_16F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        case FFDAS_R_32F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_R_16F, FFDAS_R_32F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        default:
            break;
        }
        break;
    case FFDAS_R_32F:
        switch (y_tensor.dtype) {
        case FFDAS_R_16F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_R_32F, FFDAS_R_16F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        case FFDAS_R_32F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_R_32F, FFDAS_R_32F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        default:
            break;
        }
        break;
    case FFDAS_C_16I:
        switch (y_tensor.dtype) {
        case FFDAS_C_16I:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_C_16I, FFDAS_C_16I>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        case FFDAS_C_32F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_C_16I, FFDAS_C_32F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        default:
            break;
        }
        break;
    case FFDAS_C_16F:
        switch (y_tensor.dtype) {
        case FFDAS_C_16F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_C_16F, FFDAS_C_16F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        case FFDAS_C_32F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_C_16F, FFDAS_C_32F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        default:
            break;
        }
        break;
    case FFDAS_C_32F:
        switch (y_tensor.dtype) {
        case FFDAS_C_16F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_C_32F, FFDAS_C_16F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        case FFDAS_C_32F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_C_32F, FFDAS_C_32F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        default:
            break;
        }
        break;
    case FFDAS_R_64F:
        switch (y_tensor.dtype) {
        case FFDAS_R_64F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_R_64F, FFDAS_R_64F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        default:
            break;
        }
        break;
    case FFDAS_C_64F:
        switch (y_tensor.dtype) {
        case FFDAS_C_64F:
            return ffdas::detail::ffdas_gather_dispatch<FFDAS_C_64F, FFDAS_C_64F>(*handle, x_tensor, x, y_tensor, y, mode, num_indices, indices);
        default:
            break;
        }
        break;
    default:
        break;
    }
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}
