#include "truncate_rank_impl.cuh"
#include "context.cuh"
#include "error_checking.h"


ffdas_error_t ffdas_truncate_rank(
    ffdas_handle_t handle,
    ffdas_tensor_desc_t x_desc,
    const void *x,
    int64_t start, 
    int64_t stop,
    ffdas_tensor_desc_t y_desc,
    void *y
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(y_desc);
    FFDAS_CHECK(handle->check_device());

    const ffdas_tensor_desc &x_tensor = *x_desc;
    const ffdas_tensor_desc &y_tensor = *y_desc;

    if (x_tensor.dtype != y_tensor.dtype)
        return FFDAS_ERROR_UNSUPPORTED_TYPE;

    ffdas::detail::nvtx_range nvtx(*handle, "truncate_rank");

    switch (x_tensor.dtype) {
    case FFDAS_R_32F:
        return ffdas::detail::truncate_rank_dispatch<FFDAS_R_32F>(*handle, x_tensor, x, y_tensor, y, start, stop);
    case FFDAS_C_32F:
        return ffdas::detail::truncate_rank_dispatch<FFDAS_C_32F>(*handle, x_tensor, x, y_tensor, y, start, stop);
    case FFDAS_R_64F:
        return ffdas::detail::truncate_rank_dispatch<FFDAS_R_64F>(*handle, x_tensor, x, y_tensor, y, start, stop);
    case FFDAS_C_64F:
        return ffdas::detail::truncate_rank_dispatch<FFDAS_C_64F>(*handle, x_tensor, x, y_tensor, y, start, stop);
    default:
        break;
    }

    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}
