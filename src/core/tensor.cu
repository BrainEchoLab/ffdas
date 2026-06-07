#include <stdlib.h>
#include <string.h>
#include <vector>

#include "ffdas.h"
#include "tensor.cuh"


ffdas_error_t ffdas_create_tensor_desc(
    ffdas_tensor_desc_t *desc,
    int64_t ndim,
    const int64_t *dims,
    const int64_t *strides,
    ffdas_datatype_t dtype)
{
    if (!desc || ndim <= 0 || !dims)
        return FFDAS_ERROR_INVALID_ARGUMENT;
    if (ffdas_type_size(dtype) == 0)
        return FFDAS_ERROR_UNSUPPORTED_TYPE;

    try {
        std::vector<int64_t> d(dims, dims + ndim);
        if (strides) {
            *desc = new ffdas_tensor_desc(std::move(d),
                        std::vector<int64_t>(strides, strides + ndim), dtype);
        } else {
            *desc = new ffdas_tensor_desc(std::move(d), dtype);
        }
        return FFDAS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return FFDAS_ERROR_ALLOCATION_FAILED;
    }
}

ffdas_error_t ffdas_destroy_tensor_desc(
    ffdas_tensor_desc_t desc
) {
    if (desc)
        delete desc;

    return FFDAS_SUCCESS;
}
