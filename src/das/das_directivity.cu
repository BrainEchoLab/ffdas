#include <new>

#include "ffdas.h"
#include "das_directivity.h"
#include "error_checking.h"


ffdas_error_t ffdas_create_directivity_desc(
    ffdas_directivity_desc_t *desc,
    int64_t channels,
    int64_t events,
    const float *axes
) {
    CHECK_NULL_PTR(desc);
    CHECK_NULL_PTR(axes);
    if (channels <= 0 || events < 0)
        return FFDAS_ERROR_INVALID_DIMS;

    try {
        *desc = new ffdas_directivity_desc{channels, events, axes};
        return FFDAS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return FFDAS_ERROR_ALLOCATION_FAILED;
    }
}

ffdas_error_t ffdas_destroy_directivity_desc(
    ffdas_directivity_desc_t desc
) {
    delete desc;
    return FFDAS_SUCCESS;
}
