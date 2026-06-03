#pragma once

#include "ffdas.h"
#include "context.cuh"
#include "tensor.cuh"
#include "type_utils.h"


namespace ffdas::detail {

template<typename Tx, typename Ty>
ffdas_error_t greens_sum_impl(
    ffdas_context &handle,
    const float3 *xpos,
    const float *wavenums,
    const ffdas_tensor_desc &x_desc,
    const Tx* x,
    const float3 *ypos,
    const ffdas_tensor_desc &y_desc,
    Ty* y
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}


template<ffdas_datatype_t Tx_t, ffdas_datatype_t Ty_t>
ffdas_error_t greens_sum_dispatch(
    ffdas_context &handle,
    const float3 *xpos,
    const float *wavenums,
    const ffdas_tensor_desc &x_desc,
    const void* x,
    const float3 *ypos,
    const ffdas_tensor_desc &y_desc,
    void* y
) {
    using Tx = typename ffdas_traits<Tx_t>::type;
    using Ty = typename ffdas_traits<Ty_t>::type;

    return greens_sum_impl<Tx, Ty>(
        handle, xpos, wavenums,
        x_desc, static_cast<const Tx*>(x),
        ypos, y_desc, static_cast<Ty*>(y)
    );
}

}  // namespace ffdas::detail
