#pragma once

#include "ffdas.h"
#include "context.cuh"
#include "tensor.cuh"
#include "type_utils.h"


namespace ffdas::detail {

template<typename Tx, typename Ty>
ffdas_error_t greens_impl(
    ffdas_context &handle,
    const float3 *srcpos,
    const float *wavenums,
    const ffdas_tensor_desc &x_desc,
    const Tx* x,
    const float3 *dstpos,
    const ffdas_tensor_desc &out_desc,
    Ty* out
);


template<ffdas_datatype_t Tx_t, ffdas_datatype_t Ty_t>
ffdas_error_t greens_dispatch(
    ffdas_context &handle,
    const float3 *srcpos,
    const float *wavenums,
    const ffdas_tensor_desc &x_desc,
    const void* x,
    const float3 *dstpos,
    const ffdas_tensor_desc &out_desc,
    void* out
) {
    using Tx = typename ffdas_traits<Tx_t>::type;
    using Ty = typename ffdas_traits<Ty_t>::type;

    return greens_impl<Tx, Ty>(
        handle, srcpos, wavenums,
        x_desc, static_cast<const Tx*>(x),
        dstpos, out_desc, static_cast<Ty*>(out)
    );
}

}  // namespace ffdas::detail
