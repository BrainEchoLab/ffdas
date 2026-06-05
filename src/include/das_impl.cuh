#pragma once

#include <cuda_runtime.h>

#include "ffdas_api.h"
#include "ffdas_types.h"
#include "type_utils.h"
#include "tensor.cuh"
#include "error_checking.h"

#include "das/das_alg1.h"
#include "das/das_alg2.h"
#include "das/das_alg4.h"


namespace ffdas::detail {

template<typename Tx, typename Ty>
ffdas_error_t das_impl(
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
    ffdas_compute_type_t compute_type,
    ffdas_alg_t alg
) {
    if (!can_use_int32_indexing(x_desc) || !can_use_int32_indexing(out_desc))
        return FFDAS_ERROR_INVALID_DIMS;

    if constexpr (std::is_same_v<Ty, double> || std::is_same_v<Ty, double2>) {
        if (alg != FFDAS_ALG_DEFAULT && alg != FFDAS_ALG1)
            return FFDAS_ERROR_UNSUPPORTED_TYPE;
    }

    switch (alg)
    {
    case FFDAS_ALG_DEFAULT:
    case FFDAS_ALG1:
        return das_alg1_strided<Tx, Ty>(handle, srcpos, srcdir, wavenum, x_desc, x, dstpos, offsets, weights, beta, out_desc, out, compute_type);
    case FFDAS_ALG2:
        return das_alg2_strided<Tx, Ty>(handle, srcpos, srcdir, wavenum, x_desc, x, dstpos, offsets, weights, beta, out_desc, out, compute_type);
    case FFDAS_ALG4:
        return das_alg4_strided<Tx, Ty>(handle, srcpos, srcdir, wavenum, x_desc, x, dstpos, offsets, weights, beta, out_desc, out, compute_type);
    default:
        break;
    }

    return FFDAS_ERROR_INVALID_ARGUMENT;
}


template<ffdas_datatype_t Tx_t, ffdas_datatype_t Ty_t>
ffdas_error_t ffdas_das_dispatch(
    ffdas_context &handle,
    const float3 *srcpos, 
    const float4 *srcdir, 
    float wavenum,
    const ffdas_tensor_desc &x_desc, 
    const void* x,
    const float3 *dstpos, 
    const float *offsets, 
    const float *weights, 
    const void *beta, 
    const ffdas_tensor_desc &out_desc, 
    void* out,
    ffdas_compute_type_t compute_type,
    ffdas_alg_t alg
) {
    using Tx = typename ffdas_traits<Tx_t>::type;
    using Ty = typename ffdas_traits<Ty_t>::type;

    return das_impl<Tx, Ty>(
        handle,
        srcpos,
        srcdir,
        wavenum,
        x_desc,
        static_cast<const Tx*>(x),
        dstpos,
        offsets,
        weights,
        *reinterpret_cast<const Ty*>(beta),
        out_desc,
        static_cast<Ty*>(out),
        compute_type,
        alg
    );
}

}  // namespace ffdas::detail
