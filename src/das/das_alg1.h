#pragma once

#include "ffdas.h"


namespace ffdas::detail {

template<typename Tx, typename Ty>
ffdas_error_t das_alg1_strided(
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
    ffdas_compute_type_t compute_type
);


template<typename Tx, typename Ty>
ffdas_error_t das_alg1_sparse(
    ffdas_context &handle,
    const float3 *srcpos, 
    const float4 *srcdir, 
    float wavenum,
    const ffdas_tensor_desc &x_desc, 
    const Tx* x,
    const float3 *dstpos, 
    const float *offsets, 
    const float *weights, 
    int sparse_count,
    const int *sparse_indices, 
    Ty beta, 
    const ffdas_tensor_desc &out_desc, 
    Ty* out,
    ffdas_compute_type_t compute_type
);

}  // namespace ffdas::detail
