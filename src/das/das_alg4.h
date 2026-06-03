#pragma once

#include "ffdas.h"
#include "ffdas_types.h"


constexpr int DAS_ALG4_MAX_CHANNELS = 4096;


namespace ffdas::detail {

template<typename Tx, typename Ty>
ffdas_error_t das_alg4_strided(
    ffdas_context &handle,
    const float3 *xpos, 
    const float4 *xdir, 
    float wavenum,
    const ffdas_tensor_desc &x_desc, 
    const Tx* x,
    const float3 *ypos, 
    const float *offsets, 
    const float *weights, 
    Ty beta, 
    const ffdas_tensor_desc &y_desc, 
    Ty* y,
    ffdas_compute_type_t compute_type
);

}  // namespace ffdas::detail
