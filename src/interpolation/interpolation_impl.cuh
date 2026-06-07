#pragma once

#include <cstdint>

#include "ffdas.h"
#include "tensor.cuh"
#include "spatial_hash.cuh"
#include "error_checking.h"
#include "type_utils.h"


struct ffdas_interpolation_plan {
    ffdas::detail::spatial_hash hash;  // spatial hash structure
    int64_t nz, ny, nx;  // grid dimensions
    ffdas_interp_mode_t mode;  // interpolation mode
    
    // Cached query points state for preprocessing optimization
    thrust::device_vector<float3> cached_querypos;
    thrust::device_vector<int> cached_nearest_indices;  // nearest mode
    thrust::device_vector<int4> cached_simplex_indices;  // linear mode
    thrust::device_vector<float4> cached_barycentric_coords;  // linear mode
    int64_t cached_num_querypos = 0;
    bool is_preprocessed = false;
};


namespace ffdas::detail {

template<typename T>
ffdas_error_t interpolation_impl(
    ffdas_context &handle,
    ffdas_interpolation_plan &plan,
    int64_t num_querypos,
    const float *querypos,
    const ffdas_tensor_desc &x_desc,
    const T *x,
    T *out,
    const T &fill_value
);


template<ffdas_datatype_t T_t>
ffdas_error_t interpolation_dispatch(
    ffdas_context &handle,
    ffdas_interpolation_plan &plan,
    int64_t num_querypos,
    const float *querypos,
    const ffdas_tensor_desc &x_desc,
    const void *x,
    void *out,
    const void *fill_value
) {
    using T = typename ffdas_traits<T_t>::type;
    return interpolation_impl<T>(
        handle, plan, num_querypos, querypos,
        x_desc, static_cast<const T*>(x),
        static_cast<T*>(out),
        *static_cast<const T*>(fill_value)
    );
}


}  // namespace ffdas::detail
