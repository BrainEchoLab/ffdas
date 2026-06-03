#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <float.h>
#include <cstdint>

#include "ffdas.h"
#include "error_checking.h"


struct bbox {
    float3 min{FLT_MAX, FLT_MAX, FLT_MAX};
    float3 max{-FLT_MAX, -FLT_MAX, -FLT_MAX};

    __host__ __device__ bbox operator()(float3 v) const {
        return bbox{v, v};
    }
};


namespace ffdas::detail {

struct spatial_hash {
    int nbins;
    bbox bounds;
    thrust::device_vector<float3> vertices;
    thrust::device_vector<int4> simplex_vertex_ids;
    thrust::device_vector<int4> simplex_raster_info;
    thrust::device_vector<int> simplex_offsets;
    thrust::device_vector<uint32_t> simplex_hashes;
    thrust::device_vector<int> hash_ids;
    thrust::device_vector<int> bin_offsets;
};


ffdas_error_t build_spatial_hash(
    ffdas_context &handle,
    int64_t nx, int64_t ny, int64_t nz,
    const thrust::device_vector<float3>& points,
    spatial_hash &hash
);


ffdas_error_t find_nearest_vertex(
    ffdas_context &handle,
    const spatial_hash& hash,
    const thrust::device_vector<float3>& query,
    thrust::device_vector<int>& vertex_ids
);


ffdas_error_t find_simplex(
    ffdas_context &handle,
    const spatial_hash& hash,
    const thrust::device_vector<float3>& query,
    thrust::device_vector<int4>& vertex_ids,
    thrust::device_vector<float4>& barycentric_coords
);

}  // namespace ffdas::detail
