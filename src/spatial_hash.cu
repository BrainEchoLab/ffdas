#include "spatial_hash.cuh"
#include "spatial_hash_kernels.cuh"

#include "context.cuh"
#include "error_checking.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/transform_reduce.h>
#include <thrust/gather.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>


namespace ffdas::detail {

struct bbox_reduce_functor {
    __host__ __device__ bbox operator()(bbox a, bbox b) const {
        return bbox{
            make_float3(fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z)),
            make_float3(fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z))
        };
    }
};

ffdas_error_t build_spatial_hash(
    ffdas_context &handle,
    int64_t nx, 
    int64_t ny, 
    int64_t nz,
    const thrust::device_vector<float3>& points,
    spatial_hash &hash
) {
    int num_points = nz*ny*nx;
    int num_cells = (nz-1)*(ny-1)*(nx-1);
    int num_simplices = num_cells * 6;

    bbox init{make_float3(FLT_MAX, FLT_MAX, FLT_MAX), make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)};
    bbox bounds = thrust::transform_reduce(
        points.begin(), points.end(),
        bbox{},
        init,
        bbox_reduce_functor{}
    );

    int nbins = 128;
    float3 scale = make_float3(
        (float)(nbins-1) / (bounds.max.x - bounds.min.x),
        (float)(nbins-1) / (bounds.max.y - bounds.min.y),
        (float)(nbins-1) / (bounds.max.z - bounds.min.z)
    );

    hash.nbins = nbins;
    hash.bounds = bounds;
    hash.vertices = points;
    hash.simplex_vertex_ids.resize(num_simplices);  // 6 tetrahedra per cell
    hash.simplex_offsets.resize(num_simplices);  // 6 tetrahedra per cell
    hash.simplex_raster_info.resize(num_simplices);  // 6 tetrahedra per cell

    dim3 block_dim(256);
    dim3 grid_dim((num_cells + block_dim.x - 1) / block_dim.x);
    make_simplices_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(
        nx, ny, nz,
        thrust::raw_pointer_cast(hash.simplex_vertex_ids.data())
    );

    CUDA_LAUNCH_CHECK();
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_vector<int3> quantized_points(num_points);

    block_dim = dim3(256);
    grid_dim = dim3((num_points + block_dim.x - 1) / block_dim.x);
    quantize_points_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(
        num_points,
        thrust::raw_pointer_cast(points.data()),
        thrust::raw_pointer_cast(quantized_points.data()),
        bounds.min.x, bounds.min.y, bounds.min.z,
        scale.x, scale.y, scale.z
    );
    
    CUDA_LAUNCH_CHECK();
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    block_dim = dim3(256);
    grid_dim = dim3((num_simplices + block_dim.x - 1) / block_dim.x);
    rasterize_simplices_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(
        thrust::raw_pointer_cast(quantized_points.data()),
        num_simplices,
        thrust::raw_pointer_cast(hash.simplex_vertex_ids.data()),
        thrust::raw_pointer_cast(hash.simplex_raster_info.data()),
        nbins, nbins, nbins
    );

    CUDA_LAUNCH_CHECK();
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // cumulative bin count -> simplex offsets
    thrust::transform_exclusive_scan(
        hash.simplex_raster_info.begin(),
        hash.simplex_raster_info.end(),
        hash.simplex_offsets.begin(),
        [] __device__ (int4 v) {  // compute the total bin count for each simplex
            return v.x * v.y * v.z;
        },
        0,
        thrust::plus<int>()
    );

    // find total number of used bins (last offset plus num used by last simplex)
    int num_hashes;
    CUDA_CHECK(cudaMemcpyAsync(
        &num_hashes,
        thrust::raw_pointer_cast(hash.simplex_offsets.data()) + (num_simplices - 1), 
        sizeof(int), cudaMemcpyDeviceToHost, handle.stream)
    );

    int4 last_raster_info;
    CUDA_CHECK(cudaMemcpyAsync(
        &last_raster_info,
        thrust::raw_pointer_cast(hash.simplex_raster_info.data()) + (num_simplices - 1), 
        sizeof(int4), cudaMemcpyDeviceToHost, handle.stream)
    );
    num_hashes += (last_raster_info.x * last_raster_info.y * last_raster_info.z);

    hash.simplex_hashes.resize(num_hashes);
    hash.hash_ids.resize(num_hashes);

    block_dim = dim3(256);
    grid_dim = dim3((num_simplices + block_dim.x - 1) / block_dim.x);
    hash_simplices_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(
        num_simplices,
        thrust::raw_pointer_cast(hash.simplex_raster_info.data()),
        thrust::raw_pointer_cast(hash.simplex_offsets.data()),
        thrust::raw_pointer_cast(hash.simplex_hashes.data()),
        thrust::raw_pointer_cast(hash.hash_ids.data()),
        nbins, nbins, nbins
    );

    CUDA_LAUNCH_CHECK();

    thrust::stable_sort_by_key(
        hash.simplex_hashes.begin(), 
        hash.simplex_hashes.end(), 
        hash.hash_ids.begin()
    );

    hash.bin_offsets.resize(nbins*nbins*nbins+1);
    thrust::lower_bound(
        hash.simplex_hashes.begin(), 
        hash.simplex_hashes.end(),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(nbins*nbins*nbins+1),
        hash.bin_offsets.begin()
    );

    return FFDAS_SUCCESS;
}

ffdas_error_t find_nearest_vertex(
    ffdas_context &handle,
    const spatial_hash& hash,
    const thrust::device_vector<float3>& query,
    thrust::device_vector<int>& vertex_ids
) {
    float3 scale = make_float3(
        (float)(hash.nbins-1) / (hash.bounds.max.x - hash.bounds.min.x),
        (float)(hash.nbins-1) / (hash.bounds.max.y - hash.bounds.min.y),
        (float)(hash.nbins-1) / (hash.bounds.max.z - hash.bounds.min.z)
    );

    int num_points = query.size();
    thrust::device_vector<uint32_t> query_hashed(num_points);

    dim3 block_dim(256);
    dim3 grid_dim((num_points + block_dim.x - 1) / block_dim.x);
    morton_transform_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(
        num_points,
        thrust::raw_pointer_cast(query.data()),
        thrust::raw_pointer_cast(query_hashed.data()),
        hash.nbins, hash.nbins, hash.nbins,
        hash.bounds.min.x, hash.bounds.min.y, hash.bounds.min.z,
        scale.x, scale.y, scale.z
    );

    CUDA_LAUNCH_CHECK();

    vertex_ids.resize(num_points);

    query_hash_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(
        num_points,
        thrust::raw_pointer_cast(query.data()),
        thrust::raw_pointer_cast(query_hashed.data()),
        thrust::raw_pointer_cast(hash.bin_offsets.data()),
        thrust::raw_pointer_cast(hash.hash_ids.data()),
        thrust::raw_pointer_cast(hash.simplex_vertex_ids.data()),
        thrust::raw_pointer_cast(hash.vertices.data()),
        thrust::raw_pointer_cast(vertex_ids.data()),
        nullptr,
        nullptr
    );

    CUDA_LAUNCH_CHECK();

    return FFDAS_SUCCESS;
}


ffdas_error_t find_simplex(
    ffdas_context &handle,
    const spatial_hash& hash,
    const thrust::device_vector<float3>& query,
    thrust::device_vector<int4>& vertex_ids,
    thrust::device_vector<float4>& barycentric_coords
) {
    float3 scale = make_float3(
        (float)(hash.nbins-1) / (hash.bounds.max.x - hash.bounds.min.x),
        (float)(hash.nbins-1) / (hash.bounds.max.y - hash.bounds.min.y),
        (float)(hash.nbins-1) / (hash.bounds.max.z - hash.bounds.min.z)
    );

    int num_points = query.size();
    thrust::device_vector<uint32_t> query_hashed(num_points);

    dim3 block_dim(256);
    dim3 grid_dim((num_points + block_dim.x - 1) / block_dim.x);
    morton_transform_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(
        num_points,
        thrust::raw_pointer_cast(query.data()),
        thrust::raw_pointer_cast(query_hashed.data()),
        hash.nbins, hash.nbins, hash.nbins,
        hash.bounds.min.x, hash.bounds.min.y, hash.bounds.min.z,
        scale.x, scale.y, scale.z
    );

    CUDA_LAUNCH_CHECK();

    vertex_ids.resize(num_points);
    barycentric_coords.resize(num_points);

    query_hash_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(
        num_points,
        thrust::raw_pointer_cast(query.data()),
        thrust::raw_pointer_cast(query_hashed.data()),
        thrust::raw_pointer_cast(hash.bin_offsets.data()),
        thrust::raw_pointer_cast(hash.hash_ids.data()),
        thrust::raw_pointer_cast(hash.simplex_vertex_ids.data()),
        thrust::raw_pointer_cast(hash.vertices.data()),
        nullptr,
        thrust::raw_pointer_cast(vertex_ids.data()),
        thrust::raw_pointer_cast(barycentric_coords.data())
    );

    CUDA_LAUNCH_CHECK();

    return FFDAS_SUCCESS;
}

}  // namespace ffdas::detail
