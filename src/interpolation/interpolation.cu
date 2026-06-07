#include "interpolation_impl.cuh"
#include "interpolation_kernels.cuh"
#include "context.cuh"
#include "tensor.cuh"
#include "spatial_hash.cuh"
#include "error_checking.h"

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <new>
#include <cstdio>


namespace ffdas::detail {

template<typename T>
ffdas_error_t interpolate_nearest(
    ffdas_context &handle,
    const T *x,
    T *out,
    int64_t num_indices,
    const int *indices,
    int64_t batch_size,
    int64_t batch_stride,
    const T &fill_value
) {
    dim3 block_dim(256);
    dim3 grid_dim(
        (num_indices + block_dim.x - 1) / block_dim.x, 
        batch_size
    );
    interp_nearest_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(x, out, (int)num_indices, indices, (int)batch_size, (int)batch_stride, fill_value);

    CUDA_LAUNCH_CHECK();

    return FFDAS_SUCCESS;
}


template<typename T>
ffdas_error_t interpolate_linear(
    ffdas_context &handle,
    const T *x,
    T *out,
    int64_t num_indices,
    const int4 *indices,
    const float4 *weights,
    int64_t batch_size,
    int64_t batch_stride,
    const T &fill_value
) {
    dim3 block_dim(256);
    dim3 grid_dim((num_indices + block_dim.x - 1) / block_dim.x, batch_size);
    interp_linear_kernel<<<grid_dim, block_dim, 0, handle.stream>>>(x, out, (int)num_indices, indices, weights, (int)batch_size, (int)batch_stride, fill_value);

    CUDA_LAUNCH_CHECK();

    return FFDAS_SUCCESS;
}


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
) {
    // Check if we can use preprocessed data
    bool use_preprocessed = plan.is_preprocessed && 
                           plan.cached_num_querypos == num_querypos;
    
    // Use pointers to avoid copying device_vectors and causing double-free
    thrust::device_vector<int>* nearest_indices_ptr = nullptr;
    thrust::device_vector<int4>* simplex_indices_ptr = nullptr;
    thrust::device_vector<float4>* barycentric_coords_ptr = nullptr;
    
    // Local vectors for non-preprocessed case
    thrust::device_vector<float3> device_query_points;
    thrust::device_vector<int> local_nearest_indices;
    thrust::device_vector<int4> local_simplex_indices;
    thrust::device_vector<float4> local_barycentric_coords;

    if (use_preprocessed) {
        // Use cached data - just point to the cached vectors
        if (plan.mode == FFDAS_INTERP_NEAREST) {
            nearest_indices_ptr = &plan.cached_nearest_indices;
        } else {
            simplex_indices_ptr = &plan.cached_simplex_indices;
            barycentric_coords_ptr = &plan.cached_barycentric_coords;
        }
    } else {
        // Create device vector directly from input data (avoid copy)
        const float3* query_points_f3 = reinterpret_cast<const ffdas_float3*>(querypos);
        device_query_points.assign(query_points_f3, query_points_f3 + num_querypos);
        
        ffdas_error_t err;
        if (plan.mode == FFDAS_INTERP_NEAREST) {
            err = find_nearest_vertex(handle, plan.hash, device_query_points, local_nearest_indices);
            nearest_indices_ptr = &local_nearest_indices;
        } else {
            err = find_simplex(handle, plan.hash, device_query_points, local_simplex_indices, local_barycentric_coords);
            simplex_indices_ptr = &local_simplex_indices;
            barycentric_coords_ptr = &local_barycentric_coords;
        }
        
        if (err != FFDAS_SUCCESS) {
            return err;
        }
    }

    // Calculate batch parameters
    int64_t batch_size = 1;
    int64_t batch_stride = plan.nz * plan.ny * plan.nx;

    if (x_desc.ndim() == 4) {
        batch_size = x_desc.dims[0];
    }
    
    // Perform interpolation
    if (plan.mode == FFDAS_INTERP_NEAREST) {
        return interpolate_nearest<T>(
            handle,
            x, out, num_querypos,
            thrust::raw_pointer_cast(nearest_indices_ptr->data()),
            batch_size, batch_stride, fill_value
        );
    } else {
        return interpolate_linear<T>(
            handle,
            x, out, num_querypos,
            thrust::raw_pointer_cast(simplex_indices_ptr->data()),
            thrust::raw_pointer_cast(barycentric_coords_ptr->data()),
            batch_size, batch_stride, fill_value
        );
    }
}

}  // namespace ffdas::detail


ffdas_error_t ffdas_create_interpolation_plan(
    ffdas_handle_t handle,
    ffdas_interpolation_plan_t *plan,
    int64_t nx, 
    int64_t ny, 
    int64_t nz,
    const float *gridpos,
    ffdas_interp_mode_t mode
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(plan);
    CHECK_NULL_PTR(gridpos);

    if (nz <= 0 || ny <= 0 || nx <= 0)
        return FFDAS_ERROR_INVALID_DIMS;

    ffdas::detail::nvtx_range nvtx(*handle, "interpolation_plan");

    try {
        // Allocate plan structure
        *plan = new ffdas_interpolation_plan;
    } catch (const std::exception& e) {
        if (*plan) {
            delete *plan;
            *plan = nullptr;
        }
        return FFDAS_ERROR_ALLOCATION_FAILED;
    }

    (*plan)->nz = nz;
    (*plan)->ny = ny;
    (*plan)->nx = nx;
    (*plan)->mode = mode;
    (*plan)->is_preprocessed = false;
    (*plan)->cached_num_querypos = 0;

    // Copy grid points to device
    int64_t num_points = nz * ny * nx;
    thrust::device_vector<float3> device_grid_points(num_points);
    
    // The input gridpos is a flat array of floats with 3*num_points elements
    // representing (x,out,z) coordinates
    CUDA_CHECK(cudaMemcpyAsync(
        thrust::raw_pointer_cast(device_grid_points.data()),
        gridpos,
        num_points * 3 * sizeof(float),
        cudaMemcpyDeviceToDevice,
        handle->stream
    ));

    // Build spatial hash
    ffdas_error_t err = ffdas::detail::build_spatial_hash(*handle, nx, ny, nz, device_grid_points, (*plan)->hash);
    if (err != FFDAS_SUCCESS) {
        delete *plan;
        *plan = nullptr;
        return err;
    }

    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_interpolation(
    ffdas_handle_t handle,
    ffdas_interpolation_plan_t plan,
    int64_t num_querypos,
    const float *querypos,
    ffdas_tensor_desc_t x_desc,
    const void *x,
    void *out,
    const void *fill_value
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(plan);
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(x);
    CHECK_NULL_PTR(out);
    CHECK_NULL_PTR(fill_value);
    
    // querypos can be NULL when using preprocessing
    if (!querypos && !plan->is_preprocessed)
        return FFDAS_ERROR_INVALID_ARGUMENT;
    if (num_querypos <= 0)
        return FFDAS_ERROR_INVALID_DIMS;

    ffdas::detail::nvtx_range nvtx(*handle, "interpolation");

    // Validate x tensor descriptor
    ffdas_tensor_desc x_tensor = *x_desc;
    
    // Values shape: (nz, ny, nx) or (batch, nz, ny, nx)
    if (x_tensor.ndim() == 3) {
        if (x_tensor.dims[0] != plan->nz ||
            x_tensor.dims[1] != plan->ny ||
            x_tensor.dims[2] != plan->nx)
            return FFDAS_ERROR_INVALID_DIMS;
    } else if (x_tensor.ndim() == 4) {
        if (x_tensor.dims[1] != plan->nz ||
            x_tensor.dims[2] != plan->ny ||
            x_tensor.dims[3] != plan->nx)
            return FFDAS_ERROR_INVALID_DIMS;
    } else {
        return FFDAS_ERROR_INVALID_DIMS;
    }

    // Dispatch based on data type
    switch (x_tensor.dtype) {
        case FFDAS_R_32F:
            return ffdas::detail::interpolation_dispatch<FFDAS_R_32F>(
                *handle, *plan, num_querypos, querypos,
                x_tensor, x, out, fill_value
            );
        case FFDAS_C_32F:
            return ffdas::detail::interpolation_dispatch<FFDAS_C_32F>(
                *handle, *plan, num_querypos, querypos,
                x_tensor, x, out, fill_value
            );
        case FFDAS_R_64F:
            return ffdas::detail::interpolation_dispatch<FFDAS_R_64F>(
                *handle, *plan, num_querypos, querypos,
                x_tensor, x, out, fill_value
            );
        case FFDAS_C_64F:
            return ffdas::detail::interpolation_dispatch<FFDAS_C_64F>(
                *handle, *plan, num_querypos, querypos,
                x_tensor, x, out, fill_value
            );
        default:
            return FFDAS_ERROR_UNSUPPORTED_TYPE;
    }
}


ffdas_error_t ffdas_destroy_interpolation_plan(
    ffdas_handle_t handle,
    ffdas_interpolation_plan_t plan
) {
    if (!plan)
        return FFDAS_SUCCESS;

    try {
        // thrust::device_vector destructors will automatically free GPU memory
        // when the plan object is destroyed
        delete plan;
        return FFDAS_SUCCESS;
    } catch (const std::exception& e) {
        return FFDAS_ERROR_INTERNAL;
    }
}

ffdas_error_t ffdas_interpolation_preprocess(
    ffdas_handle_t handle,
    ffdas_interpolation_plan_t plan,
    int64_t num_querypos,
    const float *querypos
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(plan);
    CHECK_NULL_PTR(querypos);

    if (num_querypos <= 0)
        return FFDAS_ERROR_INVALID_DIMS;

    ffdas::detail::nvtx_range nvtx(*handle, "interpolation_preprocess");

    try {
        // Create device vector directly from input data (avoid copy)
        const float3* query_points_f3 = reinterpret_cast<const ffdas_float3*>(querypos);
        plan->cached_querypos.assign(query_points_f3, query_points_f3 + num_querypos);
        plan->cached_num_querypos = num_querypos;
    } catch (const std::exception& e) {
        return FFDAS_ERROR_ALLOCATION_FAILED;
    }

    // Precompute interpolation parameters based on mode
    if (plan->mode == FFDAS_INTERP_NEAREST) {
        // Use find_nearest_vertex
        FFDAS_CHECK(ffdas::detail::find_nearest_vertex(
            *handle,
            plan->hash,
            plan->cached_querypos,
            plan->cached_nearest_indices
        ));
    } else {
        // Use find_simplex for linear interpolation
        FFDAS_CHECK(ffdas::detail::find_simplex(
            *handle,
            plan->hash,
            plan->cached_querypos,
            plan->cached_simplex_indices,
            plan->cached_barycentric_coords
        ));
    }

    plan->is_preprocessed = true;
    return FFDAS_SUCCESS;
}
