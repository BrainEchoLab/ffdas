#pragma once

#include "ffdas.h"

#ifndef FFDAS_API
#define FFDAS_API
#endif

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuComplex.h>


typedef enum {
    FFDAS_SUCCESS = 0,
    
    // Input validation errors
    FFDAS_ERROR_INVALID_ARGUMENT = 1,
    FFDAS_ERROR_INVALID_DIMS = 2,
    FFDAS_ERROR_UNSUPPORTED_TYPE = 3,
    FFDAS_ERROR_NULL_POINTER = 4,
    FFDAS_ERROR_INVALID_HANDLE = 5,
    
    // Memory management errors  
    FFDAS_ERROR_ALLOCATION_FAILED = 10,
    FFDAS_ERROR_DEVICE_MEMORY_INSUFFICIENT = 11,
    FFDAS_ERROR_HOST_MEMORY_INSUFFICIENT = 12,
    FFDAS_ERROR_FAILED_MALLOC = 13, // Legacy compatibility
    
    // Device/hardware errors
    FFDAS_ERROR_DEVICE_NOT_SUPPORTED = 20,
    FFDAS_ERROR_INSUFFICIENT_COMPUTE_CAPABILITY = 21,
    FFDAS_ERROR_DEVICE_NOT_AVAILABLE = 22,
    FFDAS_ERROR_DEVICE_MISMATCH = 23,
    
    // CUDA runtime errors
    FFDAS_ERROR_CUDA_DRIVER = 30,
    FFDAS_ERROR_CUDA_RUNTIME = 31,
    FFDAS_ERROR_CUDA_LAUNCH_FAILED = 32,
    FFDAS_ERROR_CUDA_SYNC_FAILED = 33,
    FFDAS_ERROR_CUDA_MEMCPY_FAILED = 34,
    FFDAS_ERROR_CUDA = 35, // Legacy compatibility alias
    
    // Library errors
    FFDAS_ERROR_CUBLAS = 40,
    FFDAS_ERROR_CUSOLVER = 41,
    FFDAS_ERROR_CUFFT = 42,
    FFDAS_ERROR_NVTX = 43,
    
    // Algorithm/computation errors
    FFDAS_ERROR_ALGORITHM_FAILED = 50,
    FFDAS_ERROR_CONVERGENCE_FAILED = 51,
    FFDAS_ERROR_NUMERICAL_INSTABILITY = 52,
    
    // Configuration/state errors
    FFDAS_ERROR_NOT_INITIALIZED = 60,
    FFDAS_ERROR_ALREADY_INITIALIZED = 61,
    FFDAS_ERROR_INVALID_STATE = 62,

    FFDAS_ERROR_OUT_OF_BOUNDS_INDEX = 63,
    
    // Generic catch-all
    FFDAS_ERROR_INTERNAL = 100,
    FFDAS_ERROR_UNKNOWN = 101
} ffdas_error_t;

typedef enum {
    FFDAS_ALG_DEFAULT = 0,
    FFDAS_ALG1 = 1,
    FFDAS_ALG2 = 2,
    FFDAS_ALG3 = 3,
    FFDAS_ALG4 = 4
} ffdas_alg_t;

typedef enum {
    FFDAS_INTERP_NEAREST = 0,
    FFDAS_INTERP_LINEAR = 1
} ffdas_interp_mode_t;

#include "ffdas_types.h"

struct ffdas_context;
typedef struct ffdas_context* ffdas_handle_t;

struct ffdas_tensor_desc;
typedef struct ffdas_tensor_desc* ffdas_tensor_desc_t;

struct ffdas_contraction_plan;
typedef struct ffdas_contraction_plan* ffdas_contraction_plan_t;

struct ffdas_interpolation_plan;
typedef struct ffdas_interpolation_plan* ffdas_interpolation_plan_t;


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

FFDAS_API size_t ffdas_type_size(ffdas_datatype_t datatype);

FFDAS_API bool iscomplextype(ffdas_datatype_t datatype);

FFDAS_API const char* ffdas_error_string(ffdas_error_t error);

FFDAS_API ffdas_error_t ffdas_create_tensor_desc(
    ffdas_tensor_desc_t *desc,
    int64_t ndim,
    const int64_t *dims,
    const int64_t *strides,
    ffdas_datatype_t dtype
);

FFDAS_API ffdas_error_t ffdas_destroy_tensor_desc(
    ffdas_tensor_desc_t desc
);

FFDAS_API ffdas_error_t ffdas_create(
    ffdas_handle_t *handle
);

FFDAS_API ffdas_error_t ffdas_destroy(
    ffdas_handle_t handle
);

FFDAS_API ffdas_error_t ffdas_set_stream(
    ffdas_handle_t handle,
    uintptr_t stream
);

FFDAS_API ffdas_error_t ffdas_get_stream(
    ffdas_handle_t handle,
    uintptr_t *stream
);

// FFDAS_API ffdas_error_t ffdas_get_bounds_checking(
//     ffdas_handle_t handle,
//     int *value
// );

// FFDAS_API ffdas_error_t ffdas_set_bounds_checking(
//     ffdas_handle_t handle,
//     int value
// );

FFDAS_API ffdas_error_t ffdas_contiguous_copy(
    ffdas_handle_t handle,
    ffdas_tensor_desc_t x_desc,
    const void *x,
    ffdas_tensor_desc_t out_desc,
    void *out
);

FFDAS_API ffdas_error_t ffdas_gather(
    ffdas_handle_t handle,
    ffdas_tensor_desc_t x_desc,
    const void *x,
    ffdas_tensor_desc_t out_desc,
    void *out,
    int64_t mode,
    size_t num_indices,
    const int *indices
);

FFDAS_API ffdas_error_t ffdas_scatter(
    ffdas_handle_t handle,
    ffdas_tensor_desc_t x_desc,
    const void *x,
    ffdas_tensor_desc_t out_desc,
    void *out,
    int64_t mode,
    const int *indices
);

FFDAS_API ffdas_error_t ffdas_truncate_rank(
    ffdas_handle_t handle, 
    ffdas_tensor_desc_t x_desc,
    const void *x,
    int64_t start, 
    int64_t stop,
    ffdas_tensor_desc_t out_desc,
    void *out
);

FFDAS_API ffdas_error_t ffdas_das(
    ffdas_handle_t handle,
    const float3 *srcpos, 
    const float4 *srcdir, 
    float wavenum,
    ffdas_tensor_desc_t x_desc, 
    const void* x,
    const float3 *dstpos, 
    const float *offsets, 
    const float *weights, 
    const void *beta, 
    ffdas_tensor_desc_t out_desc, 
    void* out,
    ffdas_compute_type_t compute_type,
    ffdas_alg_t alg
);

FFDAS_API ffdas_error_t ffdas_das_sparse(
    ffdas_handle_t handle,
    const float3 *srcpos, 
    const float4 *srcdir, 
    float wavenum,
    ffdas_tensor_desc_t x_desc, 
    const void* x,
    const float3 *dstpos, 
    const float *offsets, 
    const float *weights, 
    int sparse_count,
    const int *sparse_indices, 
    const void *beta, 
    ffdas_tensor_desc_t out_desc, 
    void* out,
    ffdas_compute_type_t compute_type,
    ffdas_alg_t alg
);

FFDAS_API ffdas_error_t ffdas_create_contraction(
    ffdas_handle_t handle,
    ffdas_contraction_plan_t *plan,
    ffdas_tensor_desc_t a_desc,
    const int *a_modes,
    ffdas_tensor_desc_t b_desc,
    const int *b_modes,
    ffdas_tensor_desc_t out_desc,
    const int *out_modes
);

FFDAS_API ffdas_error_t ffdas_contraction(
    ffdas_handle_t handle,
    ffdas_contraction_plan_t plan,
    const void *a,
    const void *b,
    void *out
);

FFDAS_API ffdas_error_t ffdas_destroy_contraction(
    ffdas_handle_t handle,
    ffdas_contraction_plan_t plan
);

FFDAS_API ffdas_error_t ffdas_create_interpolation_plan(
    ffdas_handle_t handle,
    ffdas_interpolation_plan_t *plan,
    int64_t nx, int64_t ny, int64_t nz,
    const float *gridpos,
    ffdas_interp_mode_t mode
);

FFDAS_API ffdas_error_t ffdas_interpolation_preprocess(
    ffdas_handle_t handle,
    ffdas_interpolation_plan_t plan,
    int64_t num_querypos,
    const float *querypos
);

FFDAS_API ffdas_error_t ffdas_interpolation(
    ffdas_handle_t handle,
    ffdas_interpolation_plan_t plan,
    int64_t num_querypos,
    const float *querypos,
    ffdas_tensor_desc_t x_desc,
    const void *x,
    void *out,
    const void *fill_value
);

FFDAS_API ffdas_error_t ffdas_destroy_interpolation_plan(
    ffdas_handle_t handle,
    ffdas_interpolation_plan_t plan
);

FFDAS_API ffdas_error_t ffdas_greens_sum(
    ffdas_handle_t handle,
    const float3 *srcpos, 
    const float *wavenums,
    ffdas_tensor_desc_t x_desc, 
    const void* x,
    const float3 *dstpos, 
    ffdas_tensor_desc_t out_desc, 
    void* out
);

FFDAS_API ffdas_error_t ffdas_device_get(
    int *device
);

FFDAS_API ffdas_error_t ffdas_device_set(
    int device
);

FFDAS_API ffdas_error_t ffdas_event_create(
    uintptr_t *event
);

FFDAS_API ffdas_error_t ffdas_event_destroy(
    uintptr_t event
);

FFDAS_API ffdas_error_t ffdas_event_record(
    ffdas_handle_t handle,
    uintptr_t event
);

FFDAS_API ffdas_error_t ffdas_event_synchronize(
    uintptr_t event
);

FFDAS_API ffdas_error_t ffdas_event_elapsed_time(
    uintptr_t start,
    uintptr_t stop,
    float *ms
);

#if defined(__cplusplus)
}
#endif /* __cplusplus */
