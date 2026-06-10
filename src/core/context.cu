#include <iostream>
#include <memory>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cub/cub.cuh>

#include "ffdas.h"
#include "context.cuh"
#include "error_checking.h"


ffdas_error_t ffdas_create(ffdas_handle_t *handle) {
    CHECK_NULL_PTR(handle);
    
    *handle = new ffdas_context();

    if (!*handle) {
        return FFDAS_ERROR_HOST_MEMORY_INSUFFICIENT;
    }

    cudaDeviceProp prop;
    int arch_code;
    ffdas_error_t err;

    err = (cudaGetDevice(&(*handle)->device_id) == cudaSuccess) ? FFDAS_SUCCESS : FFDAS_ERROR_CUDA_RUNTIME;
    if (err != FFDAS_SUCCESS) goto cleanup;
    
    err = (cudaGetDeviceProperties(&prop, (*handle)->device_id) == cudaSuccess) ? FFDAS_SUCCESS : FFDAS_ERROR_CUDA_RUNTIME;
    if (err != FFDAS_SUCCESS) goto cleanup;

    arch_code = prop.major * 100 + prop.minor * 10;
    if (arch_code < 530) {
        err = FFDAS_ERROR_INSUFFICIENT_COMPUTE_CAPABILITY;
        goto cleanup;
    }

    err = (cublasCreate(&(*handle)->cublas_h) == CUBLAS_STATUS_SUCCESS) ? FFDAS_SUCCESS : FFDAS_ERROR_CUBLAS;
    if (err != FFDAS_SUCCESS) goto cleanup;

    err = (cusolverDnCreate(&(*handle)->cusolver_h) == CUSOLVER_STATUS_SUCCESS) ? FFDAS_SUCCESS : FFDAS_ERROR_CUSOLVER;
    if (err != FFDAS_SUCCESS) goto cleanup;

    (*handle)->arch_code = arch_code;

#ifdef FFDAS_USE_NVTX
    (*handle)->nvtx_domain = nvtxDomainCreateA("ffdas");
#endif

    (*handle)->allocator = std::make_unique<cub::CachingDeviceAllocator>( 
        false,
        false
    );
    return FFDAS_SUCCESS;

cleanup:
    if ((*handle)->cublas_h) {
        cublasDestroy((*handle)->cublas_h);
    }
    if ((*handle)->cusolver_h) {
        cusolverDnDestroy((*handle)->cusolver_h);
    }
    delete *handle;
    *handle = nullptr;
    return err;
}


ffdas_error_t ffdas_destroy(ffdas_handle_t handle) {
    CHECK_HANDLE(handle);

    // If the CUDA context is already gone, skip all CUDA cleanup 
    // and let the OS reclaim the resources when the process exits
    if (!handle->is_alive()) {
        delete handle;
        return FFDAS_SUCCESS;
    }

    ffdas_error_t last_error = FFDAS_SUCCESS;

    if (handle->cublas_h) {
        cublasStatus_t status = cublasDestroy(handle->cublas_h);
        if (status != CUBLAS_STATUS_SUCCESS && last_error == FFDAS_SUCCESS) {
            last_error = FFDAS_ERROR_CUBLAS;
        }
    }

    if (handle->cusolver_h) {
        cusolverStatus_t status = cusolverDnDestroy(handle->cusolver_h);
        if (status != CUSOLVER_STATUS_SUCCESS && last_error == FFDAS_SUCCESS) {
            last_error = FFDAS_ERROR_CUSOLVER;
        }
    }

#ifdef FFDAS_USE_NVTX
    nvtxDomainDestroy(handle->nvtx_domain);
#endif
    
    delete handle;
    return last_error;
}


// Set the context's active stream
ffdas_error_t ffdas_set_stream(
    ffdas_handle_t handle,
    uintptr_t stream
) {
    CHECK_HANDLE(handle);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    handle->stream = s;

    if (handle->cublas_h) {
        cublasStatus_t status = cublasSetStream(handle->cublas_h, s);
        if (status != CUBLAS_STATUS_SUCCESS) return FFDAS_ERROR_CUBLAS;
    }

    if (handle->cusolver_h) {
        cusolverStatus_t status = cusolverDnSetStream(handle->cusolver_h, s);
        if (status != CUSOLVER_STATUS_SUCCESS) return FFDAS_ERROR_CUSOLVER;
    }

    return FFDAS_SUCCESS;
}


// Return the context's active stream
ffdas_error_t ffdas_get_stream(
    ffdas_handle_t handle,
    uintptr_t *stream
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(stream);
    *stream = reinterpret_cast<uintptr_t>(handle->stream);
    return FFDAS_SUCCESS;
}

// Allocate memory associated with this context's active stream.
ffdas_error_t ffdas_context::device_alloc(size_t bytes, void **ptr) const {
    cudaError_t cuda_err = allocator->DeviceAllocate(
        device_id, 
        ptr, 
        bytes, 
        stream
    );
    return (cuda_err == cudaSuccess) ? FFDAS_SUCCESS : FFDAS_ERROR_CUDA_RUNTIME;
}

// Free workspace memory allocated by this context, synchronous to the 
// stream in which it was allocated.
ffdas_error_t ffdas_context::device_free(void *ptr) const {
    cudaError_t cuda_err = allocator->DeviceFree(device_id, ptr);
    return (cuda_err == cudaSuccess) ? FFDAS_SUCCESS : FFDAS_ERROR_CUDA_RUNTIME;
}


bool ffdas_context::is_alive() const {
    CUdevice dev = static_cast<CUdevice>(device_id);
    unsigned int flags = 0;
    int active = 0;
    CUresult res = cuDevicePrimaryCtxGetState(dev, &flags, &active);
    return (res == CUDA_SUCCESS && active != 0);
}


ffdas_error_t ffdas_context::check_device() const {
    int current_device;
    cudaError_t err = cudaGetDevice(&current_device);
    if (err != cudaSuccess) return FFDAS_ERROR_CUDA_RUNTIME;
    if (current_device != device_id) {
        return FFDAS_ERROR_DEVICE_MISMATCH;
    }
    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_device_get(int *device) {
    CUDA_CHECK(cudaGetDevice(device));
    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_device_set(int device) {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (device < 0 || device >= count)
        return FFDAS_ERROR_INVALID_DEVICE;
    CUDA_CHECK(cudaSetDevice(device));
    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_device_count(int *count) {
    CHECK_NULL_PTR(count);
    CUDA_CHECK(cudaGetDeviceCount(count));
    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_event_create(uintptr_t *event) {
    CHECK_NULL_PTR(event);
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreate(&e));
    *event = reinterpret_cast<uintptr_t>(e);
    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_event_destroy(uintptr_t event) {
    if (event == 0)
        return FFDAS_SUCCESS;
    CUDA_CHECK(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)));
    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_event_record(ffdas_handle_t handle, uintptr_t event) {
    CHECK_HANDLE(handle);
    if (event == 0)
        return FFDAS_ERROR_INVALID_ARGUMENT;
    FFDAS_CHECK(handle->check_device());
    CUDA_CHECK(cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(event), handle->stream));
    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_event_synchronize(uintptr_t event) {
    if (event == 0)
        return FFDAS_ERROR_INVALID_ARGUMENT;
    CUDA_CHECK(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)));
    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_event_elapsed_time(uintptr_t start, uintptr_t stop, float *ms) {
    CHECK_NULL_PTR(ms);
    if (start == 0 || stop == 0)
        return FFDAS_ERROR_INVALID_ARGUMENT;
    CUDA_CHECK(cudaEventElapsedTime(
        ms,
        reinterpret_cast<cudaEvent_t>(start),
        reinterpret_cast<cudaEvent_t>(stop)));
    return FFDAS_SUCCESS;
}
