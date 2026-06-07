#ifndef FFDAS_ERROR_CHECKING_H_
#define FFDAS_ERROR_CHECKING_H_

#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cufftXt.h>

#include "ffdas.h"

// Legacy exception class for backward compatibility during transition
class ffdas_runtime_error : public std::runtime_error {
public:
    explicit ffdas_runtime_error(ffdas_error_t code, const std::string &message) : std::runtime_error(message) { api_error = code; };

    ffdas_error_t api_error;
};

// Return code based error checking macros
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            return FFDAS_ERROR_CUDA_RUNTIME;                                 \
        }                                                                      \
    } while(0)

#define CUDA_DRIVER_CHECK(call)                                                \
    do {                                                                       \
        CUresult err = call;                                                   \
        if (err != CUDA_SUCCESS) {                                             \
            return FFDAS_ERROR_CUDA_DRIVER;                                  \
        }                                                                      \
    } while(0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t err = call;                                             \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            return FFDAS_ERROR_CUBLAS;                                       \
        }                                                                      \
    } while(0)

#define CUSOLVER_CHECK(call)                                                   \
    do {                                                                       \
        cusolverStatus_t err = call;                                           \
        if (err != CUSOLVER_STATUS_SUCCESS) {                                  \
            return FFDAS_ERROR_CUSOLVER;                                     \
        }                                                                      \
    } while(0)

#define CUFFT_CHECK(call)                                                      \
    do {                                                                       \
        cufftResult err = call;                                                \
        if (err != CUFFT_SUCCESS) {                                            \
            return FFDAS_ERROR_CUFFT;                                        \
        }                                                                      \
    } while(0)

// Memory allocation checking
#define CUDA_MALLOC_CHECK(ptr, size)                                          \
    do {                                                                       \
        cudaError_t err = cudaMalloc(ptr, size);                               \
        if (err != cudaSuccess) {                                              \
            if (err == cudaErrorMemoryAllocation) {                            \
                return FFDAS_ERROR_DEVICE_MEMORY_INSUFFICIENT;               \
            } else {                                                           \
                return FFDAS_ERROR_ALLOCATION_FAILED;                        \
            }                                                                  \
        }                                                                      \
    } while(0)

#define HOST_MALLOC_CHECK(ptr, size)                                          \
    do {                                                                       \
        ptr = malloc(size);                                                    \
        if (ptr == nullptr) {                                                  \
            return FFDAS_ERROR_HOST_MEMORY_INSUFFICIENT;                     \
        }                                                                      \
    } while(0)

// Argument validation macros
#define CHECK_NULL_PTR(ptr)                                                    \
    do {                                                                       \
        if (ptr == nullptr) {                                                  \
            return FFDAS_ERROR_NULL_POINTER;                                 \
        }                                                                      \
    } while(0)

#define CHECK_HANDLE(handle)                                                   \
    do {                                                                       \
        if (handle == nullptr) {                                               \
            return FFDAS_ERROR_INVALID_HANDLE;                               \
        }                                                                      \
    } while(0)

#define CHECK_POSITIVE(value)                                                  \
    do {                                                                       \
        if (value <= 0) {                                                      \
            return FFDAS_ERROR_INVALID_ARGUMENT;                             \
        }                                                                      \
    } while(0)

#define CHECK_NON_NEGATIVE(value)                                              \
    do {                                                                       \
        if (value < 0) {                                                       \
            return FFDAS_ERROR_INVALID_ARGUMENT;                             \
        }                                                                      \
    } while(0)

#ifdef FFDAS_DEBUG
#define CUDA_LAUNCH_CHECK() do { \
    cudaStreamSynchronize(handle.stream); \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) return FFDAS_ERROR_CUDA_RUNTIME; \
} while(0)
#else
#define CUDA_LAUNCH_CHECK() do { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) return FFDAS_ERROR_CUDA_RUNTIME; \
} while(0)
#endif

// Propagate error if not successful
#define FFDAS_CHECK(call)                                                    \
    do {                                                                       \
        ffdas_error_t err = call;                                            \
        if (err != FFDAS_SUCCESS) {                                          \
            return err;                                                        \
        }                                                                      \
    } while(0)

// Void function error checking macros (for internal functions that cannot return error codes)
#define CUDA_CHECK_VOID(call)                                                  \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            return;                                                            \
        }                                                                      \
    } while(0)

#define CUBLAS_CHECK_VOID(call)                                                \
    do {                                                                       \
        cublasStatus_t err = call;                                             \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            return;                                                            \
        }                                                                      \
    } while(0)

#define CUSOLVER_CHECK_VOID(call)                                              \
    do {                                                                       \
        cusolverStatus_t err = call;                                           \
        if (err != CUSOLVER_STATUS_SUCCESS) {                                  \
            return;                                                            \
        }                                                                      \
    } while(0)

#define CUFFT_CHECK_VOID(call)                                                 \
    do {                                                                       \
        cufftResult err = call;                                                \
        if (err != CUFFT_SUCCESS) {                                            \
            return;                                                            \
        }                                                                      \
    } while(0)

// Legacy exception-based macros for backward compatibility during transition
#define CUDA_THROW(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw ffdas_runtime_error(FFDAS_ERROR_CUDA_RUNTIME, "CUDA error: " + std::string(cudaGetErrorString(err)) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while(0)

#define CUDA_DRIVER_THROW(call)                                                \
    do {                                                                       \
        CUresult err = call;                                                   \
        if (err != CUDA_SUCCESS) {                                             \
            const char *errstr;                                                \
            const char *errname;                                               \
            cuGetErrorName(err, &errname);                                     \
            cuGetErrorString(err, &errstr);                                    \
            throw ffdas_runtime_error(FFDAS_ERROR_CUDA_DRIVER, "CUDA driver error " + std::string(errname) + ": " + std::string(errstr) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while(0)

#define CUBLAS_THROW(call)                                                     \
    do {                                                                       \
        cublasStatus_t err = call;                                             \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            throw ffdas_runtime_error(FFDAS_ERROR_CUBLAS, "cuBLAS error: " + std::string(cublasGetErrorString(err)) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while(0)

#define CUSOLVER_THROW(call)                                                   \
    do {                                                                       \
        cusolverStatus_t err = call;                                           \
        if (err != CUSOLVER_STATUS_SUCCESS) {                                  \
            throw ffdas_runtime_error(FFDAS_ERROR_CUSOLVER, "cuSOLVER error: " + std::string(cusolverGetErrorString(err)) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while(0)

#define CUFFT_THROW(call)                                                      \
    do {                                                                       \
        cufftResult err = call;                                                \
        if (err != CUFFT_SUCCESS) {                                            \
            throw ffdas_runtime_error(FFDAS_ERROR_CUFFT, "cuFFT error: " + std::string(cufftGetErrorString(err)) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while(0)

// Function to convert cuBLAS errors to a human-readable string
static const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "Unknown cuBLAS error";
    }
}

// Function to convert cuSOLVER errors to a human-readable string
static const char* cusolverGetErrorString(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
        default:
            return "Unknown cuSOLVER error";
    }
}

// Function to convert cuFFT errors to a human-readable string
static const char* cufftGetErrorString(cufftResult status) {
    switch (status) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
        default:
            return "Unknown cuFFT error";
    }
}

#endif  // FFDAS_ERROR_CHECKING_H_