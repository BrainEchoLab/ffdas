#include "ffdas.h"

const char* ffdas_error_string(ffdas_error_t error) {
    switch (error) {
        case FFDAS_SUCCESS:
            return "Success";
            
        // Input validation errors
        case FFDAS_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case FFDAS_ERROR_INVALID_DIMS:
            return "Invalid dimensions";
        case FFDAS_ERROR_UNSUPPORTED_TYPE:
            return "Unsupported data type";
        case FFDAS_ERROR_NULL_POINTER:
            return "Null pointer argument";
        case FFDAS_ERROR_INVALID_HANDLE:
            return "Invalid handle";
            
        // Memory management errors
        case FFDAS_ERROR_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case FFDAS_ERROR_DEVICE_MEMORY_INSUFFICIENT:
            return "Insufficient device memory";
        case FFDAS_ERROR_HOST_MEMORY_INSUFFICIENT:
            return "Insufficient host memory";
            
        // Device/hardware errors
        case FFDAS_ERROR_DEVICE_NOT_SUPPORTED:
            return "Device not supported";
        case FFDAS_ERROR_INSUFFICIENT_COMPUTE_CAPABILITY:
            return "Insufficient compute capability";
        case FFDAS_ERROR_DEVICE_NOT_AVAILABLE:
            return "Device not available";
        case FFDAS_ERROR_DEVICE_MISMATCH:
            return "Device mismatch";
            
        // CUDA runtime errors
        case FFDAS_ERROR_CUDA_DRIVER:
            return "CUDA driver error";
        case FFDAS_ERROR_CUDA_RUNTIME:
            return "CUDA runtime error";
        case FFDAS_ERROR_CUDA_LAUNCH_FAILED:
            return "CUDA kernel launch failed";
        case FFDAS_ERROR_CUDA_SYNC_FAILED:
            return "CUDA synchronization failed";
        case FFDAS_ERROR_CUDA_MEMCPY_FAILED:
            return "CUDA memory copy failed";
            
        // Library errors
        case FFDAS_ERROR_CUBLAS:
            return "cuBLAS library error";
        case FFDAS_ERROR_CUSOLVER:
            return "cuSOLVER library error";
        case FFDAS_ERROR_CUFFT:
            return "cuFFT library error";
        case FFDAS_ERROR_NVTX:
            return "NVTX error";
            
        // Algorithm/computation errors
        case FFDAS_ERROR_ALGORITHM_FAILED:
            return "Algorithm execution failed";
        case FFDAS_ERROR_CONVERGENCE_FAILED:
            return "Algorithm failed to converge";
        case FFDAS_ERROR_NUMERICAL_INSTABILITY:
            return "Numerical instability detected";
            
        // Configuration/state errors
        case FFDAS_ERROR_NOT_INITIALIZED:
            return "Library not initialized";
        case FFDAS_ERROR_ALREADY_INITIALIZED:
            return "Library already initialized";
        case FFDAS_ERROR_INVALID_STATE:
            return "Invalid library state";
        case FFDAS_ERROR_OUT_OF_BOUNDS_INDEX:
            return "Out of bounds index";
            
        // Generic catch-all
        case FFDAS_ERROR_INTERNAL:
            return "Internal error";
        case FFDAS_ERROR_UNKNOWN:
            return "Unknown error";
            
        default:
            return "Unrecognized error code";
    }
}