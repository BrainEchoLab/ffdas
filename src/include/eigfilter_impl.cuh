#pragma once

#include <cstdint>
#include <climits>

#include "ffdas.h"
#include "context.cuh"
#include "error_checking.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "tensor.cuh"
#include "type_utils.h"


namespace ffdas::detail {


template<typename T, typename Tscal>
ffdas_error_t eigfilter_execute(
    ffdas_context &handle,
    int m, int n,
    int k0, int k1,
    const T *x,
    Tscal *w,
    T *V,
    T *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    T *y
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}

template<>
ffdas_error_t eigfilter_execute(
    ffdas_context &handle,
    int m, int n,
    int k0, int k1,
    const float *x,
    float *w,
    float *V,
    float *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    float *y
) {
    float alpha = 1.0f / (float)m;
    float beta = 0.0f;

    // SGEMM instead of SSYRK for X^T X: avoids rounding errors observed
    // in SSYRK when m >> n.
    CUBLAS_CHECK(cublasSgemm(handle.cublas_h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, n, m,
        &alpha,
        x, m,
        x, m,
        &beta,
        V, n
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n,
        CUDA_R_32F, V, n,
        CUDA_R_32F, w,
        CUDA_R_32F,
        d_syevdbuf, d_syevdbuf_size,
        h_syevdbuf, h_syevdbuf_size,
        d_info.get()
    ));

    int h_info;
    CUDA_CHECK(cudaMemcpyAsync(&h_info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost, handle.stream));
    CUDA_CHECK(cudaStreamSynchronize(handle.stream));

    if (h_info != 0)
        return FFDAS_ERROR_CUSOLVER;

    alpha = 1.0f;

    // P = V_k * V_k^T (projection onto eigenvectors k0..k1)
    CUBLAS_CHECK(cublasSsyrk(
        handle.cublas_h,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        n, k1 - k0,
        &alpha,
        V + (n - k1) * n, n,
        &beta,
        P, n
    ));

    // y = x * P
    CUBLAS_CHECK(cublasSsymm(
        handle.cublas_h,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        m, n,
        &alpha,
        P, n,
        x, m,
        &beta,
        y, m
    ));

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t eigfilter_execute(
    ffdas_context &handle,
    int m, int n,
    int k0, int k1,
    const float2 *x,
    float *w,
    float2 *V,
    float2 *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    float2 *y
) {
    float2 calpha = make_float2(1.0f / (float)m, 0.0f);
    float2 cbeta = make_float2(0.0f, 0.0f);

    // CGEMM instead of CHERK for X^H X: avoids rounding errors observed
    // in CHERK when m >> n.
    CUBLAS_CHECK(cublasCgemm3m(handle.cublas_h,
        CUBLAS_OP_C, CUBLAS_OP_N,
        n, n, m,
        &calpha,
        x, m,
        x, m,
        &cbeta,
        V, n
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n,
        CUDA_C_32F, V, n,
        CUDA_R_32F, w,
        CUDA_C_32F,
        d_syevdbuf, d_syevdbuf_size,
        h_syevdbuf, h_syevdbuf_size,
        d_info.get()
    ));

    int h_info;
    CUDA_CHECK(cudaMemcpyAsync(&h_info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost, handle.stream));
    CUDA_CHECK(cudaStreamSynchronize(handle.stream));

    if (h_info != 0)
        return FFDAS_ERROR_CUSOLVER;

    float falpha = 1.0f;
    float fbeta = 0.0f;

    // P = V_k * V_k^H (projection onto eigenvectors k0..k1)
    CUBLAS_CHECK(cublasCherk(
        handle.cublas_h,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        n, k1 - k0,
        &falpha,
        V + (n - k1) * n, n,
        &fbeta,
        P, n
    ));

    calpha = make_float2(1.0f, 0.0f);

    // y = x * P
    CUBLAS_CHECK(cublasChemm(
        handle.cublas_h,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        m, n,
        &calpha,
        P, n,
        x, m,
        &cbeta,
        y, m
    ));

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t eigfilter_execute(
    ffdas_context &handle,
    int m, int n,
    int k0, int k1,
    const double *x,
    double *w,
    double *V,
    double *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    double *y
) {
    double alpha = 1.0 / (double)m;
    double beta = 0.0;

    CUBLAS_CHECK(cublasDgemm(handle.cublas_h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, n, m,
        &alpha,
        x, m,
        x, m,
        &beta,
        V, n
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n,
        CUDA_R_64F, V, n,
        CUDA_R_64F, w,
        CUDA_R_64F,
        d_syevdbuf, d_syevdbuf_size,
        h_syevdbuf, h_syevdbuf_size,
        d_info.get()
    ));

    int h_info;
    CUDA_CHECK(cudaMemcpyAsync(&h_info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost, handle.stream));
    CUDA_CHECK(cudaStreamSynchronize(handle.stream));

    if (h_info != 0)
        return FFDAS_ERROR_CUSOLVER;

    alpha = 1.0;

    CUBLAS_CHECK(cublasDsyrk(
        handle.cublas_h,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        n, k1 - k0,
        &alpha,
        V + (n - k1) * n, n,
        &beta,
        P, n
    ));

    CUBLAS_CHECK(cublasDsymm(
        handle.cublas_h,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        m, n,
        &alpha,
        P, n,
        x, m,
        &beta,
        y, m
    ));

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t eigfilter_execute(
    ffdas_context &handle,
    int m, int n,
    int k0, int k1,
    const double2 *x,
    double *w,
    double2 *V,
    double2 *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    double2 *y
) {
    cuDoubleComplex calpha = make_cuDoubleComplex(1.0 / (double)m, 0.0);
    cuDoubleComplex cbeta = make_cuDoubleComplex(0.0, 0.0);

    CUBLAS_CHECK(cublasZgemm(handle.cublas_h,
        CUBLAS_OP_C, CUBLAS_OP_N,
        n, n, m,
        &calpha,
        reinterpret_cast<const cuDoubleComplex*>(x), m,
        reinterpret_cast<const cuDoubleComplex*>(x), m,
        &cbeta,
        reinterpret_cast<cuDoubleComplex*>(V), n
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n,
        CUDA_C_64F, V, n,
        CUDA_R_64F, w,
        CUDA_C_64F,
        d_syevdbuf, d_syevdbuf_size,
        h_syevdbuf, h_syevdbuf_size,
        d_info.get()
    ));

    int h_info;
    CUDA_CHECK(cudaMemcpyAsync(&h_info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost, handle.stream));
    CUDA_CHECK(cudaStreamSynchronize(handle.stream));

    if (h_info != 0)
        return FFDAS_ERROR_CUSOLVER;

    double dalpha = 1.0;
    double dbeta = 0.0;

    CUBLAS_CHECK(cublasZherk(
        handle.cublas_h,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        n, k1 - k0,
        &dalpha,
        reinterpret_cast<const cuDoubleComplex*>(V) + (n - k1) * n, n,
        &dbeta,
        reinterpret_cast<cuDoubleComplex*>(P), n
    ));

    calpha = make_cuDoubleComplex(1.0, 0.0);

    CUBLAS_CHECK(cublasZhemm(
        handle.cublas_h,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        m, n,
        &calpha,
        reinterpret_cast<const cuDoubleComplex*>(P), n,
        reinterpret_cast<const cuDoubleComplex*>(x), m,
        &cbeta,
        reinterpret_cast<cuDoubleComplex*>(y), m
    ));

    return FFDAS_SUCCESS;
}


// Eigenfilter: project x onto its top-k eigenvectors.
// x and y are column-major with shape (n, m) where n = channels, m = samples.
template<typename T>
ffdas_error_t eigfilter_impl(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc, const T* x,
    const ffdas_tensor_desc &y_desc, T* y,
    int k0, int k1
) {
    if (x_desc.ndim() != 2)
        return FFDAS_ERROR_INVALID_DIMS;
    if (!x_desc.same_dims(y_desc))
        return FFDAS_ERROR_INVALID_DIMS;
    if (x_desc.dtype != y_desc.dtype)
        return FFDAS_ERROR_UNSUPPORTED_TYPE;

    int64_t n64 = x_desc.dims[0];
    int64_t m64 = x_desc.dims[1];
    if (n64 > INT_MAX || m64 > INT_MAX)
        return FFDAS_ERROR_INVALID_DIMS;

    int n = static_cast<int>(n64);
    int m = static_cast<int>(m64);

    using w_type = typename builtin_traits<T>::scalar_type;

    device_ptr<T> V(handle);
    device_ptr<T> P(handle);
    device_ptr<w_type> w(handle);

    FFDAS_CHECK(V.alloc(n * n * sizeof(T)));
    FFDAS_CHECK(P.alloc(n * n * sizeof(T)));
    FFDAS_CHECK(w.alloc(n * sizeof(w_type)));

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        n,
        builtin_traits<T>::cuda_datatype, V.get(), n,
        builtin_traits<w_type>::cuda_datatype, w.get(),
        builtin_traits<T>::cuda_datatype,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost
    ));

    device_ptr<void> bufferOnDevice(handle);
    // n * workspaceInBytesOnDevice: empirical overallocation to work around
    // cuSOLVER underestimates on some architectures
    FFDAS_CHECK(bufferOnDevice.alloc(n * workspaceInBytesOnDevice));

    void *bufferOnHost = malloc(workspaceInBytesOnHost);
    if (!bufferOnHost)
        return FFDAS_ERROR_FAILED_MALLOC;

    ffdas_error_t err = eigfilter_execute<T, w_type>(
        handle,
        m, n, k0, k1,
        x,
        w.get(), V.get(), P.get(),
        bufferOnDevice.get(), workspaceInBytesOnDevice,
        bufferOnHost, workspaceInBytesOnHost,
        y
    );

    free(bufferOnHost);
    return err;
}


template<ffdas_datatype_t T_t>
ffdas_error_t eigfilter_dispatch(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc, const void* x,
    const ffdas_tensor_desc &y_desc, void* y,
    int k0, int k1
) {
    using T = typename ffdas_traits<T_t>::type;
    return eigfilter_impl<T>(handle, x_desc, static_cast<const T*>(x), y_desc, static_cast<T*>(y), k0, k1);
}


}  // namespace ffdas::detail
