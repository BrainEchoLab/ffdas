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
ffdas_error_t truncate_rank_execute(
    ffdas_context &handle,
    int m, int n,
    int start, int stop,
    const T *x,
    int ldx,
    Tscal *w,
    T *V,
    T *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    T *out
) {
    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}

template<>
ffdas_error_t truncate_rank_execute(
    ffdas_context &handle,
    int m, int n,
    int start, int stop,
    const float *x,
    int ldx,
    float *w,
    float *V,
    float *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    float *out
) {
    float alpha = 1.0f / (float)m;
    float beta = 0.0f;

    bool trans = n > m;
    int rows = trans ? n : m;
    int cols = trans ? m : n;

    // SGEMM instead of SSYRK for X^T X: avoids rounding errors observed
    // in SSYRK when m >> n.
    CUBLAS_CHECK(cublasSgemm(handle.cublas_h,
        trans ? CUBLAS_OP_N : CUBLAS_OP_T, trans ? CUBLAS_OP_T : CUBLAS_OP_N,
        cols, cols, rows,
        &alpha,
        x, ldx,
        x, ldx,
        &beta,
        V, cols
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        cols,
        CUDA_R_32F, V, cols,
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

    // P = V_k * V_k^T (projection onto eigenvectors start..stop)
    CUBLAS_CHECK(cublasSsyrk(
        handle.cublas_h,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        cols, stop - start,
        &alpha,
        V + (cols - stop) * cols, cols,
        &beta,
        P, cols
    ));

    // out = x * P
    CUBLAS_CHECK(cublasSsymm(
        handle.cublas_h,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        rows, cols,
        &alpha,
        P, cols,
        x, rows,
        &beta,
        out, rows
    ));

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t truncate_rank_execute(
    ffdas_context &handle,
    int m, int n,
    int start, int stop,
    const float2 *x,
    int ldx,
    float *w,
    float2 *V,
    float2 *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    float2 *out
) {
    float2 calpha = make_float2(1.0f / (float)m, 0.0f);
    float2 cbeta = make_float2(0.0f, 0.0f);

    bool trans = n > m;
    int rows = trans ? n : m;
    int cols = trans ? m : n;

    // CGEMM instead of CHERK for X^H X: avoids rounding errors observed
    // in CHERK when m >> n.
    CUBLAS_CHECK(cublasCgemm3m(handle.cublas_h,
        trans ? CUBLAS_OP_N : CUBLAS_OP_C, trans ? CUBLAS_OP_C : CUBLAS_OP_N,
        cols, cols, rows,
        &calpha,
        x, ldx,
        x, ldx,
        &cbeta,
        V, cols
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        cols,
        CUDA_C_32F, V, cols,
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

    // P = V_k * V_k^H (projection onto eigenvectors start..stop)
    CUBLAS_CHECK(cublasCherk(
        handle.cublas_h,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        cols, stop - start,
        &falpha,
        V + (cols - stop) * cols, cols,
        &fbeta,
        P, cols
    ));

    calpha = make_float2(1.0f, 0.0f);

    // out = x * P
    CUBLAS_CHECK(cublasChemm(
        handle.cublas_h,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        rows, cols,
        &calpha,
        P, cols,
        x, rows,
        &cbeta,
        out, rows
    ));

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t truncate_rank_execute(
    ffdas_context &handle,
    int m, int n,
    int start, int stop,
    const double *x,
    int ldx,
    double *w,
    double *V,
    double *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    double *out
) {
    double alpha = 1.0 / (double)m;
    double beta = 0.0;

    bool trans = n > m;
    int rows = trans ? n : m;
    int cols = trans ? m : n;

    CUBLAS_CHECK(cublasDgemm(handle.cublas_h,
        trans ? CUBLAS_OP_N : CUBLAS_OP_T, trans ? CUBLAS_OP_T : CUBLAS_OP_N,
        cols, cols, rows,
        &alpha,
        x, ldx,
        x, ldx,
        &beta,
        V, cols
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        cols,
        CUDA_R_64F, V, cols,
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
        cols, stop - start,
        &alpha,
        V + (cols - stop) * cols, cols,
        &beta,
        P, cols
    ));

    CUBLAS_CHECK(cublasDsymm(
        handle.cublas_h,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        rows, cols,
        &alpha,
        P, cols,
        x, rows,
        &beta,
        out, rows
    ));

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t truncate_rank_execute(
    ffdas_context &handle,
    int m, int n,
    int start, int stop,
    const double2 *x,
    int ldx,
    double *w,
    double2 *V,
    double2 *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    double2 *out
) {
    cuDoubleComplex calpha = make_cuDoubleComplex(1.0 / (double)m, 0.0);
    cuDoubleComplex cbeta = make_cuDoubleComplex(0.0, 0.0);

    bool trans = n > m;
    int rows = trans ? n : m;
    int cols = trans ? m : n;

    CUBLAS_CHECK(cublasZgemm(handle.cublas_h,
        trans ? CUBLAS_OP_N : CUBLAS_OP_C, trans ? CUBLAS_OP_C : CUBLAS_OP_N,
        cols, cols, rows,
        &calpha,
        reinterpret_cast<const cuDoubleComplex*>(x), ldx,
        reinterpret_cast<const cuDoubleComplex*>(x), ldx,
        &cbeta,
        reinterpret_cast<cuDoubleComplex*>(V), cols
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        cols,
        CUDA_C_64F, V, cols,
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
        cols, stop - start,
        &dalpha,
        reinterpret_cast<const cuDoubleComplex*>(V) + (cols - stop) * cols, cols,
        &dbeta,
        reinterpret_cast<cuDoubleComplex*>(P), cols
    ));

    calpha = make_cuDoubleComplex(1.0, 0.0);

    CUBLAS_CHECK(cublasZhemm(
        handle.cublas_h,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
        rows, cols,
        &calpha,
        reinterpret_cast<const cuDoubleComplex*>(P), cols,
        reinterpret_cast<const cuDoubleComplex*>(x), rows,
        &cbeta,
        reinterpret_cast<cuDoubleComplex*>(out), rows
    ));

    return FFDAS_SUCCESS;
}


// Project x onto its top-k eigenvectors.
// x and out are column-major with shape (n, m) where n = channels, m = samples.
template<typename T>
ffdas_error_t truncate_rank_impl(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc, const T* x,
    const ffdas_tensor_desc &out_desc, T* out,
    int start, int stop
) {
    if (x_desc.ndim() != 2)
        return FFDAS_ERROR_INVALID_DIMS;
    if (!x_desc.same_dims(out_desc))
        return FFDAS_ERROR_INVALID_DIMS;
    if (x_desc.dtype != out_desc.dtype)
        return FFDAS_ERROR_UNSUPPORTED_TYPE;

    int64_t n64 = x_desc.dims[0];
    int64_t m64 = x_desc.dims[1];
    if (n64 > INT_MAX || m64 > INT_MAX)
        return FFDAS_ERROR_INVALID_DIMS;

    int n = static_cast<int>(n64);
    int m = static_cast<int>(m64);
    int ncol = (m > n) ? n : m;

    using w_type = typename builtin_traits<T>::scalar_type;

    device_ptr<T> V(handle);
    device_ptr<T> P(handle);
    device_ptr<w_type> w(handle);

    FFDAS_CHECK(V.alloc(ncol * ncol * sizeof(T)));
    FFDAS_CHECK(P.alloc(ncol * ncol * sizeof(T)));
    FFDAS_CHECK(w.alloc(ncol * sizeof(w_type)));

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        ncol,
        builtin_traits<T>::cuda_datatype, V.get(), ncol,
        builtin_traits<w_type>::cuda_datatype, w.get(),
        builtin_traits<T>::cuda_datatype,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost
    ));

    device_ptr<void> bufferOnDevice(handle);
    // n * workspaceInBytesOnDevice: empirical overallocation to work around
    // cuSOLVER underestimates on some architectures
    FFDAS_CHECK(bufferOnDevice.alloc(ncol * workspaceInBytesOnDevice));

    void *bufferOnHost = malloc(workspaceInBytesOnHost);
    if (!bufferOnHost)
        return FFDAS_ERROR_ALLOCATION_FAILED;

    ffdas_error_t err = truncate_rank_execute<T, w_type>(
        handle,
        m, n, start, stop,
        x, m,
        w.get(), V.get(), P.get(),
        bufferOnDevice.get(), workspaceInBytesOnDevice,
        bufferOnHost, workspaceInBytesOnHost,
        out
    );

    free(bufferOnHost);
    return err;
}


template<ffdas_datatype_t T_t>
ffdas_error_t truncate_rank_dispatch(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc, const void* x,
    const ffdas_tensor_desc &out_desc, void* out,
    int start, int stop
) {
    using T = typename ffdas_traits<T_t>::type;
    return truncate_rank_impl<T>(handle, x_desc, static_cast<const T*>(x), out_desc, static_cast<T*>(out), start, stop);
}


}  // namespace ffdas::detail
