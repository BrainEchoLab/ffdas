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
    int nrows, int ncols,
    int proj_dim,
    bool project_rows,
    int start, int stop,
    const T *x,
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
    int nrows, int ncols,
    int proj_dim,
    bool project_rows,
    int start, int stop,
    const float *x,
    float *w,
    float *V,
    float *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    float *out
) {
    float alpha = 1.0f / (float)nrows;
    float beta = 0.0f;
    int K = project_rows ? ncols : nrows;

    // GEMM instead of SYRK for the Gram matrix: avoids rounding errors
    // observed in SSYRK when one dimension is much larger than the other.
    CUBLAS_CHECK(cublasSgemm(handle.cublas_h,
        project_rows ? CUBLAS_OP_N : CUBLAS_OP_T,
        project_rows ? CUBLAS_OP_T : CUBLAS_OP_N,
        proj_dim, proj_dim, K,
        &alpha,
        x, nrows,
        x, nrows,
        &beta,
        V, proj_dim
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        proj_dim,
        CUDA_R_32F, V, proj_dim,
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

    // P = V_k V_k^T, projection onto selected eigenvectors.
    // Eigenvalues are ascending; SVD index 0 = largest eigenvalue = last
    // column of V. So SVD range [start, stop) maps to columns
    // [proj_dim - stop, proj_dim - start).
    CUBLAS_CHECK(cublasSsyrk(
        handle.cublas_h,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        proj_dim, stop - start,
        &alpha,
        V + (proj_dim - stop) * proj_dim, proj_dim,
        &beta,
        P, proj_dim
    ));

    if (project_rows) {
        CUBLAS_CHECK(cublasSsymm(
            handle.cublas_h,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            nrows, ncols,
            &alpha,
            P, proj_dim,
            x, nrows,
            &beta,
            out, nrows
        ));
    } else {
        CUBLAS_CHECK(cublasSsymm(
            handle.cublas_h,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            nrows, ncols,
            &alpha,
            P, proj_dim,
            x, nrows,
            &beta,
            out, nrows
        ));
    }

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t truncate_rank_execute(
    ffdas_context &handle,
    int nrows, int ncols,
    int proj_dim,
    bool project_rows,
    int start, int stop,
    const float2 *x,
    float *w,
    float2 *V,
    float2 *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    float2 *out
) {
    float2 calpha = make_float2(1.0f / (float)nrows, 0.0f);
    float2 cbeta = make_float2(0.0f, 0.0f);
    int K = project_rows ? ncols : nrows;

    // CGEMM instead of CHERK: avoids rounding errors observed in CHERK
    // when one dimension is much larger than the other.
    CUBLAS_CHECK(cublasCgemm3m(handle.cublas_h,
        project_rows ? CUBLAS_OP_N : CUBLAS_OP_C,
        project_rows ? CUBLAS_OP_C : CUBLAS_OP_N,
        proj_dim, proj_dim, K,
        &calpha,
        x, nrows,
        x, nrows,
        &cbeta,
        V, proj_dim
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        proj_dim,
        CUDA_C_32F, V, proj_dim,
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

    CUBLAS_CHECK(cublasCherk(
        handle.cublas_h,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        proj_dim, stop - start,
        &falpha,
        V + (proj_dim - stop) * proj_dim, proj_dim,
        &fbeta,
        P, proj_dim
    ));

    calpha = make_float2(1.0f, 0.0f);

    if (project_rows) {
        CUBLAS_CHECK(cublasChemm(
            handle.cublas_h,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            nrows, ncols,
            &calpha,
            P, proj_dim,
            x, nrows,
            &cbeta,
            out, nrows
        ));
    } else {
        CUBLAS_CHECK(cublasChemm(
            handle.cublas_h,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            nrows, ncols,
            &calpha,
            P, proj_dim,
            x, nrows,
            &cbeta,
            out, nrows
        ));
    }

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t truncate_rank_execute(
    ffdas_context &handle,
    int nrows, int ncols,
    int proj_dim,
    bool project_rows,
    int start, int stop,
    const double *x,
    double *w,
    double *V,
    double *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    double *out
) {
    double alpha = 1.0 / (double)nrows;
    double beta = 0.0;
    int K = project_rows ? ncols : nrows;

    CUBLAS_CHECK(cublasDgemm(handle.cublas_h,
        project_rows ? CUBLAS_OP_N : CUBLAS_OP_T,
        project_rows ? CUBLAS_OP_T : CUBLAS_OP_N,
        proj_dim, proj_dim, K,
        &alpha,
        x, nrows,
        x, nrows,
        &beta,
        V, proj_dim
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        proj_dim,
        CUDA_R_64F, V, proj_dim,
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
        proj_dim, stop - start,
        &alpha,
        V + (proj_dim - stop) * proj_dim, proj_dim,
        &beta,
        P, proj_dim
    ));

    if (project_rows) {
        CUBLAS_CHECK(cublasDsymm(
            handle.cublas_h,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            nrows, ncols,
            &alpha,
            P, proj_dim,
            x, nrows,
            &beta,
            out, nrows
        ));
    } else {
        CUBLAS_CHECK(cublasDsymm(
            handle.cublas_h,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            nrows, ncols,
            &alpha,
            P, proj_dim,
            x, nrows,
            &beta,
            out, nrows
        ));
    }

    return FFDAS_SUCCESS;
}

template<>
ffdas_error_t truncate_rank_execute(
    ffdas_context &handle,
    int nrows, int ncols,
    int proj_dim,
    bool project_rows,
    int start, int stop,
    const double2 *x,
    double *w,
    double2 *V,
    double2 *P,
    void *d_syevdbuf,
    size_t d_syevdbuf_size,
    void *h_syevdbuf,
    size_t h_syevdbuf_size,
    double2 *out
) {
    cuDoubleComplex calpha = make_cuDoubleComplex(1.0 / (double)nrows, 0.0);
    cuDoubleComplex cbeta = make_cuDoubleComplex(0.0, 0.0);
    int K = project_rows ? ncols : nrows;

    CUBLAS_CHECK(cublasZgemm(handle.cublas_h,
        project_rows ? CUBLAS_OP_N : CUBLAS_OP_C,
        project_rows ? CUBLAS_OP_C : CUBLAS_OP_N,
        proj_dim, proj_dim, K,
        &calpha,
        reinterpret_cast<const cuDoubleComplex*>(x), nrows,
        reinterpret_cast<const cuDoubleComplex*>(x), nrows,
        &cbeta,
        reinterpret_cast<cuDoubleComplex*>(V), proj_dim
    ));

    device_ptr<int> d_info(handle);
    FFDAS_CHECK(d_info.alloc(sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXsyevd(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        proj_dim,
        CUDA_C_64F, V, proj_dim,
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
        proj_dim, stop - start,
        &dalpha,
        reinterpret_cast<const cuDoubleComplex*>(V) + (proj_dim - stop) * proj_dim, proj_dim,
        &dbeta,
        reinterpret_cast<cuDoubleComplex*>(P), proj_dim
    ));

    calpha = make_cuDoubleComplex(1.0, 0.0);

    if (project_rows) {
        CUBLAS_CHECK(cublasZhemm(
            handle.cublas_h,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            nrows, ncols,
            &calpha,
            reinterpret_cast<const cuDoubleComplex*>(P), proj_dim,
            reinterpret_cast<const cuDoubleComplex*>(x), nrows,
            &cbeta,
            reinterpret_cast<cuDoubleComplex*>(out), nrows
        ));
    } else {
        CUBLAS_CHECK(cublasZhemm(
            handle.cublas_h,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            nrows, ncols,
            &calpha,
            reinterpret_cast<const cuDoubleComplex*>(P), proj_dim,
            reinterpret_cast<const cuDoubleComplex*>(x), nrows,
            &cbeta,
            reinterpret_cast<cuDoubleComplex*>(out), nrows
        ));
    }

    return FFDAS_SUCCESS;
}


// Reconstruct x from a subset of its singular vectors.
//
// x is row-major with dims (dims[0], dims[1]). cuBLAS sees this as a
// column-major (nrows x ncols) matrix with ld = nrows, where
// nrows = dims[1] and ncols = dims[0].
//
// To avoid forming a large Gram matrix, the eigendecomposition is performed
// on the smaller of X^H X (ncols x ncols) or X X^H (nrows x nrows).
// project_rows tracks which was used, determining whether the projection
// is applied as a left- or right-multiply.
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

    if (x_desc.dims[0] > INT_MAX || x_desc.dims[1] > INT_MAX)
        return FFDAS_ERROR_INVALID_DIMS;

    int nrows = static_cast<int>(x_desc.dims[1]);
    int ncols = static_cast<int>(x_desc.dims[0]);
    bool project_rows = nrows < ncols;
    int proj_dim = project_rows ? nrows : ncols;

    using w_type = typename builtin_traits<T>::scalar_type;

    device_ptr<T> V(handle);
    device_ptr<T> P(handle);
    device_ptr<w_type> w(handle);

    FFDAS_CHECK(V.alloc(proj_dim * proj_dim * sizeof(T)));
    FFDAS_CHECK(P.alloc(proj_dim * proj_dim * sizeof(T)));
    FFDAS_CHECK(w.alloc(proj_dim * sizeof(w_type)));

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;

    CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
        handle.cusolver_h,
        NULL,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        proj_dim,
        builtin_traits<T>::cuda_datatype, V.get(), proj_dim,
        builtin_traits<w_type>::cuda_datatype, w.get(),
        builtin_traits<T>::cuda_datatype,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost
    ));

    device_ptr<void> bufferOnDevice(handle);
    // Empirical overallocation to work around cuSOLVER buffer size
    // underestimates on some architectures.
    FFDAS_CHECK(bufferOnDevice.alloc(proj_dim * workspaceInBytesOnDevice));

    void *bufferOnHost = malloc(workspaceInBytesOnHost);
    if (!bufferOnHost)
        return FFDAS_ERROR_ALLOCATION_FAILED;

    ffdas_error_t err = truncate_rank_execute<T, w_type>(
        handle,
        nrows, ncols, proj_dim, project_rows,
        start, stop,
        x,
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
