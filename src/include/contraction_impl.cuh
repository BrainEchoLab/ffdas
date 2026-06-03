#pragma once

#include <vector>
#include <cstdint>

#include "ffdas.h"
#include "context.cuh"
#include "tensor.cuh"
#include "type_utils.h"


enum { MODE_BROADCAST=0, MODE_BATCH=1, MODE_REDUCE=2 };


// Precomputed plan for y = contraction(x, a) expressed as a batched GEMM.
//
// The contraction is decomposed into: (1) optionally permute x and/or a into
// a memory layout compatible with batched GEMM, (2) call cuBLAS, (3) optionally
// permute the GEMM output into the user's requested y layout.
struct ffdas_contraction_plan {
    ffdas_tensor_desc x_desc, a_desc, y_desc;

    // if swap_ax is true, x is passed as cuBLAS's A matrix and a as B
    // (rather than the default a=A, x=B)
    bool swap_ax;

    // whether each operand/output requires a permutation copy
    bool do_xperm, do_aperm, do_yperm;
    std::vector<int64_t> px, pa, py;

    // cuBLAS parameters (m, n, k, leading dims, strides, batch count)
    bool transa, transb;
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    int64_t strideA, strideB, strideC;
    int64_t batchCount;

    // device workspace for permuted copies (allocated only when do_*perm is true)
    ffdas::detail::device_ptr<void> x_work;
    ffdas::detail::device_ptr<void> a_work;
    ffdas::detail::device_ptr<void> y_work;
};


namespace ffdas::detail {

template<typename T>
ffdas_error_t contraction_impl(
    ffdas_context &handle,
    ffdas_contraction_plan &plan,
    const T *x,
    const T *a,
    T *y
);

template<ffdas_datatype_t Tx_t>
ffdas_error_t ffdas_contraction_dispatch(
    ffdas_context &handle,
    ffdas_contraction_plan &plan,
    const void *x,
    const void *a,
    void *y
) {
    using T = typename ffdas_traits<Tx_t>::type;

    return contraction_impl<T>(handle, plan, static_cast<const T*>(x), static_cast<const T*>(a), static_cast<T*>(y));
}

}  // namespace ffdas::detail
