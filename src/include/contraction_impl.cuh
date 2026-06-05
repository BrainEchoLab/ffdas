#pragma once

#include <vector>
#include <cstdint>

#include "ffdas.h"
#include "context.cuh"
#include "tensor.cuh"
#include "type_utils.h"


enum { MODE_BROADCAST=0, MODE_BATCH=1, MODE_REDUCE=2 };


// Precomputed plan for out = contraction(a, b) expressed as b batched GEMM.
//
// The contraction is decomposed into: (1) optionally permute a and/or b into
// b memory layout compatible with batched GEMM, (2) call cuBLAS, (3) optionally
// permute the GEMM output into the user's requested out layout.
struct ffdas_contraction_plan {
    ffdas_tensor_desc a_desc, b_desc, out_desc;

    // if swap_ab is true, a is passed as cuBLAS's A matrix and b as B
    // (rather than the default b=A, a=B)
    bool swap_ab;

    // whether each operand/output requires b permutation copy
    bool do_aperm, do_bperm, do_outperm;
    std::vector<int64_t> pa, pb, pout;

    // cuBLAS parameters (m, n, k, leading dims, strides, batch count)
    bool transa, transb;
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    int64_t strideA, strideB, strideC;
    int64_t batchCount;

    // device workspace for permuted copies (allocated only when do_*perm is true)
    ffdas::detail::device_ptr<void> a_work;
    ffdas::detail::device_ptr<void> b_work;
    ffdas::detail::device_ptr<void> out_work;
};


namespace ffdas::detail {

template<typename T>
ffdas_error_t contraction_impl(
    ffdas_context &handle,
    ffdas_contraction_plan &plan,
    const T *a,
    const T *b,
    T *out
);

template<ffdas_datatype_t Tx_t>
ffdas_error_t ffdas_contraction_dispatch(
    ffdas_context &handle,
    ffdas_contraction_plan &plan,
    const void *a,
    const void *b,
    void *out
) {
    using T = typename ffdas_traits<Tx_t>::type;

    return contraction_impl<T>(handle, plan, static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(out));
}

}  // namespace ffdas::detail
