#include "contraction_impl.cuh"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <climits>

#include "ffdas.h"
#include "context.cuh"
#include "type_utils.h"
#include "tensor.cuh"
#include "contiguous_copy_impl.cuh"


namespace ffdas::detail {


// Classify each mode as broadcast (appears in one operand), reduce (appears
// in both operands but not the output), or batch (appears in both operands
// and the output).
static ffdas_error_t classify_modes(
    const std::vector<int>& a_modes,
    const std::vector<int>& b_modes,
    const std::vector<int>& out_modes,
    std::unordered_map<int, int>& flags
) {
    flags.clear();

    for (int m : b_modes) flags[m] = MODE_BROADCAST;
    for (int m : a_modes) flags[m] = flags.count(m) ? MODE_REDUCE : MODE_BROADCAST;
    for (int m : out_modes) {
        auto it = flags.find(m);
        if (it == flags.end())
            return FFDAS_ERROR_INVALID_ARGUMENT;
        if (it->second == MODE_REDUCE) it->second = MODE_BATCH;
    }

    return FFDAS_SUCCESS;
}


// Validate that dimension sizes agree across operands for each shared mode.
static ffdas_error_t validate_mode_dims(
    const std::vector<int64_t>& a_dims, const std::unordered_map<int, int>& a_map,
    const std::vector<int64_t>& b_dims, const std::unordered_map<int, int>& b_map,
    const std::vector<int64_t>& out_dims, const std::unordered_map<int, int>& out_map,
    const std::unordered_map<int, int>& flags
) {
    for (const auto& [mode, flag] : flags) {
        auto ai = a_map.find(mode);
        auto bi = b_map.find(mode);
        auto outi = out_map.find(mode);

        if (flag == MODE_REDUCE) {
            if (ai == a_map.end() || bi == b_map.end())
                return FFDAS_ERROR_INVALID_ARGUMENT;
            if (a_dims[ai->second] != b_dims[bi->second])
                return FFDAS_ERROR_INVALID_ARGUMENT;
        } else if (flag == MODE_BATCH) {
            if (ai == a_map.end() || bi == b_map.end() || outi == out_map.end())
                return FFDAS_ERROR_INVALID_ARGUMENT;
            int64_t d = a_dims[ai->second];
            if (b_dims[bi->second] != d || out_dims[outi->second] != d)
                return FFDAS_ERROR_INVALID_ARGUMENT;
        } else {
            if (outi == out_map.end())
                return FFDAS_ERROR_INVALID_ARGUMENT;
            if (ai != a_map.end() && a_dims[ai->second] != out_dims[outi->second])
                return FFDAS_ERROR_INVALID_ARGUMENT;
            if (bi != b_map.end() && b_dims[bi->second] != out_dims[outi->second])
                return FFDAS_ERROR_INVALID_ARGUMENT;
        }
    }

    return FFDAS_SUCCESS;
}


static ffdas_error_t group_modes(
    const std::vector<int> &modes,
    const std::unordered_map<int, int> &flags,
    std::vector<int> &broadcast,
    std::vector<int> &batch,
    std::vector<int> &reduce
) {
    broadcast.clear();
    batch.clear();
    reduce.clear();

    for (int mode : modes) {
        int flag = flags.at(mode);
        if (flag == MODE_BATCH) batch.push_back(mode);
        else if (flag == MODE_BROADCAST) broadcast.push_back(mode);
        else if (flag == MODE_REDUCE) reduce.push_back(mode);
    }

    return FFDAS_SUCCESS;
}


// One candidate memory layout for the two operands. Each layout specifies where
// the batch and reduce dimensions sit relative to the free (broadcast) dimensions,
// plus the permutation needed to get each operand into that layout.
struct batched_gemm_layout {
    bool batch_first_a, reduce_last_a, batch_first_b, reduce_last_b;
    std::vector<int64_t> a_perm, b_perm;
    std::vector<int> a_free, b_free;
    std::vector<int> batch, reduce;
};


// For each mode label in query_order, find its position in reference_modes.
// Output is b permutation (int64_t indices into reference_modes).
static ffdas_error_t modes_to_permutation(
    const std::vector<int> &reference_modes,
    const std::vector<int> &query_order,
    std::vector<int64_t> &perm
) {
    perm.clear();
    perm.reserve(query_order.size());

    for (int mode : query_order) {
        auto it = std::find(reference_modes.begin(), reference_modes.end(), mode);
        if (it == reference_modes.end())
            return FFDAS_ERROR_INVALID_ARGUMENT;
        perm.push_back(static_cast<int64_t>(it - reference_modes.begin()));
    }

    return FFDAS_SUCCESS;
}


// Enumerate all valid operand memory layouts by trying every ordering of the
// batch, reduce, and free modes, and every placement of batch-first vs
// batch-last and reduce-first vs reduce-last. Each combination produces one
// batched_gemm_layout with the permutation that rearranges the original
// operand modes into that layout.
static ffdas_error_t make_layouts(
    const std::vector<int> &a_modes,
    const std::vector<int> &b_modes,
    const std::vector<int> &a_free,
    const std::vector<int> &b_free,
    const std::vector<int> &batch,
    const std::vector<int> &reduce,
    std::vector<batched_gemm_layout> &layouts
) {
    layouts.clear();

    std::vector<int> batch_ord = batch;
    std::sort(batch_ord.begin(), batch_ord.end());
    do {
        std::vector<int> reduce_ord = reduce;
        std::sort(reduce_ord.begin(), reduce_ord.end());
        do {
            std::vector<int> a_free_ord = a_free;
            std::sort(a_free_ord.begin(), a_free_ord.end());
            do {
                for (bool batch_first_a : {false, true}) {
                    for (bool reduce_last_a : {false, true}) {
                        std::vector<int> a_layout;
                        std::vector<int64_t> a_perm;

                        if (batch_first_a)
                            a_layout.insert(a_layout.end(), batch_ord.begin(), batch_ord.end());
                        if (!reduce_last_a) {
                            a_layout.insert(a_layout.end(), reduce_ord.begin(), reduce_ord.end());
                            if (!batch_first_a)
                                a_layout.insert(a_layout.end(), batch_ord.begin(), batch_ord.end());
                        }
                        a_layout.insert(a_layout.end(), a_free_ord.begin(), a_free_ord.end());
                        if (reduce_last_a)
                            a_layout.insert(a_layout.end(), reduce_ord.begin(), reduce_ord.end());

                        FFDAS_CHECK(modes_to_permutation(a_modes, a_layout, a_perm));

                        std::vector<int> b_free_ord = b_free;
                        std::sort(b_free_ord.begin(), b_free_ord.end());
                        do {
                            for (bool batch_first_b : {false, true}) {
                                for (bool reduce_last_b : {false, true}) {
                                    std::vector<int> b_layout;
                                    std::vector<int64_t> b_perm;

                                    if (batch_first_b)
                                        b_layout.insert(b_layout.end(), batch_ord.begin(), batch_ord.end());
                                    if (!reduce_last_b) {
                                        b_layout.insert(b_layout.end(), reduce_ord.begin(), reduce_ord.end());
                                        if (!batch_first_b)
                                            b_layout.insert(b_layout.end(), batch_ord.begin(), batch_ord.end());
                                    }
                                    b_layout.insert(b_layout.end(), b_free_ord.begin(), b_free_ord.end());
                                    if (reduce_last_b)
                                        b_layout.insert(b_layout.end(), reduce_ord.begin(), reduce_ord.end());

                                    FFDAS_CHECK(modes_to_permutation(b_modes, b_layout, b_perm));

                                    layouts.push_back({
                                        batch_first_a, reduce_last_a,
                                        batch_first_b, reduce_last_b,
                                        a_perm, std::move(b_perm),
                                        a_free_ord, b_free_ord,
                                        batch_ord, reduce_ord,
                                    });
                                }
                            }
                        } while (std::next_permutation(b_free_ord.begin(), b_free_ord.end()));
                    }
                }
            } while (std::next_permutation(a_free_ord.begin(), a_free_ord.end()));
        } while (std::next_permutation(reduce_ord.begin(), reduce_ord.end()));
    } while (std::next_permutation(batch_ord.begin(), batch_ord.end()));
    return FFDAS_SUCCESS;
}


// Compute the permutation that maps the GEMM output's mode order into the
// user's requested out mode order.
static ffdas_error_t make_output_layout(
    const std::vector<int> &out_modes,
    const std::vector<int> &a_free,
    const std::vector<int> &b_free,
    const std::vector<int> &batch,
    bool swap_ab,
    bool batch_first,
    std::vector<int64_t> &perm
) {
    std::vector<int> target_order;

    if (batch_first)
        target_order.insert(target_order.end(), batch.begin(), batch.end());

    // determine order in which a and b's dimensions get added to the C-major 
    // output dims
    const auto &first  = swap_ab ? a_free : b_free;
    const auto &second = swap_ab ? b_free : a_free;
    target_order.insert(target_order.end(), first.begin(), first.end());

    if (!batch_first)
        target_order.insert(target_order.end(), batch.begin(), batch.end());

    target_order.insert(target_order.end(), second.begin(), second.end());

    return modes_to_permutation(target_order, out_modes, perm);
}


// Heuristic cost of b permutation: dimensions that move far from their original
// position will cause more non-contiguous memory access. Larger dimensions are
// weighted more heavily because they span more of the address space.
static int64_t estimate_permutation_cost(const std::vector<int64_t> &dims, const std::vector<int64_t> &perm) {
    int64_t total_size = 1;
    int64_t ndim = static_cast<int64_t>(dims.size());

    for (int64_t d : dims) total_size *= d;

    int64_t cost = 0;
    for (int64_t i = 0; i < ndim; i++) {
        int64_t old_i = perm[i];
        if (old_i < 0 || old_i >= ndim) continue;
        if (dims[old_i] <= 0) continue;

        cost += std::abs(i - old_i) * (total_size / dims[old_i]);
    }

    return cost;
}


// Product of dimension sizes for b set of modes.
static int64_t combined_mode_size(
    const std::vector<int>& modes,
    const std::vector<int64_t>& dims,
    const std::unordered_map<int, int>& mode_map
) {
    int64_t prod = 1;
    for (int m : modes) {
        auto it = mode_map.find(m);
        if (it != mode_map.end())
            prod *= dims[it->second];
    }
    return prod;
}


static bool is_identity_permutation(const std::vector<int64_t> &perm) {
    for (size_t i = 0; i < perm.size(); i++) {
        if (perm[i] != static_cast<int64_t>(i))
            return false;
    }
    return true;
}


// Search all candidate layouts for the one with the lowest total permutation
// cost across b, b, and out. Writes the best GEMM parameters and permutations
// into the plan, then allocates device workspace for any non-identity
// permutations.
static ffdas_error_t find_contraction(
    ffdas_context &handle,
    const ffdas_tensor_desc &a_desc,
    const std::vector<int>& a_modes,
    const ffdas_tensor_desc &b_desc,
    const std::vector<int>& b_modes,
    const ffdas_tensor_desc &out_desc,
    const std::vector<int>& out_modes,
    ffdas_contraction_plan& plan
) {
    std::unordered_map<int, int> a_map, b_map, out_map;
    for (size_t i = 0; i < a_modes.size(); ++i) a_map[a_modes[i]] = static_cast<int>(i);
    for (size_t i = 0; i < b_modes.size(); ++i) b_map[b_modes[i]] = static_cast<int>(i);
    for (size_t i = 0; i < out_modes.size(); ++i) out_map[out_modes[i]] = static_cast<int>(i);

    std::unordered_map<int, int> flags;
    FFDAS_CHECK(classify_modes(a_modes, b_modes, out_modes, flags));

    const std::vector<int64_t> &a_dims = a_desc.dims;
    const std::vector<int64_t> &b_dims = b_desc.dims;
    const std::vector<int64_t> &out_dims = out_desc.dims;

    FFDAS_CHECK(validate_mode_dims(a_dims, a_map, b_dims, b_map, out_dims, out_map, flags));

    std::vector<int> a_broadcast, a_batch, a_reduce, b_broadcast, b_batch, b_reduce;
    FFDAS_CHECK(group_modes(a_modes, flags, a_broadcast, a_batch, a_reduce));
    FFDAS_CHECK(group_modes(b_modes, flags, b_broadcast, b_batch, b_reduce));

    int64_t a_free_size = combined_mode_size(a_broadcast, a_dims, a_map);
    int64_t b_free_size = combined_mode_size(b_broadcast, b_dims, b_map);
    int64_t reduce_size = combined_mode_size(a_reduce, a_dims, a_map);
    int64_t batch_size  = combined_mode_size(a_batch, a_dims, a_map);

    std::vector<batched_gemm_layout> layouts;
    FFDAS_CHECK(make_layouts(a_modes, b_modes, a_broadcast, b_broadcast, a_batch, a_reduce, layouts));

    plan.a_desc = a_desc;
    plan.b_desc = b_desc;
    plan.out_desc = out_desc;

    int64_t best_cost = INT64_MAX;

    for (const batched_gemm_layout &lay : layouts) {
        int64_t a_cost = estimate_permutation_cost(a_dims, lay.a_perm);
        int64_t b_cost = estimate_permutation_cost(b_dims, lay.b_perm);

        for (bool swap_ab : {false, true}) {
            for (bool batch_first : {false, true}) {
                std::vector<int64_t> out_perm;
                FFDAS_CHECK(make_output_layout(out_modes, lay.a_free, lay.b_free, lay.batch, swap_ab, batch_first, out_perm));

                std::vector<int64_t> out_perm_inv = invert_permutation(out_perm);

                std::vector<int64_t> perm_dims(out_perm.size());
                for (size_t i = 0; i < out_perm.size(); i++)
                    perm_dims[i] = out_dims[out_perm_inv[i]];

                int64_t out_cost = estimate_permutation_cost(perm_dims, out_perm);
                int64_t total_cost = a_cost + b_cost + out_cost;

                if (total_cost >= best_cost)
                    continue;

                int64_t m = a_free_size;
                int64_t n = b_free_size;
                int64_t k = reduce_size;
                int64_t bc = batch_size;

                int64_t lda = lay.reduce_last_a
                    ? (lay.batch_first_a ? k : k * bc)
                    : (lay.batch_first_a ? m : m * bc);
                int64_t ldb = lay.reduce_last_b
                    ? (lay.batch_first_b ? k : k * bc)
                    : (lay.batch_first_b ? n : n * bc);

                int64_t strideA = lay.batch_first_a ? (k * m) : (lay.reduce_last_a ? k : m);
                int64_t strideB = lay.batch_first_b ? (k * n) : (lay.reduce_last_b ? k : n);

                if (swap_ab) {
                    std::swap(m, n);
                    std::swap(lda, ldb);
                    std::swap(strideA, strideB);
                }

                int64_t ldc = batch_first ? m : (m * bc);
                int64_t strideC = batch_first ? (m * n) : m;

                plan.swap_ab = swap_ab;
                plan.pa = lay.a_perm;
                plan.pb = lay.b_perm;
                plan.pout = std::move(out_perm);
                plan.transa = swap_ab ? lay.reduce_last_b : lay.reduce_last_a;
                plan.transb = swap_ab ? !lay.reduce_last_a : !lay.reduce_last_b;
                plan.m = m;
                plan.n = n;
                plan.k = k;
                plan.lda = lda;
                plan.ldb = ldb;
                plan.ldc = ldc;
                plan.strideA = strideA;
                plan.strideB = strideB;
                plan.strideC = strideC;
                plan.batchCount = bc;

                best_cost = total_cost;
            }
        }
    }

    if (best_cost == INT64_MAX)
        return FFDAS_ERROR_INTERNAL;

    plan.do_aperm = !is_identity_permutation(plan.pa);
    plan.do_bperm = !is_identity_permutation(plan.pb);
    plan.do_outperm = !is_identity_permutation(plan.pout);

    if (plan.do_aperm)
        FFDAS_CHECK(plan.a_work.alloc(handle, a_desc.nbytes()));
    if (plan.do_bperm)
        FFDAS_CHECK(plan.b_work.alloc(handle, b_desc.nbytes()));
    if (plan.do_outperm)
        FFDAS_CHECK(plan.out_work.alloc(handle, out_desc.nbytes()));

    return FFDAS_SUCCESS;
}


template<typename T>
ffdas_error_t contraction_impl(
    ffdas_context &handle,
    ffdas_contraction_plan &plan,
    const T *a,
    const T *b,
    T *out
) {
    if (plan.do_aperm) {
        ffdas_tensor_desc ap_desc = plan.a_desc.permute(plan.pa);
        FFDAS_CHECK(contiguous_copy_impl(handle, ap_desc, a, static_cast<T*>(plan.a_work.get())));
    }

    if (plan.do_bperm) {
        ffdas_tensor_desc bp_desc = plan.b_desc.permute(plan.pb);
        FFDAS_CHECK(contiguous_copy_impl(handle, bp_desc, b, static_cast<T*>(plan.b_work.get())));
    }

    cudaDataType_t dtype = builtin_traits<T>::cuda_datatype;

    T alpha = builtin_traits<T>::one();
    T beta = builtin_traits<T>::zero();

    // swap_ab controls which tensor gets the A vs B role in cuBLAS.
    // The workspace pointer is used if the operand was permuted, otherwise
    // the original user pointer is passed directly.
    const T *Aptr = static_cast<const T*>(plan.swap_ab
        ? (plan.do_bperm ? plan.b_work.get() : b)
        : (plan.do_aperm ? plan.a_work.get() : a)
    );
    const T *Bptr = static_cast<const T*>(plan.swap_ab
        ? (plan.do_aperm ? plan.a_work.get() : a)
        : (plan.do_bperm ? plan.b_work.get() : b)
    );
    T *Cptr = static_cast<T*>(plan.do_outperm ? plan.out_work.get() : out);

    cublasComputeType_t compute_type;
    if (plan.a_desc.dtype == FFDAS_R_16F || plan.a_desc.dtype == FFDAS_C_16F)
        compute_type = CUBLAS_COMPUTE_16F;
    else if (plan.a_desc.dtype == FFDAS_R_64F || plan.a_desc.dtype == FFDAS_C_64F)
        compute_type = CUBLAS_COMPUTE_64F;
    else
        compute_type = CUBLAS_COMPUTE_32F;

    cublasOperation_t transa = plan.transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = plan.transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t cublas_err;

    if (plan.batchCount == 1 && plan.m == 1 && plan.n == 1) {
        cublas_err = cublasDotEx_64(
            handle.cublas_h, plan.k,
            Aptr, dtype, 1, Bptr, dtype, 1,
            Cptr, dtype, dtype);
    } else if (plan.batchCount == 1) {
        cublas_err = cublasGemmEx_64(
            handle.cublas_h, transa, transb,
            plan.m, plan.n, plan.k,
            &alpha,
            Aptr, dtype, plan.lda,
            Bptr, dtype, plan.ldb,
            &beta,
            Cptr, dtype, plan.ldc,
            compute_type, CUBLAS_GEMM_DEFAULT);
    } else {
        cublas_err = cublasGemmStridedBatchedEx_64(
            handle.cublas_h, transa, transb,
            plan.m, plan.n, plan.k,
            &alpha, Aptr, dtype, plan.lda, plan.strideA,
            Bptr, dtype, plan.ldb, plan.strideB,
            &beta, Cptr, dtype, plan.ldc, plan.strideC,
            plan.batchCount,
            compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    if (cublas_err != CUBLAS_STATUS_SUCCESS) return FFDAS_ERROR_CUBLAS;

    // The GEMM wrote its result in the intermediate layout. Reconstruct that
    // layout's descriptor, permute it into the user's out layout, and copy.
    if (plan.do_outperm) {
        std::vector<int64_t> out_perm_inv = invert_permutation(plan.pout);

        std::vector<int64_t> out_dims(out_perm_inv.size());
        for (size_t i = 0; i < out_perm_inv.size(); i++)
            out_dims[i] = plan.out_desc.dims[out_perm_inv[i]];

        std::vector<int64_t> out_strides = make_contiguous_strides(out_dims);
        ffdas_tensor_desc out_desc(std::move(out_dims), std::move(out_strides), plan.out_desc.dtype);
        ffdas_tensor_desc outp_desc = out_desc.permute(plan.pout);

        FFDAS_CHECK(contiguous_copy_impl(handle, outp_desc, static_cast<const T*>(Cptr), out));
    }

    return FFDAS_SUCCESS;
}


}  // namespace ffdas::detail


ffdas_error_t ffdas_create_contraction(
    ffdas_handle_t handle,
    ffdas_contraction_plan_t *plan,
    ffdas_tensor_desc_t a_desc,
    const int *a_modes,
    ffdas_tensor_desc_t b_desc,
    const int *b_modes,
    ffdas_tensor_desc_t out_desc,
    const int *out_modes
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(plan);
    CHECK_NULL_PTR(a_desc);
    CHECK_NULL_PTR(b_desc);
    CHECK_NULL_PTR(out_desc);
    CHECK_NULL_PTR(a_modes);
    CHECK_NULL_PTR(b_modes);
    CHECK_NULL_PTR(out_modes);

    FFDAS_CHECK(handle->check_device());

    ffdas::detail::nvtx_range nvtx(*handle, "einsum_plan");

    const ffdas_tensor_desc &a_tensor = *a_desc;
    const ffdas_tensor_desc &b_tensor = *b_desc;
    const ffdas_tensor_desc &out_tensor = *out_desc;

    if (!out_tensor.is_contiguous())
        return FFDAS_ERROR_INVALID_DIMS;

    if (a_tensor.dtype != b_tensor.dtype || out_tensor.dtype != a_tensor.dtype)
        return FFDAS_ERROR_UNSUPPORTED_TYPE;

    for (int64_t dim : a_tensor.dims) {
        if (dim <= 0) return FFDAS_ERROR_INVALID_DIMS;
    }
    for (int64_t dim : b_tensor.dims) {
        if (dim <= 0) return FFDAS_ERROR_INVALID_DIMS;
    }
    for (int64_t dim : out_tensor.dims) {
        if (dim <= 0) return FFDAS_ERROR_INVALID_DIMS;
    }

    std::vector<int> am(a_modes, a_modes + a_tensor.ndim());
    std::vector<int> bm(b_modes, b_modes + b_tensor.ndim());
    std::vector<int> outm(out_modes, out_modes + out_tensor.ndim());

    try {
        *plan = new ffdas_contraction_plan();
    } catch (const std::bad_alloc&) {
        *plan = nullptr;
        return FFDAS_ERROR_ALLOCATION_FAILED;
    }

    ffdas_error_t err = ffdas::detail::find_contraction(
        *handle, a_tensor, am, b_tensor, bm, out_tensor, outm, **plan);

    if (err != FFDAS_SUCCESS) {
        delete *plan;
        *plan = nullptr;
        return err;
    }

    return FFDAS_SUCCESS;
}


ffdas_error_t ffdas_contraction(
    ffdas_handle_t handle,
    ffdas_contraction_plan_t plan,
    const void *a,
    const void *b,
    void *out
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(plan);
    FFDAS_CHECK(handle->check_device());

    ffdas::detail::nvtx_range nvtx(*handle, "einsum");

    switch (plan->a_desc.dtype) {
    case FFDAS_R_16F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_R_16F>(*handle, *plan, a, b, out);
    case FFDAS_C_16F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_C_16F>(*handle, *plan, a, b, out);
    case FFDAS_R_32F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_R_32F>(*handle, *plan, a, b, out);
    case FFDAS_C_32F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_C_32F>(*handle, *plan, a, b, out);
    case FFDAS_R_64F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_R_64F>(*handle, *plan, a, b, out);
    case FFDAS_C_64F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_C_64F>(*handle, *plan, a, b, out);
    default:
        break;
    }

    return FFDAS_ERROR_UNSUPPORTED_TYPE;
}


ffdas_error_t ffdas_destroy_contraction(
    ffdas_handle_t handle,
    ffdas_contraction_plan_t plan
) {
    if (!plan)
        return FFDAS_SUCCESS;

    delete plan;
    return FFDAS_SUCCESS;
}
