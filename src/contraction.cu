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
    const std::vector<int>& x_modes,
    const std::vector<int>& a_modes,
    const std::vector<int>& y_modes,
    std::unordered_map<int, int>& flags
) {
    flags.clear();

    for (int m : x_modes) flags[m] = MODE_BROADCAST;
    for (int m : a_modes) flags[m] = flags.count(m) ? MODE_REDUCE : MODE_BROADCAST;
    for (int m : y_modes) {
        auto it = flags.find(m);
        if (it == flags.end())
            return FFDAS_ERROR_INVALID_ARGUMENT;
        if (it->second == MODE_REDUCE) it->second = MODE_BATCH;
    }

    return FFDAS_SUCCESS;
}


// Validate that dimension sizes agree across operands for each shared mode.
static ffdas_error_t validate_mode_dims(
    const std::vector<int64_t>& x_dims, const std::unordered_map<int, int>& x_map,
    const std::vector<int64_t>& a_dims, const std::unordered_map<int, int>& a_map,
    const std::vector<int64_t>& y_dims, const std::unordered_map<int, int>& y_map,
    const std::unordered_map<int, int>& flags
) {
    for (const auto& [mode, flag] : flags) {
        auto xi = x_map.find(mode);
        auto ai = a_map.find(mode);
        auto yi = y_map.find(mode);

        if (flag == MODE_REDUCE) {
            if (xi == x_map.end() || ai == a_map.end())
                return FFDAS_ERROR_INVALID_ARGUMENT;
            if (x_dims[xi->second] != a_dims[ai->second])
                return FFDAS_ERROR_INVALID_ARGUMENT;
        } else if (flag == MODE_BATCH) {
            if (xi == x_map.end() || ai == a_map.end() || yi == y_map.end())
                return FFDAS_ERROR_INVALID_ARGUMENT;
            int64_t d = x_dims[xi->second];
            if (a_dims[ai->second] != d || y_dims[yi->second] != d)
                return FFDAS_ERROR_INVALID_ARGUMENT;
        } else {
            if (yi == y_map.end())
                return FFDAS_ERROR_INVALID_ARGUMENT;
            if (xi != x_map.end() && x_dims[xi->second] != y_dims[yi->second])
                return FFDAS_ERROR_INVALID_ARGUMENT;
            if (ai != a_map.end() && a_dims[ai->second] != y_dims[yi->second])
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
    bool batch_first_x, reduce_last_x, batch_first_a, reduce_last_a;
    std::vector<int64_t> x_perm, a_perm;
    std::vector<int> x_free, a_free;
    std::vector<int> batch, reduce;
};


// For each mode label in query_order, find its position in reference_modes.
// Output is a permutation (int64_t indices into reference_modes).
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
    const std::vector<int> &x_modes,
    const std::vector<int> &a_modes,
    const std::vector<int> &x_free,
    const std::vector<int> &a_free,
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
            std::vector<int> x_free_ord = x_free;
            std::sort(x_free_ord.begin(), x_free_ord.end());
            do {
                for (bool batch_first_x : {false, true}) {
                    for (bool reduce_last_x : {false, true}) {
                        std::vector<int> x_layout;
                        std::vector<int64_t> x_perm;

                        if (batch_first_x)
                            x_layout.insert(x_layout.end(), batch_ord.begin(), batch_ord.end());
                        if (!reduce_last_x) {
                            x_layout.insert(x_layout.end(), reduce_ord.begin(), reduce_ord.end());
                            if (!batch_first_x)
                                x_layout.insert(x_layout.end(), batch_ord.begin(), batch_ord.end());
                        }
                        x_layout.insert(x_layout.end(), x_free_ord.begin(), x_free_ord.end());
                        if (reduce_last_x)
                            x_layout.insert(x_layout.end(), reduce_ord.begin(), reduce_ord.end());

                        FFDAS_CHECK(modes_to_permutation(x_modes, x_layout, x_perm));

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

                                    layouts.push_back({
                                        batch_first_x, reduce_last_x,
                                        batch_first_a, reduce_last_a,
                                        x_perm, std::move(a_perm),
                                        x_free_ord, a_free_ord,
                                        batch_ord, reduce_ord,
                                    });
                                }
                            }
                        } while (std::next_permutation(a_free_ord.begin(), a_free_ord.end()));
                    }
                }
            } while (std::next_permutation(x_free_ord.begin(), x_free_ord.end()));
        } while (std::next_permutation(reduce_ord.begin(), reduce_ord.end()));
    } while (std::next_permutation(batch_ord.begin(), batch_ord.end()));
    return FFDAS_SUCCESS;
}


// Compute the permutation that maps the GEMM output's mode order into the
// user's requested y mode order.
static ffdas_error_t make_output_layout(
    const std::vector<int> &y_modes,
    const std::vector<int> &a_free,
    const std::vector<int> &x_free,
    const std::vector<int> &batch,
    bool swap_ax,
    bool batch_first,
    std::vector<int64_t> &perm
) {
    std::vector<int> target_order;

    if (batch_first)
        target_order.insert(target_order.end(), batch.begin(), batch.end());

    const auto &first  = swap_ax ? a_free : x_free;
    const auto &second = swap_ax ? x_free : a_free;
    target_order.insert(target_order.end(), first.begin(), first.end());

    if (!batch_first)
        target_order.insert(target_order.end(), batch.begin(), batch.end());

    target_order.insert(target_order.end(), second.begin(), second.end());

    return modes_to_permutation(target_order, y_modes, perm);
}


// Heuristic cost of a permutation: dimensions that move far from their original
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


// Product of dimension sizes for a set of modes.
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
// cost across x, a, and y. Writes the best GEMM parameters and permutations
// into the plan, then allocates device workspace for any non-identity
// permutations.
static ffdas_error_t find_contraction(
    ffdas_context &handle,
    const ffdas_tensor_desc &x_desc,
    const std::vector<int>& x_modes,
    const ffdas_tensor_desc &a_desc,
    const std::vector<int>& a_modes,
    const ffdas_tensor_desc &y_desc,
    const std::vector<int>& y_modes,
    ffdas_contraction_plan& plan
) {
    std::unordered_map<int, int> x_map, a_map, y_map;
    for (size_t i = 0; i < x_modes.size(); ++i) x_map[x_modes[i]] = static_cast<int>(i);
    for (size_t i = 0; i < a_modes.size(); ++i) a_map[a_modes[i]] = static_cast<int>(i);
    for (size_t i = 0; i < y_modes.size(); ++i) y_map[y_modes[i]] = static_cast<int>(i);

    std::unordered_map<int, int> flags;
    FFDAS_CHECK(classify_modes(x_modes, a_modes, y_modes, flags));

    const std::vector<int64_t> &x_dims = x_desc.dims;
    const std::vector<int64_t> &a_dims = a_desc.dims;
    const std::vector<int64_t> &y_dims = y_desc.dims;

    FFDAS_CHECK(validate_mode_dims(x_dims, x_map, a_dims, a_map, y_dims, y_map, flags));

    std::vector<int> x_broadcast, x_batch, x_reduce, a_broadcast, a_batch, a_reduce;
    FFDAS_CHECK(group_modes(x_modes, flags, x_broadcast, x_batch, x_reduce));
    FFDAS_CHECK(group_modes(a_modes, flags, a_broadcast, a_batch, a_reduce));

    int64_t x_free_size = combined_mode_size(x_broadcast, x_dims, x_map);
    int64_t a_free_size = combined_mode_size(a_broadcast, a_dims, a_map);
    int64_t reduce_size = combined_mode_size(x_reduce, x_dims, x_map);
    int64_t batch_size  = combined_mode_size(x_batch, x_dims, x_map);

    std::vector<batched_gemm_layout> layouts;
    FFDAS_CHECK(make_layouts(x_modes, a_modes, x_broadcast, a_broadcast, x_batch, x_reduce, layouts));

    plan.x_desc = x_desc;
    plan.a_desc = a_desc;
    plan.y_desc = y_desc;

    int64_t best_cost = INT64_MAX;

    for (const batched_gemm_layout &lay : layouts) {
        int64_t x_cost = estimate_permutation_cost(x_dims, lay.x_perm);
        int64_t a_cost = estimate_permutation_cost(a_dims, lay.a_perm);

        for (bool swap_ax : {false, true}) {
            for (bool batch_first : {false, true}) {
                std::vector<int64_t> y_perm;
                FFDAS_CHECK(make_output_layout(y_modes, lay.a_free, lay.x_free, lay.batch, swap_ax, batch_first, y_perm));

                std::vector<int64_t> y_perm_inv = invert_permutation(y_perm);

                std::vector<int64_t> out_dims(y_perm.size());
                for (size_t i = 0; i < y_perm.size(); i++)
                    out_dims[i] = y_dims[y_perm_inv[i]];

                int64_t y_cost = estimate_permutation_cost(out_dims, y_perm);
                int64_t total_cost = x_cost + a_cost + y_cost;

                if (total_cost >= best_cost)
                    continue;

                int64_t m = a_free_size;
                int64_t n = x_free_size;
                int64_t k = reduce_size;
                int64_t bc = batch_size;

                int64_t lda = lay.reduce_last_a
                    ? (lay.batch_first_a ? k : k * bc)
                    : (lay.batch_first_a ? m : m * bc);
                int64_t ldb = lay.reduce_last_x
                    ? (lay.batch_first_x ? k : k * bc)
                    : (lay.batch_first_x ? n : n * bc);

                int64_t strideA = lay.batch_first_a ? (k * m) : (lay.reduce_last_a ? k : m);
                int64_t strideB = lay.batch_first_x ? (k * n) : (lay.reduce_last_x ? k : n);

                if (swap_ax) {
                    std::swap(m, n);
                    std::swap(lda, ldb);
                    std::swap(strideA, strideB);
                }

                int64_t ldc = batch_first ? m : (m * bc);
                int64_t strideC = batch_first ? (m * n) : m;

                plan.swap_ax = swap_ax;
                plan.px = lay.x_perm;
                plan.pa = lay.a_perm;
                plan.py = std::move(y_perm);
                plan.transa = swap_ax ? lay.reduce_last_x : lay.reduce_last_a;
                plan.transb = swap_ax ? !lay.reduce_last_a : !lay.reduce_last_x;
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

    plan.do_xperm = !is_identity_permutation(plan.px);
    plan.do_aperm = !is_identity_permutation(plan.pa);
    plan.do_yperm = !is_identity_permutation(plan.py);

    if (plan.do_xperm)
        FFDAS_CHECK(plan.x_work.alloc(handle, x_desc.nbytes()));
    if (plan.do_aperm)
        FFDAS_CHECK(plan.a_work.alloc(handle, a_desc.nbytes()));
    if (plan.do_yperm)
        FFDAS_CHECK(plan.y_work.alloc(handle, y_desc.nbytes()));

    return FFDAS_SUCCESS;
}


template<typename T>
ffdas_error_t contraction_impl(
    ffdas_context &handle,
    ffdas_contraction_plan &plan,
    const T *x,
    const T *a,
    T *y
) {
    if (plan.do_xperm) {
        ffdas_tensor_desc xp_desc = plan.x_desc.permute(plan.px);
        FFDAS_CHECK(contiguous_copy_impl(handle, xp_desc, x, static_cast<T*>(plan.x_work.get())));
    }

    if (plan.do_aperm) {
        ffdas_tensor_desc ap_desc = plan.a_desc.permute(plan.pa);
        FFDAS_CHECK(contiguous_copy_impl(handle, ap_desc, a, static_cast<T*>(plan.a_work.get())));
    }

    cudaDataType_t dtype = builtin_traits<T>::cuda_datatype;

    T alpha = builtin_traits<T>::one();
    T beta = builtin_traits<T>::zero();

    // swap_ax controls which tensor gets the A vs B role in cuBLAS.
    // The workspace pointer is used if the operand was permuted, otherwise
    // the original user pointer is passed directly.
    const T *Aptr = static_cast<const T*>(plan.swap_ax
        ? (plan.do_xperm ? plan.x_work.get() : x)
        : (plan.do_aperm ? plan.a_work.get() : a));
    const T *Bptr = static_cast<const T*>(plan.swap_ax
        ? (plan.do_aperm ? plan.a_work.get() : a)
        : (plan.do_xperm ? plan.x_work.get() : x));
    T *Cptr = static_cast<T*>(plan.do_yperm ? plan.y_work.get() : y);

    cublasComputeType_t compute_type;
    if (plan.x_desc.dtype == FFDAS_R_16F || plan.x_desc.dtype == FFDAS_C_16F)
        compute_type = CUBLAS_COMPUTE_16F;
    else if (plan.x_desc.dtype == FFDAS_R_64F || plan.x_desc.dtype == FFDAS_C_64F)
        compute_type = CUBLAS_COMPUTE_64F;
    else
        compute_type = CUBLAS_COMPUTE_32F;

    cublasOperation_t transa = plan.transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = plan.transb ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t cublas_err;

    if (plan.batchCount == 1 && plan.m == 1 && plan.n == 1) {
        cublas_err = cublasDotEx_64(
            handle.cublas_h, plan.k,
            Bptr, dtype, 1, Aptr, dtype, 1,
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
    // layout's descriptor, permute it into the user's y layout, and copy.
    if (plan.do_yperm) {
        std::vector<int64_t> y_perm_inv = invert_permutation(plan.py);

        std::vector<int64_t> out_dims(y_perm_inv.size());
        for (size_t i = 0; i < y_perm_inv.size(); i++)
            out_dims[i] = plan.y_desc.dims[y_perm_inv[i]];

        std::vector<int64_t> out_strides = make_contiguous_strides(out_dims);
        ffdas_tensor_desc out_desc(std::move(out_dims), std::move(out_strides), plan.y_desc.dtype);
        ffdas_tensor_desc yp_desc = out_desc.permute(plan.py);

        FFDAS_CHECK(contiguous_copy_impl(handle, yp_desc, static_cast<const T*>(Cptr), y));
    }

    return FFDAS_SUCCESS;
}


}  // namespace ffdas::detail


ffdas_error_t ffdas_create_contraction(
    ffdas_handle_t handle,
    ffdas_contraction_plan_t *plan,
    ffdas_tensor_desc_t x_desc,
    const int *x_modes,
    ffdas_tensor_desc_t a_desc,
    const int *a_modes,
    ffdas_tensor_desc_t y_desc,
    const int *y_modes
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(plan);
    CHECK_NULL_PTR(x_desc);
    CHECK_NULL_PTR(a_desc);
    CHECK_NULL_PTR(y_desc);
    CHECK_NULL_PTR(x_modes);
    CHECK_NULL_PTR(a_modes);
    CHECK_NULL_PTR(y_modes);

    FFDAS_CHECK(handle->check_device());

    ffdas::detail::nvtx_range nvtx(*handle, "einsum_plan");

    const ffdas_tensor_desc &x_tensor = *x_desc;
    const ffdas_tensor_desc &a_tensor = *a_desc;
    const ffdas_tensor_desc &y_tensor = *y_desc;

    if (!y_tensor.is_contiguous())
        return FFDAS_ERROR_INVALID_DIMS;

    if (a_tensor.dtype != x_tensor.dtype || y_tensor.dtype != x_tensor.dtype)
        return FFDAS_ERROR_UNSUPPORTED_TYPE;

    for (int64_t dim : x_tensor.dims) {
        if (dim <= 0) return FFDAS_ERROR_INVALID_DIMS;
    }
    for (int64_t dim : a_tensor.dims) {
        if (dim <= 0) return FFDAS_ERROR_INVALID_DIMS;
    }
    for (int64_t dim : y_tensor.dims) {
        if (dim <= 0) return FFDAS_ERROR_INVALID_DIMS;
    }

    std::vector<int> xm(x_modes, x_modes + x_tensor.ndim());
    std::vector<int> am(a_modes, a_modes + a_tensor.ndim());
    std::vector<int> ym(y_modes, y_modes + y_tensor.ndim());

    try {
        *plan = new ffdas_contraction_plan();
    } catch (const std::bad_alloc&) {
        *plan = nullptr;
        return FFDAS_ERROR_ALLOCATION_FAILED;
    }

    ffdas_error_t err = ffdas::detail::find_contraction(
        *handle, x_tensor, xm, a_tensor, am, y_tensor, ym, **plan);

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
    const void *x,
    const void *a,
    void *y
) {
    CHECK_HANDLE(handle);
    CHECK_NULL_PTR(plan);
    FFDAS_CHECK(handle->check_device());

    ffdas::detail::nvtx_range nvtx(*handle, "einsum");

    switch (plan->x_desc.dtype) {
    case FFDAS_R_16F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_R_16F>(*handle, *plan, x, a, y);
    case FFDAS_C_16F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_C_16F>(*handle, *plan, x, a, y);
    case FFDAS_R_32F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_R_32F>(*handle, *plan, x, a, y);
    case FFDAS_C_32F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_C_32F>(*handle, *plan, x, a, y);
    case FFDAS_R_64F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_R_64F>(*handle, *plan, x, a, y);
    case FFDAS_C_64F:
        return ffdas::detail::ffdas_contraction_dispatch<FFDAS_C_64F>(*handle, *plan, x, a, y);
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
