#pragma once

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <vector>
#include <memory>
#include <complex>
#include <algorithm>
#include <cstdint>
#include <type_traits>


namespace ndarray {

namespace device {
    struct gpu {};
    struct host {};
}

namespace access {
    struct read_only {};
    struct writable {};
}

struct require_vector {};

template<int64_t N> struct ndim {};

namespace detail {

template<typename T> struct is_device : std::false_type {};
template<> struct is_device<device::gpu> : std::true_type {};
template<> struct is_device<device::host> : std::true_type {};

template<typename T> struct is_access : std::false_type {};
template<> struct is_access<access::read_only> : std::true_type {};
template<> struct is_access<access::writable> : std::true_type {};

template<typename T> struct is_ndim : std::false_type {};
template<int64_t N> struct is_ndim<ndim<N>> : std::true_type {};

template<typename T> struct is_require_vector : std::false_type {};
template<> struct is_require_vector<require_vector> : std::true_type {};

template<typename T>
struct is_dtype : std::bool_constant<
    !is_device<T>::value && !is_access<T>::value && !is_ndim<T>::value && !is_require_vector<T>::value> {};

// find first type in pack matching Pred, or Default
template<template<typename> class Pred, typename Default, typename... Ts>
struct find_or { using type = Default; };

template<template<typename> class Pred, typename Default, typename T, typename... Rest>
struct find_or<Pred, Default, T, Rest...> {
    using type = std::conditional_t<
        Pred<T>::value, T, typename find_or<Pred, Default, Rest...>::type>;
};

template<template<typename> class Pred, typename Default, typename... Ts>
using find_or_t = typename find_or<Pred, Default, Ts...>::type;

// extract ndim value from tag
template<typename T> struct ndim_value { static constexpr int64_t value = -1; };
template<int64_t N> struct ndim_value<ndim<N>> { static constexpr int64_t value = N; };

// stdlib type to mex class and complexity
template<typename T> struct mx_traits;
template<> struct mx_traits<float> { static constexpr auto cls = mxSINGLE_CLASS; static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<double> { static constexpr auto cls = mxDOUBLE_CLASS; static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<int64_t> { static constexpr auto cls = mxINT64_CLASS;  static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<uint64_t> { static constexpr auto cls = mxUINT64_CLASS; static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<int32_t> { static constexpr auto cls = mxINT32_CLASS;  static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<uint32_t> { static constexpr auto cls = mxUINT32_CLASS; static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<int16_t> { static constexpr auto cls = mxINT16_CLASS;  static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<uint16_t> { static constexpr auto cls = mxUINT16_CLASS; static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<int8_t> { static constexpr auto cls = mxINT8_CLASS;   static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<uint8_t> { static constexpr auto cls = mxUINT8_CLASS;  static constexpr auto cplx = mxREAL; };
template<> struct mx_traits<std::complex<int16_t>> { static constexpr auto cls = mxINT16_CLASS; static constexpr auto cplx = mxCOMPLEX; };
template<> struct mx_traits<std::complex<float>> { static constexpr auto cls = mxSINGLE_CLASS; static constexpr auto cplx = mxCOMPLEX; };
template<> struct mx_traits<std::complex<double>> { static constexpr auto cls = mxDOUBLE_CLASS; static constexpr auto cplx = mxCOMPLEX; };

} // namespace detail


template<typename... Args>
struct ndarray {
    using device_t = detail::find_or_t<detail::is_device, device::gpu, Args...>;
    using access_t = detail::find_or_t<detail::is_access, access::read_only, Args...>;
    using dtype_t = detail::find_or_t<detail::is_dtype, void, Args...>;

    static constexpr bool on_gpu = std::is_same_v<device_t, device::gpu>;
    static constexpr bool on_host = std::is_same_v<device_t, device::host>;
    static constexpr bool is_ro = std::is_same_v<access_t, access::read_only>;
    static constexpr int expected_ndim =
        detail::ndim_value<detail::find_or_t<detail::is_ndim, ndim<-1>, Args...>>::value;
    static constexpr bool expect_vector =
        !std::is_same_v<detail::find_or_t<detail::is_require_vector, void, Args...>, void>;

    using base_ptr_t = std::conditional_t<std::is_void_v<dtype_t>, void, dtype_t>;
    using ptr_t = std::conditional_t<is_ro, const base_ptr_t *, base_ptr_t *>;

    mxClassID class_id;
    mxComplexity complexity;
    std::vector<int64_t> dims;  // row-major order
    std::vector<int64_t> strides;  // row-major strides, in elements

    // construct from mxArray
    explicit ndarray(const mxArray *mx) : mx_(mx) {
        if constexpr (on_gpu) {
            if (!mxIsGPUArray(mx))
                mexErrMsgIdAndTxt("ndarray:ndarray", "expected a gpuArray");
            if constexpr (is_ro)
                gpu_.reset(const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(mx)));
            else
                gpu_.reset(mxGPUCopyFromMxArray(mx));
        } else if constexpr (on_host) {
            if (mxIsGPUArray(mx))
                mexErrMsgIdAndTxt("ndarray:ndarray", "expected a host array, got gpuArray");
        }
        init();
    }

    // construct from mxGPUArray (takes ownership, GPU only)
    explicit ndarray(mxGPUArray *arr) : gpu_(arr) {
        static_assert(on_gpu, "mxGPUArray* constructor requires device::gpu");
        init();
    }

    ~ndarray() = default;

    ndarray(ndarray &&o) noexcept
        : gpu_(std::move(o.gpu_)), mx_(o.mx_),
          class_id(o.class_id), complexity(o.complexity),
          dims(std::move(o.dims)), strides(std::move(o.strides)) {
        o.mx_ = nullptr;
    }

    ndarray &operator=(ndarray &&o) noexcept {
        if (this != &o) {
            gpu_ = std::move(o.gpu_);
            mx_ = o.mx_;
            o.mx_ = nullptr;
            class_id = o.class_id;
            complexity = o.complexity;
            dims = std::move(o.dims);
            strides = std::move(o.strides);
        }
        return *this;
    }

    ndarray(const ndarray &) = delete;
    ndarray &operator=(const ndarray &) = delete;

    ptr_t data() const {
        const void *p;
        if constexpr (on_gpu) {
            if constexpr (is_ro)
                p = mxGPUGetDataReadOnly(gpu_.get());
            else
                p = mxGPUGetData(gpu_.get());
        } else {
            p = mxGetData(const_cast<mxArray *>(mx_));
        }
        return const_cast<ptr_t>(static_cast<const base_ptr_t *>(p));
    }

    int64_t ndim_val() const { return dims.size(); }

    int64_t shape(int i) const { return dims[i]; }

    int64_t stride(int i) const { return strides[i]; }

    bool is_on_gpu() const {
        return (bool)gpu_;
    }

    bool is_contiguous() const {
        if (strides.empty())
            return true;

        int prev = strides[0];
        for (int i = 1; i < strides.size(); i++) {
            if (strides[i] > prev) return false;
            prev = strides[i];
        }

        return true;
    }

    size_t numel() const {
        size_t n = 1;
        for (int64_t d : dims) n *= d;
        return n;
    }

    void reshape(const std::vector<int64_t>& new_dims) {
        if (!is_contiguous())
            mexErrMsgIdAndTxt("ndarray:ndarray",
                "cannot reshape an ndarray with non-contiguous memory");

        int64_t old_count = 1, new_count = 1;

        for (int64_t d : dims) 
            old_count *= d;
        for (int64_t d : new_dims)
            new_count *= d;

        if (old_count != new_count)
            mexErrMsgIdAndTxt("ndarray:reshape", "total elements must match");

        // compute strides
        int64_t new_ndim = new_dims.size();
        std::vector<int64_t> new_strides(new_ndim);

        if (!new_dims.empty()) {
            new_strides[new_ndim-1] = 1;

            for (int i = new_ndim-2; i >= 0; i--) {
                new_strides[i] = new_strides[i+1] * new_dims[i+1];
            }
        }

        strides = new_strides;
        dims = new_dims;
    }

    void permute(const std::vector<int64_t>& order) {
        size_t nd = dims.size();
        if (order.size() != nd) {
            mexErrMsgIdAndTxt(
                "ndarray:ndarray",
                "permutation size must equal number of dimensions"
            );
        }

        std::vector<bool> seen(nd);
        for (int64_t p : order) {
            if (p < 0 || size_t(p) >= nd || seen[p]) {
                mexErrMsgIdAndTxt(
                    "ndarray:ndarray",
                    "invalid or duplicate axis in permutation: %d", 
                    p
                );
            }
            seen[p] = true;
        }

        std::vector<int64_t> new_dims(nd), new_strides(nd);
        for (size_t i = 0; i < nd; ++i) {
            new_dims[i] = dims[order[i]];
            new_strides[i] = strides[order[i]];
        }

        strides = new_strides;
        dims = new_dims;
    }

    // return ownership to matlab as a gpuArray mxArray*
    mxArray *release() {
        static_assert(on_gpu, "release() requires device::gpu");

        if (!is_contiguous())
            mexErrMsgIdAndTxt("ndarray:ndarray",
                "cannot release an ndarray with non-contiguous memory");

        std::vector<mwSize> mw_dims(dims.begin(), dims.end());
        std::reverse(mw_dims.begin(), mw_dims.end());
        mxGPUSetDimensions(gpu_.get(), mw_dims.data(), mw_dims.size());

        mxArray *out = mxGPUCreateMxArrayOnGPU(gpu_.get());
        gpu_.reset();
        return out;
    }

private:
    struct GpuDel {
        void operator()(mxGPUArray *p) const { if (p) mxGPUDestroyGPUArray(p); }
    };
    std::unique_ptr<mxGPUArray, GpuDel> gpu_;
    const mxArray *mx_ = nullptr;

    void init() {
        size_t nd;
        const mwSize *mx_dims;

        if constexpr (on_gpu) {
            class_id = mxGPUGetClassID(gpu_.get());
            complexity = mxGPUGetComplexity(gpu_.get());
            nd = static_cast<size_t>(mxGPUGetNumberOfDimensions(gpu_.get()));
            mx_dims = mxGPUGetDimensions(gpu_.get());
        } else {
            class_id = mxGetClassID(mx_);
            complexity = mxIsComplex(mx_) ? mxCOMPLEX : mxREAL;
            nd = static_cast<size_t>(mxGetNumberOfDimensions(mx_));
            mx_dims = mxGetDimensions(mx_);
        }

        check_dtype();
        check_ndim(nd);

        // col-major to row-major
        dims.resize(nd);
        strides.resize(nd);

        for (int i = 0; i < nd; i++)
            dims[i] = static_cast<int64_t>(mx_dims[nd - 1 - i]);
        strides[nd - 1] = 1;
        for (int i = nd - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * dims[i + 1];

        if constexpr (expect_vector) {
            int nonunit = 0;
            int64_t len = 1;
            for (int64_t d : dims) {
                if (d > 1) { nonunit++; len = d; }
            }
            if (nonunit > 1)
                mexErrMsgIdAndTxt("ndarray:ndarray",
                    "expected a vector, got array with %d non-unit dimensions", nonunit);
            dims = {(nonunit == 0) ? 0 : len};
            strides = {1};
        }
    }

    void check_dtype() {
        if constexpr (!std::is_void_v<dtype_t>) {
            using traits = detail::mx_traits<dtype_t>;
            if (class_id != traits::cls || complexity != traits::cplx)
                mexErrMsgIdAndTxt("ndarray:ndarray",
                    "dtype mismatch: expected class %d/%d, got %d/%d",
                    traits::cls, traits::cplx, class_id, complexity);
        }
    }

    void check_ndim(size_t nd) {
        if constexpr (expected_ndim >= 0) {
            if (nd != expected_ndim)
                mexErrMsgIdAndTxt("ndarray:ndarray",
                    "ndim mismatch: expected %d, got %d", expected_ndim, nd);
        }
    }
};


template<typename... Args>
inline ndarray<access::writable, Args...>
make_ndarray(const std::vector<int64_t> &c_dims, mxClassID cls, mxComplexity cplx) {
    using device_t = detail::find_or_t<detail::is_device, device::gpu, Args...>;

    size_t nd = c_dims.size();
    std::vector<mwSize> mw_dims(nd);
    for (int i = 0; i < nd; i++)
        mw_dims[i] = static_cast<mwSize>(c_dims[nd - 1 - i]);

    if constexpr (std::is_same_v<device_t, device::gpu>) {
        mxGPUArray *arr = mxGPUCreateGPUArray(
            nd, mw_dims.data(), cls, cplx, MX_GPU_DO_NOT_INITIALIZE);
        return ndarray<access::writable, Args...>(arr);
    }

    mxArray *arr = mxCreateUninitNumericArray(
        nd, mw_dims.data(), cls, cplx
    );
    return ndarray<access::writable, Args...>(arr);
}

template<typename T, typename... Args>
inline ndarray<access::writable, T, Args...>
make_ndarray(const std::vector<int64_t> &c_dims) {
    mxClassID cls = detail::mx_traits<T>::cls;
    mxComplexity cplx = detail::mx_traits<T>::cplx;
    return make_ndarray<T, Args...>(c_dims, cls, cplx);
}

template<typename... Args>
inline ndarray<access::writable, Args...> make_ndarray_like(const ndarray<Args...> &src) {
    return make_ndarray<Args...>(src.dims, src.class_id, src.complexity);
}

} // namespace ndarray
