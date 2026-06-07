#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cassert>

#include "ffdas.h"
#include "error_checking.h"


static constexpr int FFDAS_MAX_DIMS = 16;


struct ffdas_tensor_desc {
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    ffdas_datatype_t dtype;

    ffdas_tensor_desc() : dtype(FFDAS_R_32F) {}

    ffdas_tensor_desc(std::vector<int64_t> dims, std::vector<int64_t> strides,
                        ffdas_datatype_t dtype)
        : dims(std::move(dims)), strides(std::move(strides)), dtype(dtype) {
        assert(this->dims.size() == this->strides.size());
    }

    ffdas_tensor_desc(std::vector<int64_t> dims, ffdas_datatype_t dtype)
        : dims(std::move(dims)), dtype(dtype) {
        int n = this->dims.size();
        strides.resize(n);
        if (n > 0) {
            strides[n - 1] = 1;
            for (int i = n - 2; i >= 0; i--)
                strides[i] = strides[i + 1] * this->dims[i + 1];
        }
    }

    size_t ndim() const { 
        return dims.size(); 
    }

    size_t numel() const {
        size_t n = 1;
        for (int64_t d : dims) 
            n *= d;
        return n;
    };

    size_t nbytes() const {
        size_t itemsize = ffdas_type_size(dtype);
        size_t size = numel();
        return size * itemsize;
    };

    bool is_contiguous() const {
        size_t nd = dims.size();
        size_t s = 1;
        for (int i = nd-1; i >= 0; i--) {
            if (strides[i] != s)
                return false;
            s *= dims[i];
        }
        return true;
    };

    bool same_dims(const ffdas_tensor_desc& other) const {
        size_t nd = dims.size();
        if (nd != other.dims.size()) 
            return false;
        for (int i = 0; i < nd; i++) {
            if (dims[i] != other.dims[i])
                return false;
        }
        return true;
    };

    ffdas_tensor_desc permute(const std::vector<int64_t> &order) const {
        assert(order.size() == dims.size());
        std::vector<bool> seen(dims.size(), false);
        for (int64_t o : order) {
            assert(o >= 0 && o < (int64_t)dims.size() && !seen[o]);
            seen[o] = true;
        }

        std::vector<int64_t> new_dims(dims.size()), new_strides(dims.size());
        for (size_t i = 0; i < dims.size(); i++) {
            new_dims[i] = dims[order[i]];
            new_strides[i] = strides[order[i]];
        }
        return ffdas_tensor_desc(std::move(new_dims), std::move(new_strides), dtype);
    };
};


namespace ffdas::detail {

static bool can_use_int32_indexing(const ffdas_tensor_desc &desc) {
    int64_t max_offset = 0;
    for (size_t i = 0; i < desc.dims.size(); i++) {
        if (desc.dims[i] > 1)
            max_offset += (desc.dims[i] - 1) * std::abs(desc.strides[i]);
    }
    int64_t n = static_cast<int64_t>(desc.numel());
    return max_offset <= INT32_MAX && n <= INT32_MAX;
}

static std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t> &dims) {
    size_t nd = dims.size();
    assert(nd > 0);

    std::vector<int64_t> strides(nd);

    strides[nd-1] = 1;
    for (int i = nd-2; i >= 0; i--) {
        strides[i] = strides[i+1] * dims[i+1];
    }

    return strides;
}

static std::vector<int64_t> invert_permutation(const std::vector<int64_t> &order) {
    std::vector<int64_t> inverted(order.size());
    std::iota(inverted.begin(), inverted.end(), 0);
    std::sort(inverted.begin(), inverted.end(), [&](int64_t i, int64_t j) {return order[i] < order[j]; });
    return inverted;
}

}  // namespace ffdas::detail
