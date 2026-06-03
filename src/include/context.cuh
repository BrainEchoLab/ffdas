#pragma once

#include <iostream>
#include <memory>
#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cub/cub.cuh>

#ifdef FFDAS_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "ffdas.h"
#include "error_checking.h"


struct ffdas_context {
    cublasHandle_t cublas_h;
    cusolverDnHandle_t cusolver_h;
    int arch_code;
    int device_id;
    cudaStream_t stream;
    std::unique_ptr<cub::CachingDeviceAllocator> allocator;
#ifdef FFDAS_USE_NVTX
    nvtxDomainHandle_t nvtx_domain;
#endif
    int bounds_checking;

    ffdas_error_t device_alloc(size_t bytes, void **ptr) const;
    ffdas_error_t device_free(void *ptr) const;
    bool is_alive() const;
    ffdas_error_t check_device() const;

    ffdas_context()
      : cublas_h(nullptr),
        cusolver_h(nullptr),
        arch_code(0),
        device_id(0),
        stream(0),
        allocator(nullptr),
#ifdef FFDAS_USE_NVTX
        nvtx_domain(nullptr),
#endif
        bounds_checking(1)
    {}
};


namespace ffdas::detail {

class nvtx_range {
public:
#ifdef FFDAS_USE_NVTX
    nvtx_range(const ffdas_context& ctx, const char* name)
      : domain_(ctx.nvtx_domain) {
        nvtxEventAttributes_t attrib = {0};
        attrib.version = NVTX_VERSION;
        attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attrib.colorType = NVTX_COLOR_ARGB;
        attrib.color = 0xFFFFFFFF;
        attrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attrib.message.ascii = name;
        nvtxDomainRangePushEx(domain_, &attrib);
    }
    ~nvtx_range() { nvtxDomainRangePop(domain_); }
private:
    nvtxDomainHandle_t domain_;
#else
    nvtx_range(const ffdas_context&, const char*) {}
#endif
    nvtx_range(const nvtx_range&) = delete;
    nvtx_range& operator=(const nvtx_range&) = delete;
};


template<typename T>
class device_ptr {
    ffdas_context* ctx_ = nullptr;
    T* ptr_ = nullptr;

public:
    device_ptr() = default;
    explicit device_ptr(ffdas_context &ctx) : ctx_(&ctx) {}
    ~device_ptr() { if (ptr_ && ctx_) ctx_->device_free(ptr_); }

    ffdas_error_t alloc(ffdas_context &ctx, size_t bytes) {
        ctx_ = &ctx;
        return ctx_->device_alloc(bytes, reinterpret_cast<void**>(&ptr_));
    }

    ffdas_error_t alloc(size_t bytes) {
        assert(ctx_);
        return ctx_->device_alloc(bytes, reinterpret_cast<void**>(&ptr_));
    }

    T* get() const { return ptr_; }

    T* release() {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }

    device_ptr(device_ptr&& o) noexcept : ctx_(o.ctx_), ptr_(o.ptr_) { o.ptr_ = nullptr; }
    device_ptr& operator=(device_ptr<T>&& o) noexcept {
        if (this != &o) {
            if (ptr_ && ctx_) ctx_->device_free(ptr_);
            ctx_ = o.ctx_;
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
        }
        return *this;
    }
    device_ptr(const device_ptr&) = delete;
    device_ptr& operator=(const device_ptr&) = delete;
};

}  // namespace ffdas::detail


ffdas_error_t ffdas_set_stream(
    ffdas_handle_t handle,
    uintptr_t stream
);

ffdas_error_t ffdas_get_stream(
    ffdas_handle_t handle,
    uintptr_t *stream
);
