#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "ffdas.h"
#include "tensor.cuh"


// The receive directivity of the channels, as passed to ffdas_das_ex / ffdas_das_sparse_ex. It
// refers to device arrays it does not own.
struct ffdas_directivity_desc {
    int64_t channels = 0;            // 0 (internal only): take the channel count from x
    int64_t events = 0;              // rows of axes; 0 = one row shared by every event
    const float *axes = nullptr;     // float4 [events or 1][channels]: look direction, cos(cut)
};


namespace ffdas::detail {

// what the kernels get, by value; axes == nullptr: no directivity
struct das_directivity {
    const float4 *axes = nullptr;    // [events or 1][channels]
    int event_stride = 0;            // channels if the look directions differ per event, else 0
};

// Check a descriptor against x (([batch,] channels, seq, samples)) and make the kernels' form.
// No descriptor: no directivity.
static ffdas_error_t make_das_directivity(
    const ffdas_directivity_desc *desc,
    const ffdas_tensor_desc &x_desc,
    das_directivity &dir
) {
    dir = das_directivity{};
    if (desc == nullptr)
        return FFDAS_SUCCESS;
    if (desc->axes == nullptr)
        return FFDAS_ERROR_NULL_POINTER;

    const int ndim = (int)x_desc.dims.size();
    if (ndim != 3 && ndim != 4)
        return FFDAS_ERROR_INVALID_DIMS;
    const int64_t channels = x_desc.dims[ndim - 3];
    const int64_t seqlen = x_desc.dims[ndim - 2];

    if (desc->channels != 0 && desc->channels != channels)
        return FFDAS_ERROR_INVALID_DIMS;
    if (desc->events != 0 && desc->events != seqlen)
        return FFDAS_ERROR_INVALID_DIMS;
    // float4 loads
    if (reinterpret_cast<uintptr_t>(desc->axes) % 16 != 0)
        return FFDAS_ERROR_INVALID_ARGUMENT;
    if (desc->events * channels > INT32_MAX)
        return FFDAS_ERROR_DIMS_TOO_LARGE;

    dir.axes = reinterpret_cast<const float4 *>(desc->axes);
    dir.event_stride = (desc->events != 0) ? (int)channels : 0;
    return FFDAS_SUCCESS;
}

}  // namespace ffdas::detail
