#pragma once

#include "math/core.cuh"
#include "math/vector.cuh"
#include "math/bitops.cuh"
#include "tensor.cuh"
#include "ffdas.h"
#include "context.cuh"
#include "type_utils.h"
#include "error_checking.h"
#include "contiguous_copy_impl.cuh"


namespace ffdas::detail {

struct das_problem_params {
    int samples;
    int seqlen; 
    int channels;
    int batch_size;
    int ndst;
    int channel_stride;
    int seq_stride;
    int sample_stride;
    int batch_stride;
    int outstride;
    bool have_batch;
    int sparse_count = 0;
};


// contiguous C-order layout, ignoring dimensions of size 1 (their stride is never used); an empty
// tensor is trivially dense
static bool is_dense(const ffdas_tensor_desc &desc) {
    if (desc.numel() == 0)
        return true;
    int64_t expected = 1;
    for (int i = (int)desc.dims.size() - 1; i >= 0; i--) {
        if (desc.dims[i] == 1)
            continue;
        if (desc.strides[i] != expected)
            return false;
        expected *= desc.dims[i];
    }
    return true;
}

static ffdas_error_t get_das_problem_params(
    const ffdas_tensor_desc &x_desc,
    const ffdas_tensor_desc &out_desc,
    das_problem_params &params
) {
    int ndim = x_desc.dims.size();

    if (ndim != 3 && ndim != 4) 
        return FFDAS_ERROR_INVALID_DIMS;
    if (out_desc.dims.size() < 1) 
        return FFDAS_ERROR_INVALID_DIMS;

    params.have_batch = (ndim == 4);
    int ofs = params.have_batch ? 1 : 0;

    params.samples = (int)x_desc.dims[ofs+2];
    params.seqlen = (int)x_desc.dims[ofs+1];
    params.channels = (int)x_desc.dims[ofs+0];
    params.batch_size = params.have_batch ? (int)x_desc.dims[0] : 1;

    if (params.have_batch && (out_desc.dims.size() < 2 || out_desc.dims[0] != params.batch_size))
        return FFDAS_ERROR_INVALID_DIMS;

    params.ndst = 1;
    for (int i = ofs; i < out_desc.dims.size(); i++) {
        params.ndst *= (int)out_desc.dims[i];
    }

    params.channel_stride = (int)x_desc.strides[ofs+0];
    params.seq_stride = (int)x_desc.strides[ofs+1];
    params.sample_stride = (int)x_desc.strides[ofs+2];
    params.batch_stride = params.have_batch ? (int)x_desc.strides[0] : 0;

    // The kernels index offsets / weights / sparse_indices, all (seq or K, ndst) row-major, as
    // [n + i*outstride], and write out[n + b*outstride]. Both strides are ndst: it used to be out's
    // batch stride, which is 0 without a batch dimension, so every event read event 0's row.
    params.outstride = params.ndst;

    // nothing to compute (see das_is_empty): array libraries may give empty arrays any strides
    if (params.ndst == 0 || params.batch_size == 0)
        return FFDAS_SUCCESS;

    // a sum over no channels, events or samples is not a problem the kernels can run
    if (params.channels == 0 || params.seqlen == 0 || params.samples == 0)
        return FFDAS_ERROR_INVALID_DIMS;

    if (params.sample_stride != 1)
        return FFDAS_ERROR_INVALID_ARGUMENT;

    // out is written as one dense ([batch,] ndst) block
    if (!is_dense(out_desc))
        return FFDAS_ERROR_NON_CONTIGUOUS;

    return FFDAS_SUCCESS;
}

// zero targets or an empty batch: the result is empty and nothing is launched (a zero-size grid
// is a CUDA launch error)
static bool das_is_empty(const das_problem_params &params) {
    return params.ndst == 0 || params.batch_size == 0;
}

}  // ffdas::detail
