#pragma once

#include "ffdas_math.cuh"
#include "tensor.cuh"
#include "ffdas_api.h"
#include "context.cuh"
#include "ffdas_types.h"
#include "type_utils.h"
#include "error_checking.h"
#include "contiguous_copy_impl.cuh"


namespace ffdas::detail {

struct das_problem_params {
    int samples;
    int seqlen; 
    int channels;
    int batch_size;
    int ny;
    int channel_stride;
    int seq_stride;
    int sample_stride;
    int batch_stride;
    int ystride;
    bool have_batch;
};


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

    params.ny = 1;
    for (int i = ofs; i < out_desc.dims.size(); i++) {
        params.ny *= (int)out_desc.dims[i];
    }

    params.channel_stride = (int)x_desc.strides[ofs+0];
    params.seq_stride = (int)x_desc.strides[ofs+1];
    params.sample_stride = (int)x_desc.strides[ofs+2];
    params.batch_stride = params.have_batch ? (int)x_desc.strides[0] : 0;
    params.ystride = params.have_batch ? (int)out_desc.strides[0] : 0;

    if (params.sample_stride != 1) 
        return FFDAS_ERROR_INVALID_ARGUMENT;

    return FFDAS_SUCCESS;
}

}  // ffdas::detail
