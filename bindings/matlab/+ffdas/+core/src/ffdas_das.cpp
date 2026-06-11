#include <math.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "mex.h"

#include "ffdas.h"
#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 11)
        mexErrMsgIdAndTxt("ffdas_das:nargs",
            "expected 10 arguments: ffdas_das(handle, x, srcpos, dstpos, offsets, weights, srcdir, wavenum, algorithm, use_fp16, channels_trailing)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_das:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> x(prhs[1]);
    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu> srcpos(prhs[2]);
    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu> dstpos(prhs[3]);
    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu> offsets(prhs[4]);
    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu> weights(prhs[5]);

    ndarray::ndarray<ndarray::access::read_only> srcdir(prhs[6]);

    float wavenum = static_cast<float>(mxGetScalar(prhs[7]));
    int algorithm = static_cast<int>(mxGetScalar(prhs[8]));
    bool use_fp16 = static_cast<bool>(mxGetScalar(prhs[9]));

    // refers to trailing from matlab's perspective (col-major ordering), so 'true' means 
    // leadig in row-major order and would require no permutation
    bool channels_trailing = static_cast<bool>(mxGetScalar(prhs[10]));

    // x must have dimensions ([batch,] channels, sequence, samples)
    if (x.ndim_val() < 3 || x.ndim_val() > 4)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "x must have 3 or 4 dimensions (got %d)", x.ndim_val());

    if (!channels_trailing) {
        // in this case, it is possible that the batch and sequence dims are both 1, 
        // and since matlab removes any trailing dimension, we have to add it back here
        if (x.ndim_val() == 2)
            x.reshape({1, x.shape(0), x.shape(1)});

        if (x.ndim_val() == 3) {
            x.permute({1, 0, 2});
        } else {
            x.permute({0, 2, 1, 3});
        }
    }

    bool have_batch = (x.ndim_val() == 4);
    int64_t sequence = channels_trailing ? x.shape(x.ndim_val()-2) : x.shape(x.ndim_val()-3);

    if (srcpos.ndim_val() != 2 || srcpos.shape(1) != 3)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "srcpos must have shape (3, channels)");

    if (srcdir.numel() > 0) {
        if (srcdir.class_id != mxSINGLE_CLASS || srcdir.complexity != mxREAL || !srcdir.is_on_gpu() || srcdir.ndim_val() != 2 || srcdir.shape(1) != 4)
            mexErrMsgIdAndTxt("ffdas_das:error",
                "srcdir must be a single array of shape (4, channels)");
    }

    // dstpos must have dimensions (..., 3)
    int64_t dstnd = dstpos.ndim_val();
    if (dstnd < 2 || dstpos.shape(dstnd - 1) != 3)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "dstpos must have shape (3, ...)");

    // offset and weight must both have dimensions (..., sequence) matching the leading dims of dstpos
    if (offsets.dims != weights.dims)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "offsets and weights must have the same shape");
    if (sequence == 1 && offsets.ndim_val() != (dstnd-1))  // matlab squeezed the sequence dimension from offsets/weights
        mexErrMsgIdAndTxt("ffdas_das:error",
            "offsets and weights must have the same number of dimensions as dstpos");
    if (sequence > 1 && (offsets.ndim_val()-1) != (dstnd-1))  // check number of spatial dimensions
        mexErrMsgIdAndTxt("ffdas_das:error",
            "offsets and weights must have the same number of dimensions as dstpos");
    if (sequence > 1 && offsets.shape(0) != sequence)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "leading dimension of offsets/weights must match the channel dimension of x");
    for (int i = 0; i < dstnd-1; i++) {
        if (offsets.shape((sequence > 1) ? i+1 : i) != dstpos.shape(i))
            mexErrMsgIdAndTxt("ffdas_das:error",
                "spatial dimensions of offsets/weights must match dstpos");
    }

    // out will have dimensions ([batch,] ...)
    int64_t spatial_ndim = dstnd - 1;
    int64_t out_ndim = spatial_ndim + (have_batch ? 1 : 0);

    std::vector<int64_t> out_dims(out_ndim);
    int o = 0;
    if (have_batch)
        out_dims[o++] = x.shape(0);
    for (int i = 0; i < spatial_ndim; i++)
        out_dims[o++] = dstpos.shape(i);

    // out follows the input data type
    mxClassID out_cls = x.class_id;
    mxComplexity out_cplx = x.complexity;
    auto out = ndarray::make_ndarray(out_dims, out_cls, out_cplx);

    ScopedTensorDesc x_desc(x);
    ScopedTensorDesc out_desc(out);

    ffdas_datatype_t out_dtype = to_ffdas_dtype(out_cls, out_cplx);
    void *beta = calloc(1, ffdas_type_size(out_dtype));

    ffdas_compute_type_t compute_type;
    if (out_cls == mxDOUBLE_CLASS)
        compute_type = FFDAS_COMPUTE_64F;
    else if (use_fp16)
        compute_type = FFDAS_COMPUTE_16F;
    else
        compute_type = FFDAS_COMPUTE_32F;

    ffdas_error_t err = ffdas_das(
        handle,
        srcpos.data(),
        srcdir.numel() > 0 ? static_cast<const float*>(srcdir.data()) : nullptr,
        wavenum,
        x_desc.desc, 
        x.data(),
        dstpos.data(),
        static_cast<const float*>(offsets.data()),
        static_cast<const float*>(weights.data()),
        beta,
        out_desc.desc, 
        out.data(),
        compute_type,
        static_cast<ffdas_alg_t>(algorithm)
    );

    free(beta);

    if (err)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "ffdas_das returned error %d: %s",
            err, ffdas_error_string(err));

    plhs[0] = out.release();
}
