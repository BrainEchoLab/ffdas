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

    // x as ([batch,] channels, sequence, samples)
    das_input_layout(x, channels_trailing, "ffdas_das:error");

    bool have_batch = (x.ndim_val() == 4);
    int64_t channels = x.shape(x.ndim_val()-3);
    int64_t sequence = x.shape(x.ndim_val()-2);

    if (srcpos.ndim_val() != 2 || srcpos.shape(1) != 3 || srcpos.shape(0) != channels)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "srcpos must have shape (3, channels) for the %d channels of x", (int)channels);

    if (srcdir.numel() > 0) {
        if (srcdir.class_id != mxSINGLE_CLASS || srcdir.complexity != mxREAL || !srcdir.is_on_gpu() || srcdir.ndim_val() != 2 || srcdir.shape(1) != 4 || srcdir.shape(0) != channels)
            mexErrMsgIdAndTxt("ffdas_das:error",
                "srcdir must be a single array of shape (4, channels)");
    }

    // dstpos must have dimensions (..., 3)
    int64_t dstnd = dstpos.ndim_val();
    if (dstnd < 2 || dstpos.shape(dstnd - 1) != 3)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "dstpos must have shape (3, ...)");

    // offsets and weights: (..., sequence) in MATLAB order, ... the spatial dimensions of dstpos.
    // Compared without size-1 dimensions, so a (pixels x 1) array for a single event is accepted.
    if (offsets.dims != weights.dims)
        mexErrMsgIdAndTxt("ffdas_das:error",
            "offsets and weights must have the same shape");
    std::vector<int64_t> expected(dstpos.dims.begin(), dstpos.dims.end() - 1);
    expected.insert(expected.begin(), sequence);
    if (squeezed(offsets.dims) != squeezed(expected))
        mexErrMsgIdAndTxt("ffdas_das:error",
            "offsets and weights must have shape (..., %d): the spatial dimensions of dstpos, then one entry per sequence event of x",
            (int)sequence);

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
