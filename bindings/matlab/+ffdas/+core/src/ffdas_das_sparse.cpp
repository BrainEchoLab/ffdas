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
    if (nrhs != 12)
        mexErrMsgIdAndTxt("ffdas_das_sparse:nargs",
            "expected 11 arguments: ffdas_das_sparse(handle, x, srcpos, dstpos, offsets, weights, sparse_indices, srcdir, wavenum, algorithm, use_fp16, channels_trailing)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_das_sparse:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> x(prhs[1]);
    ndarray::ndarray<ndarray::access::read_only, float,  ndarray::device::gpu> srcpos(prhs[2]);
    ndarray::ndarray<ndarray::access::read_only, float,  ndarray::device::gpu> dstpos(prhs[3]);
    ndarray::ndarray<ndarray::access::read_only, float,  ndarray::device::gpu> offsets(prhs[4]);
    ndarray::ndarray<ndarray::access::read_only, float,  ndarray::device::gpu> weights(prhs[5]);
    ndarray::ndarray<ndarray::access::read_only, int32_t, ndarray::device::gpu> sparse_indices(prhs[6]);

    // untyped: the default [] arrives as an empty double gpuArray
    ndarray::ndarray<ndarray::access::read_only> srcdir(prhs[7]);

    float wavenum = static_cast<float>(mxGetScalar(prhs[8]));
    int algorithm = static_cast<int>(mxGetScalar(prhs[9]));
    bool use_fp16 = static_cast<bool>(mxGetScalar(prhs[10]));
    bool channels_trailing = static_cast<bool>(mxGetScalar(prhs[11]));

    // x as ([batch,] channels, sequence, samples)
    das_input_layout(x, channels_trailing, "ffdas_das_sparse:error");

    bool have_batch = (x.ndim_val() == 4);
    int64_t channels = x.shape(x.ndim_val()-3);

    if (srcpos.ndim_val() != 2 || srcpos.shape(1) != 3 || srcpos.shape(0) != channels)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "srcpos must have shape (3, channels) for the %d channels of x", (int)channels);

    if (srcdir.numel() > 0) {
        if (srcdir.class_id != mxSINGLE_CLASS || srcdir.complexity != mxREAL || !srcdir.is_on_gpu() || srcdir.ndim_val() != 2 || srcdir.shape(1) != 4 || srcdir.shape(0) != channels)
            mexErrMsgIdAndTxt("ffdas_das_sparse:error",
                "srcdir must be a single array of shape (4, channels)");
    }

    // dstpos must have dimensions (..., 3)
    int64_t dstnd = dstpos.ndim_val();
    if (dstnd < 2 || dstpos.shape(dstnd - 1) != 3)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "dstpos must have shape (3, ...)");

    // offsets, weights and sparse_indices: (..., n) in MATLAB order, ... the spatial dimensions of
    // dstpos and n the number of events compounded per target. Compared without size-1 dimensions
    // (MATLAB drops a trailing n of 1), so n is the element count over the number of targets.
    if (offsets.dims != weights.dims || offsets.dims != sparse_indices.dims)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "offsets, weights and sparse_indices must have the same shape");
    std::vector<int64_t> spatial(dstpos.dims.begin(), dstpos.dims.end() - 1);
    int64_t targets = 1;
    for (int64_t d : spatial)
        targets *= d;
    int64_t sparse_count = (targets > 0) ? (int64_t)sparse_indices.numel() / targets : 0;
    std::vector<int64_t> expected = spatial;
    expected.insert(expected.begin(), sparse_count);
    if (targets > 0 && (sparse_count < 1 || squeezed(offsets.dims) != squeezed(expected)))
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "offsets, weights and sparse_indices must have shape (..., n): the spatial dimensions of dstpos, then the n events per target");

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
        
    ffdas_error_t err = ffdas_das_sparse(
        handle,
        srcpos.data(),
        srcdir.numel() > 0 ? static_cast<const float*>(srcdir.data()) : nullptr,
        wavenum,
        x_desc.desc, 
        x.data(),
        dstpos.data(),
        static_cast<const float*>(offsets.data()),
        static_cast<const float*>(weights.data()),
        (int)sparse_count,
        sparse_indices.data(),
        beta,
        out_desc.desc, 
        out.data(),
        compute_type,
        static_cast<ffdas_alg_t>(algorithm)
    );

    free(beta);

    if (err)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "ffdas_das_sparse returned error %d: %s",
            err, ffdas_error_string(err));

    plhs[0] = out.release();
}
