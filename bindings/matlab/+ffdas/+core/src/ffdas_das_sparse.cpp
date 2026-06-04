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
            "expected 11 arguments: ffdas_das_sparse(handle, x, xpos, ypos, offsets, weights, sparse_indices, xdir, wavenum, algorithm, use_fp16, channels_trailing)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_das_sparse:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> x(prhs[1]);
    ndarray::ndarray<ndarray::access::read_only, float,  ndarray::device::gpu> xpos(prhs[2]);
    ndarray::ndarray<ndarray::access::read_only, float,  ndarray::device::gpu> ypos(prhs[3]);
    ndarray::ndarray<ndarray::access::read_only, float,  ndarray::device::gpu> offsets(prhs[4]);
    ndarray::ndarray<ndarray::access::read_only, float,  ndarray::device::gpu> weights(prhs[5]);
    ndarray::ndarray<ndarray::access::read_only, int32_t, ndarray::device::gpu> sparse_indices(prhs[6]);

    ndarray::ndarray<ndarray::access::read_only, float> xdir(prhs[7]);

    float wavenum = static_cast<float>(mxGetScalar(prhs[8]));
    int algorithm = static_cast<int>(mxGetScalar(prhs[9]));
    bool use_fp16 = static_cast<bool>(mxGetScalar(prhs[10]));
    bool channels_trailing = static_cast<bool>(mxGetScalar(prhs[11]));

    // x must have dimensions ([batch,] channels, sequence, samples)
    if (x.ndim_val() < 3 || x.ndim_val() > 4)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "x must have 3 or 4 dimensions (got %d)", x.ndim_val());

    if (!channels_trailing) {
        if (x.ndim_val() == 3) {
            x.permute({1, 0, 2});
        } else {
            x.permute({0, 2, 1, 3});
        }
    }

    if (xpos.ndim_val() != 2 || xpos.shape(1) != 3)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "xpos must have shape (3, channels)");

    if (xdir.numel() > 0) {
        if (xdir.class_id != mxSINGLE_CLASS || xdir.complexity != mxREAL || !xdir.is_on_gpu() || xdir.ndim_val() != 2 || xdir.shape(1) != 4)
            mexErrMsgIdAndTxt("ffdas_das:error",
                "xdir must be a single array of shape (4, channels)");
    }

    // ypos must have dimensions (..., 3)
    int64_t ynd = ypos.ndim_val();
    if (ynd < 2 || ypos.shape(ynd - 1) != 3)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "ypos must have shape (3, ...)");

    // offset, weight and sparse_indices must all have dimensions (..., n) matching the leading dims of ypos
    if (offsets.dims != weights.dims || offsets.dims != sparse_indices.dims)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "offsets, weights and sparse_indices must have the same shape");
    if (offsets.ndim_val() != ynd)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "offsets, weights and sparse_indices must have the same number of dimensions as ypos");
    for (int i = 1; i < offsets.ndim_val(); i++) {
        if (offsets.shape(i) != ypos.shape(i - 1))
            mexErrMsgIdAndTxt("ffdas_das_sparse:error",
                "spatial dimensions of offsets, weights and sparse_indices must match ypos");
    }

    // y will have dimensions ([batch,] ...)
    bool have_batch = (x.ndim_val() == 4);
    int64_t ynd_spatial = ynd - 1;
    int64_t out_ndim = ynd_spatial + (have_batch ? 1 : 0);

    std::vector<int64_t> out_dims(out_ndim);
    int o = 0;
    if (have_batch)
        out_dims[o++] = x.shape(0);
    for (int i = 0; i < ynd_spatial; i++)
        out_dims[o++] = ypos.shape(i);

    // y follows the input data type
    mxClassID out_cls = x.class_id;
    mxComplexity out_cplx = x.complexity;
    ndarray::ndarray y = ndarray::make_ndarray(out_dims, out_cls, out_cplx);

    ScopedTensorDesc x_desc(x);
    ScopedTensorDesc y_desc(y);

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
        reinterpret_cast<const float3*>(xpos.data()),
        xdir.numel() > 0 ? reinterpret_cast<const float4*>(xdir.data()) : nullptr,
        wavenum,
        x_desc.desc, 
        x.data(),
        reinterpret_cast<const float3*>(ypos.data()),
        static_cast<const float*>(offsets.data()),
        static_cast<const float*>(weights.data()),
        sparse_indices.shape(0),
        sparse_indices.data(),
        beta,
        y_desc.desc, 
        y.data(),
        compute_type,
        static_cast<ffdas_alg_t>(algorithm)
    );

    free(beta);

    if (err)
        mexErrMsgIdAndTxt("ffdas_das_sparse:error",
            "ffdas_das_sparse returned error %d: %s",
            err, ffdas_error_string(err));

    plhs[0] = y.release();
}
