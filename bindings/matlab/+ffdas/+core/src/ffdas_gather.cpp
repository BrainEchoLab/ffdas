#include <math.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>

#include "mex.h"

#include "ffdas.h"
#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4)
        mexErrMsgIdAndTxt("ffdas_gather:nargs", "expected at least 4 input arguments: ffdas_gather(handle, x, indices, axis[, permutation, dtype, interleaved_complex])");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_gather:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> x(prhs[1]);
    ndarray::ndarray<int32_t, ndarray::access::read_only, ndarray::device::gpu, ndarray::require_vector> indices(prhs[2]);
    int axis = static_cast<int>(mxGetScalar(prhs[3]));

    if (axis < 1 || axis > x.ndim_val())
        mexErrMsgIdAndTxt("ffdas_gather:axis", "axis %d out of bounds for array with %d dimensions", axis + 1, x.ndim_val());
    if (indices.numel() == 0)
        mexErrMsgIdAndTxt("ffdas_gather:indices", "indices must have nonzero length");

    axis = x.ndim_val() - axis;  // column-major to row-major

    // interleaved_complex: read a real x as (real, imag) pairs along MATLAB's first dimension,
    // which is the contiguous last row-major axis: [..., 2n] real becomes [..., n] complex, no copy
    if (nrhs > 6 && mxIsLogicalScalarTrue(prhs[6])) {
        int64_t last = x.ndim_val() - 1;
        if (x.complexity != mxREAL)
            mexErrMsgIdAndTxt("ffdas_gather:interleaved_complex", "interleaved_complex expects a real array");
        if (x.class_id != mxINT16_CLASS && x.class_id != mxSINGLE_CLASS)
            mexErrMsgIdAndTxt("ffdas_gather:interleaved_complex", "interleaved_complex supports int16 and single arrays");
        if (x.strides[last] != 1 || x.dims[last] % 2 != 0)
            mexErrMsgIdAndTxt("ffdas_gather:interleaved_complex", "interleaved_complex needs an even first dimension of (real, imag) pairs");
        x.dims[last] /= 2;
        for (int64_t i = 0; i < last; i++)
            x.strides[i] /= 2;  // MATLAB arrays are contiguous: every outer stride is a multiple of the even first dimension
        x.complexity = mxCOMPLEX;
    }

    if (nrhs > 4) {
        // int32, as gather.m passes it (an int64 array here refused every permutation)
        ndarray::ndarray<int32_t, ndarray::device::host, ndarray::require_vector> p(prhs[4]);

        if (p.numel() > 0) {
            std::vector<int64_t> order(p.data(), p.data() + p.numel());

            // to 0-based and col-maj to row-maj axis index
            for (int i = 0; i < order.size(); i++) {
                order[i] = x.ndim_val() - order[i];
            }
            std::reverse(order.begin(), order.end());
            x.permute(order);

            // map gather axis through permutation
            axis = std::find(order.begin(), order.end(), axis) - order.begin();
        }
    }

    std::string dtype = (nrhs > 5) ? array_to_string(prhs[5]) : std::string();
    mxClassID out_cls = dtype.empty() ? x.class_id : class_id_from_name(dtype.c_str());
    mxComplexity out_cplx = x.complexity;

    // allocate output
    std::vector<int64_t> out_dims(x.ndim_val());
    for (int i = 0; i < x.ndim_val(); i++) {
        out_dims[i] = (i == axis) ? indices.numel() : x.shape(i);
    }

    auto out = ndarray::make_ndarray(out_dims, out_cls, out_cplx);

    ScopedTensorDesc x_desc(x);
    ScopedTensorDesc out_desc(out);

    check(ffdas_gather(
        handle,
        x_desc.desc, 
        x.data(),
        out_desc.desc, 
        out.data(),
        axis,
        indices.numel(),
        indices.data()
    ));

    plhs[0] = out.release();
}
