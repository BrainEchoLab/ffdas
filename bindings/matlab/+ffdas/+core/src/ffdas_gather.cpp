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
        mexErrMsgIdAndTxt("ffdas_gather:nargs", "expected at least 4 input arguments: ffdas_gather(handle, x, indices, axis[, permutation, dtype])");
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

    if (nrhs > 4) {
        ndarray::ndarray<int64_t, ndarray::device::host, ndarray::require_vector> p(prhs[4]);

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

    mxClassID out_cls = (nrhs > 5) ? class_id_from_name(array_to_string(prhs[5]).c_str()) : x.class_id;
    mxComplexity out_cplx = x.complexity;

    // allocate output
    std::vector<int64_t> out_dims(x.ndim_val());
    for (int i = 0; i < x.ndim_val(); i++) {
        out_dims[i] = (i == axis) ? indices.numel() : x.shape(i);
    }

    ndarray::ndarray out = ndarray::make_ndarray(out_dims, out_cls, out_cplx);

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
