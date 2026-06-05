#include <math.h>
#include <vector>
#include <cstdio>
#include <cstdint>

#include "mex.h"

#include "ffdas.h"
#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 6)
        mexErrMsgIdAndTxt("ffdas_interpolate:nargs",
            "expected 6 input arguments: ffdas_interpolate(handle, gridpos, x, querypos, mode, fill_value)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_interpolate:nargs", "expected 1 out argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu> gridpos(prhs[1]);
    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> x(prhs[2]);
    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu> querypos(prhs[3]);

    std::string interp_mode = array_to_string(prhs[4]);
    ndarray::ndarray<ndarray::access::read_only, ndarray::device::host> fill_value(prhs[5]);

    if (gridpos.ndim_val() != 4 || gridpos.shape(3) != 3)
        mexErrMsgIdAndTxt("ffdas_interpolate:error",
            "gridpos must have shape (3, nx, ny, nz)");
    if (fill_value.numel() != 1) {
        mexErrMsgIdAndTxt("ffdas_interpolate:error",
            "fill_value should contain exactly one element");
    }

    int64_t nz = gridpos.shape(0);
    int64_t ny = gridpos.shape(1);
    int64_t nx = gridpos.shape(2);

    if (nz < 2 || ny < 2 || nx < 2)
        mexErrMsgIdAndTxt("ffdas_interpolate:error",
            "grid size must be at least (2, 2, 2), got (%d, %d, %d)", nx, ny, nz);

    int64_t nv = x.ndim_val();
    if (nv < 3)
        mexErrMsgIdAndTxt("ffdas_interpolate:error",
            "x must have at least 3 dimensions");
    if (x.shape(nv - 1) != nx || x.shape(nv - 2) != ny || x.shape(nv - 3) != nz)
        mexErrMsgIdAndTxt("ffdas_interpolate:error",
            "x spatial dims must match grid shape (%d, %d, %d)", nx, ny, nz);

    int64_t nq = querypos.ndim_val();
    if (nq < 2 || querypos.shape(nq - 1) != 3)
        mexErrMsgIdAndTxt("ffdas_interpolate:error",
            "querypos must have shape (3, ...)");

    ffdas_interp_mode_t mode_enum;
    if (interp_mode == "nearest")
        mode_enum = FFDAS_INTERP_NEAREST;
    else if (interp_mode == "linear")
        mode_enum = FFDAS_INTERP_LINEAR;
    else
        mexErrMsgIdAndTxt("ffdas_interpolate:error",
            "unknown interpolation mode: %s", interp_mode.c_str());

    int64_t num_querypos = 1;
    for (int i = 0; i < nq-1; i++)
        num_querypos *= querypos.shape(i);

    std::vector<int64_t> out_dims;
    for (int i = 0; i < nv - 3; i++)
        out_dims.push_back(x.shape(i));
    for (int i = 0; i < nq - 1; i++)
        out_dims.push_back(querypos.shape(i));

    ndarray::ndarray result = ndarray::make_ndarray(out_dims, x.class_id, x.complexity);

    ScopedTensorDesc x_desc(x);

    ffdas_interpolation_plan_t plan;
    check(ffdas_create_interpolation_plan(
        handle, &plan, nx, ny, nz,
        gridpos.data(),
        mode_enum));

    ffdas_error_t err = ffdas_interpolation(
        handle, plan,
        num_querypos,
        querypos.data(),
        x_desc.desc, 
        x.data(),
        result.data(),
        fill_value.data());

    ffdas_destroy_interpolation_plan(handle, plan);

    if (err)
        mexErrMsgIdAndTxt("ffdas_interpolate:error",
            "ffdas_interpolation returned error %d: %s",
            err, ffdas_error_string(err));

    plhs[0] = result.release();
}
