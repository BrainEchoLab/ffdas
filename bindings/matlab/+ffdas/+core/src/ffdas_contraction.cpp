#include <math.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <map>
#include <cstdint>

#include "mex.h"

#include "ffdas.h"
#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 6)
        mexErrMsgIdAndTxt("ffdas_contraction:nargs",
            "expected 6 arguments: ffdas_contraction(handle, x, x_modes, a, a_modes, y_modes)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_contraction:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> x(prhs[1]);
    ndarray::ndarray<int32_t, ndarray::access::read_only, ndarray::device::host, ndarray::require_vector> x_modes_arr(prhs[2]);
    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> a(prhs[3]);
    ndarray::ndarray<int32_t, ndarray::access::read_only, ndarray::device::host, ndarray::require_vector> a_modes_arr(prhs[4]);
    ndarray::ndarray<int32_t, ndarray::access::read_only, ndarray::device::host, ndarray::require_vector> y_modes_arr(prhs[5]);

    std::vector<int> x_modes(x_modes_arr.data(), x_modes_arr.data() + x_modes_arr.numel());
    std::vector<int> a_modes(a_modes_arr.data(), a_modes_arr.data() + a_modes_arr.numel());
    std::vector<int> y_modes(y_modes_arr.data(), y_modes_arr.data() + y_modes_arr.numel());

    std::reverse(x_modes.begin(), x_modes.end());
    std::reverse(a_modes.begin(), a_modes.end());
    std::reverse(y_modes.begin(), y_modes.end());

    if ((int)x_modes.size() != x.ndim_val())
        mexErrMsgIdAndTxt("ffdas_contraction:error",
            "x_modes length (%zu) must match x dimensions (%d)",
            x_modes.size(), x.ndim_val());
    if ((int)a_modes.size() != a.ndim_val())
        mexErrMsgIdAndTxt("ffdas_contraction:error",
            "a_modes length (%zu) must match a dimensions (%d)",
            a_modes.size(), a.ndim_val());
    if (x.class_id != a.class_id || x.complexity != a.complexity)
        mexErrMsgIdAndTxt("ffdas_contraction:error",
            "input datatypes must match exactly");

    std::map<int, int> mode_to_dim;

    for (int i = 0; i < x.ndim_val(); i++) {
        int mode = x_modes[i];
        int dim = x.shape(i);
        auto it = mode_to_dim.find(mode);
        if (it != mode_to_dim.end() && it->second != dim)
            mexErrMsgIdAndTxt("ffdas_contraction:error",
                "mode %d has conflicting dimensions: %d vs %d",
                mode, it->second, dim);
        mode_to_dim[mode] = dim;
    }

    for (int i = 0; i < a.ndim_val(); i++) {
        int mode = a_modes[i];
        int dim = a.shape(i);
        auto it = mode_to_dim.find(mode);
        if (it != mode_to_dim.end() && it->second != dim)
            mexErrMsgIdAndTxt("ffdas_contraction:error",
                "mode %d has conflicting dimensions: %d vs %d",
                mode, it->second, dim);
        mode_to_dim[mode] = dim;
    }

    std::vector<int64_t> out_dims;
    for (int mode : y_modes) {
        auto it = mode_to_dim.find(mode);
        if (it == mode_to_dim.end())
            mexErrMsgIdAndTxt("ffdas_contraction:error",
                "output mode %d not found in input modes", mode);
        out_dims.push_back(it->second);
    }

    ndarray::ndarray y = ndarray::make_ndarray(out_dims, x.class_id, x.complexity);

    ScopedTensorDesc x_desc(x);
    ScopedTensorDesc a_desc(a);
    ScopedTensorDesc y_desc(y);

    ffdas_contraction_plan_t plan;
    check(ffdas_create_contraction(
        handle, &plan,
        x_desc.desc, x_modes.data(),
        a_desc.desc, a_modes.data(),
        y_desc.desc, y_modes.data()));

    ffdas_error_t err = ffdas_contraction(
        handle, plan, x.data(), a.data(), y.data());

    ffdas_destroy_contraction(handle, plan);

    if (err)
        mexErrMsgIdAndTxt("ffdas_contraction:error",
            "ffdas_contraction returned error %d: %s",
            err, ffdas_error_string(err));

    plhs[0] = y.release();
}
