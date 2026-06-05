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
            "expected 6 arguments: ffdas_contraction(handle, a, a_modes, b, b_modes, out_modes)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_contraction:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> a(prhs[1]);
    ndarray::ndarray<int32_t, ndarray::access::read_only, ndarray::device::host, ndarray::require_vector> x_modes_arr(prhs[2]);
    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> b(prhs[3]);
    ndarray::ndarray<int32_t, ndarray::access::read_only, ndarray::device::host, ndarray::require_vector> a_modes_arr(prhs[4]);
    ndarray::ndarray<int32_t, ndarray::access::read_only, ndarray::device::host, ndarray::require_vector> y_modes_arr(prhs[5]);

    std::vector<int> a_modes(x_modes_arr.data(), x_modes_arr.data() + x_modes_arr.numel());
    std::vector<int> b_modes(a_modes_arr.data(), a_modes_arr.data() + a_modes_arr.numel());
    std::vector<int> out_modes(y_modes_arr.data(), y_modes_arr.data() + y_modes_arr.numel());

    std::reverse(a_modes.begin(), a_modes.end());
    std::reverse(b_modes.begin(), b_modes.end());
    std::reverse(out_modes.begin(), out_modes.end());

    if ((int)a_modes.size() != a.ndim_val())
        mexErrMsgIdAndTxt("ffdas_contraction:error",
            "a_modes length (%zu) must match a dimensions (%d)",
            a_modes.size(), a.ndim_val());
    if ((int)b_modes.size() != b.ndim_val())
        mexErrMsgIdAndTxt("ffdas_contraction:error",
            "b_modes length (%zu) must match b dimensions (%d)",
            b_modes.size(), b.ndim_val());
    if (a.class_id != b.class_id || a.complexity != b.complexity)
        mexErrMsgIdAndTxt("ffdas_contraction:error",
            "input datatypes must match exactly");

    std::map<int, int> mode_to_dim;

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

    for (int i = 0; i < b.ndim_val(); i++) {
        int mode = b_modes[i];
        int dim = b.shape(i);
        auto it = mode_to_dim.find(mode);
        if (it != mode_to_dim.end() && it->second != dim)
            mexErrMsgIdAndTxt("ffdas_contraction:error",
                "mode %d has conflicting dimensions: %d vs %d",
                mode, it->second, dim);
        mode_to_dim[mode] = dim;
    }

    std::vector<int64_t> out_dims;
    for (int mode : out_modes) {
        auto it = mode_to_dim.find(mode);
        if (it == mode_to_dim.end())
            mexErrMsgIdAndTxt("ffdas_contraction:error",
                "output mode %d not found in input modes", mode);
        out_dims.push_back(it->second);
    }

    ndarray::ndarray out = ndarray::make_ndarray(out_dims, a.class_id, a.complexity);

    ScopedTensorDesc a_desc(a);
    ScopedTensorDesc b_desc(b);
    ScopedTensorDesc out_desc(out);

    ffdas_contraction_plan_t plan;
    check(ffdas_create_contraction(
        handle, &plan,
        a_desc.desc, a_modes.data(),
        b_desc.desc, b_modes.data(),
        out_desc.desc, out_modes.data()));

    ffdas_error_t err = ffdas_contraction(
        handle, plan, a.data(), b.data(), out.data());

    ffdas_destroy_contraction(handle, plan);

    if (err)
        mexErrMsgIdAndTxt("ffdas_contraction:error",
            "ffdas_contraction returned error %d: %s",
            err, ffdas_error_string(err));

    plhs[0] = out.release();
}
