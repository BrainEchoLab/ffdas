#include <math.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdint>

#include "mex.h"

#include "ffdas.h"
#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 5)
        mexErrMsgIdAndTxt("ffdas_greens:nargs", "expected 5 input arguments: ffdas_greens(handle, xpos, wavenums, x, ypos)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_greens:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu> xpos(prhs[1]);
    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu, ndarray::require_vector> wavenums(prhs[2]);
    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu, ndarray::ndim<3>> x(prhs[3]);
    ndarray::ndarray<ndarray::access::read_only, float, ndarray::device::gpu> ypos(prhs[4]);

    int64_t batch_size = x.shape(0);
    int64_t channels = x.shape(1);
    int64_t samples = x.shape(2);

    int64_t ny = 1;
    std::vector<int64_t> out_shape(2 + ypos.ndim_val()-1);
    out_shape[0] = batch_size;
    out_shape[1 + ypos.ndim_val()-1] = samples;

    for (int i = 0; i < ypos.ndim_val()-1; i++) {
        ny *= ypos.shape(i);
        out_shape[i+1] = ypos.shape(i);
    }

    ndarray::ndarray y = ndarray::make_ndarray({batch_size, ny, samples}, x.class_id, x.complexity);

    ScopedTensorDesc x_desc(x);
    ScopedTensorDesc y_desc(y);

    check(ffdas_greens_sum(
        handle, 
        reinterpret_cast<const float3*>(xpos.data()), 
        wavenums.data(), 
        x_desc.desc, 
        x.data(), 
        reinterpret_cast<const float3*>(ypos.data()), 
        y_desc.desc, 
        y.data()
    ));

    y.reshape(out_shape);

    plhs[0] = y.release();
}
