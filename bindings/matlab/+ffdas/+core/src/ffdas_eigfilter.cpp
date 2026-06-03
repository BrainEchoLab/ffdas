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
    if (nrhs != 4)
        mexErrMsgIdAndTxt("ffdas_eigfilter:nargs", "expected 4 input arguments: ffdas_eigfilter(handle, x, k0, k1)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_eigfilter:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> x(prhs[1]);
    int64_t k0 = static_cast<int64_t>(mxGetScalar(prhs[2])) - 1;
    int64_t k1 = static_cast<int64_t>(mxGetScalar(prhs[3]));

    if (x.ndim_val() < 2)
        mexErrMsgIdAndTxt("ffdas_eigfilter:error", "input must have at least two dimensions (got %d)", x.ndim_val());

    int64_t n = x.shape(0);
    int64_t m = 1;
    std::vector<int64_t> orig_shape(x.ndim_val());
    orig_shape[0] = n;

    for (int i = 1; i < x.ndim_val(); i++) {
        m *= x.shape(i);
        orig_shape[i] = x.shape(i);
    }

    if (k0 < 0 || k0 >= std::min(m, n))
        mexErrMsgIdAndTxt("ffdas_eigfilter:error", "k0 must be > 0 and < min(m, n) (got %d)", k0);
    if (k1 <= k0 || k1 > std::min(m, n))
        mexErrMsgIdAndTxt("ffdas_eigfilter:error", "k1 must be > k0 and <= min(m, n) (got %d)", k1);

    x.reshape({n, m});

    ndarray::ndarray y = ndarray::make_ndarray_like(x);

    ScopedTensorDesc x_desc(x);
    ScopedTensorDesc y_desc(y);

    check(ffdas_eigfilter(
        handle, 
        x_desc.desc, 
        x.data(), 
        k0, 
        k1, 
        y_desc.desc, 
        y.data()
    ));

    y.reshape(orig_shape);

    plhs[0] = y.release();
}
