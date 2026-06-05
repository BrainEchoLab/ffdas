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
        mexErrMsgIdAndTxt("ffdas_truncate_rank:nargs", "expected 4 input arguments: ffdas_truncate_rank(handle, x, start, stop)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_truncate_rank:nargs", "expected 1 output argument");

    ffdas_handle_t handle = get_handle(prhs[0]);

    ndarray::ndarray<ndarray::access::read_only, ndarray::device::gpu> x(prhs[1]);
    int64_t start = static_cast<int64_t>(mxGetScalar(prhs[2])) - 1;
    int64_t stop = static_cast<int64_t>(mxGetScalar(prhs[3]));

    if (x.ndim_val() < 2)
        mexErrMsgIdAndTxt("ffdas_truncate_rank:error", "input must have at least two dimensions (got %d)", x.ndim_val());

    int64_t n = x.shape(0);
    int64_t m = 1;
    std::vector<int64_t> orig_shape(x.ndim_val());
    orig_shape[0] = n;

    for (int i = 1; i < x.ndim_val(); i++) {
        m *= x.shape(i);
        orig_shape[i] = x.shape(i);
    }

    if (start < 0 || start >= std::min(m, n))
        mexErrMsgIdAndTxt("ffdas_truncate_rank:error", "start must be > 0 and < min(m, n) (got %d)", start);
    if (stop <= start || stop > std::min(m, n))
        mexErrMsgIdAndTxt("ffdas_truncate_rank:error", "stop must be > start and <= min(m, n) (got %d)", stop);

    x.reshape({n, m});

    ndarray::ndarray out = ndarray::make_ndarray_like(x);

    ScopedTensorDesc x_desc(x);
    ScopedTensorDesc out_desc(out);

    check(ffdas_truncate_rank(
        handle, 
        x_desc.desc, 
        x.data(), 
        start, 
        stop, 
        out_desc.desc, 
        out.data()
    ));

    out.reshape(orig_shape);

    plhs[0] = out.release();
}
