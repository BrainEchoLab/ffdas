#include "_ffdas_mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1)
        mexErrMsgIdAndTxt("ffdas_device_set:nargs", "expected 1 argument: ffdas_device_set(device)");
    if (nlhs != 0)
        mexErrMsgIdAndTxt("ffdas_device_set:nargs", "expected 0 output arguments");

    int device = static_cast<int>(mxGetScalar(prhs[0]));
    check(ffdas_device_set(device));
}
