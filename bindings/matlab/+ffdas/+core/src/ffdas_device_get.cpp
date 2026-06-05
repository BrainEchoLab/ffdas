#include "_ffdas_mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 0)
        mexErrMsgIdAndTxt("ffdas_device_get:nargs", "expected 0 arguments: ffdas_device_get()");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_device_get:nargs", "expected 1 output argument");

    int device;
    check(ffdas_device_get(&device));
    mxArray *mx_device = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *static_cast<int *>(mxGetData(mx_device)) = device;
    plhs[0] = mx_device;
}
