#include "_ffdas_mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 0)
        mexErrMsgIdAndTxt("ffdas_device_count:nargs", "expected 0 arguments: ffdas_device_count()");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_device_count:nargs", "expected 1 output argument");

    int count;
    check(ffdas_device_count(&count));
    mxArray *mx_count = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *static_cast<int *>(mxGetData(mx_count)) = count;
    plhs[0] = mx_count;
}
