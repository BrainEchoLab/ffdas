#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("ffdas_event_elapsed_time:nargs", "expected 2 arguments: ffdas_event_elapsed_time(start, stop)");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_event_elapsed_time:nargs", "expected 1 output argument");

    const mxArray *mx_start = prhs[0];
    if (!mxIsUint64(mx_start) || mxIsEmpty(mx_start))
        mexErrMsgIdAndTxt("ffdas:event_elapsed_time", "invalid start event");

    const mxArray *mx_stop = prhs[1];
    if (!mxIsUint64(mx_stop) || mxIsEmpty(mx_stop))
        mexErrMsgIdAndTxt("ffdas:event_synchronize", "invalid stop event");

    uintptr_t start = static_cast<uintptr_t>(*static_cast<uint64_t *>(mxGetData(mx_start)));
    uintptr_t stop = static_cast<uintptr_t>(*static_cast<uint64_t *>(mxGetData(mx_stop)));

    float ms;
    check(ffdas_event_elapsed_time(start, stop, &ms));

    mxArray *mx_ms = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    *static_cast<float *>(mxGetData(mx_ms)) = ms;
    plhs[0] = mx_ms;
}
