#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 0)
        mexErrMsgIdAndTxt("ffdas_event_create:nargs", "expected 0 arguments: ffdas_event_create()");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("ffdas_event_create:nargs", "expected 1 output argument");

    uintptr_t event;
    check(ffdas_event_create(&event));
    mxArray *mx = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t *>(mxGetData(mx)) = event;
    plhs[0] = mx;
}
