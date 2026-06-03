#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1)
        mexErrMsgIdAndTxt("ffdas_event_destroy:nargs", "expected 1 arguments: ffdas_event_destroy(event)");
    if (nlhs != 0)
        mexErrMsgIdAndTxt("ffdas_event_destroy:nargs", "expected 0 output arguments");

    const mxArray *mx = prhs[0];
    if (!mxIsUint64(mx) || mxIsEmpty(mx))
        mexErrMsgIdAndTxt("ffdas:event_destroy", "invalid event");
    uintptr_t event = static_cast<uintptr_t>(*static_cast<uint64_t *>(mxGetData(mx)));
    check(ffdas_event_destroy(event));
}
