#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("ffdas_event_elapsed_time:nargs", "expected 2 arguments: ffdas_event_elapsed_time(handle, event)");
    if (nlhs != 0)
        mexErrMsgIdAndTxt("ffdas_event_elapsed_time:nargs", "expected 0 output arguments");

    ffdas_handle_t h = get_handle(prhs[0]);
    const mxArray *mx = prhs[1];
    if (!mxIsUint64(mx) || mxIsEmpty(mx))
        mexErrMsgIdAndTxt("ffdas:event_record", "invalid event");
    uintptr_t event = static_cast<uintptr_t>(*static_cast<uint64_t *>(mxGetData(mx)));
    check(ffdas_event_record(h, event));
}
