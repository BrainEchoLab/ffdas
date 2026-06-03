#include "_ffdas_mex_common.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    ffdas_handle_t h = get_handle(prhs[0]);
    check(ffdas_destroy(h));
}
