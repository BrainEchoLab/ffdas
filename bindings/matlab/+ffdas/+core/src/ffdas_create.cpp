#include "_ffdas_mex_common.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mxInitGPU();
    ffdas_handle_t h;
    check(ffdas_create(&h));
    plhs[0] = pack_handle(h);
}
