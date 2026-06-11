#pragma once

#include <vector>
#include <cstring>

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "_mex_ndarray.h"
#include "ffdas.h"


inline void check(ffdas_error_t err) {
    if (err != FFDAS_SUCCESS)
        mexErrMsgIdAndTxt("ffdas:check", "%s", ffdas_error_string(err));
}

inline ffdas_handle_t get_handle(const mxArray *mx) {
    if (!mxIsUint64(mx) || mxIsEmpty(mx))
        mexErrMsgIdAndTxt("ffdas:get_handle", "invalid handle");
    return reinterpret_cast<ffdas_handle_t>(
        *static_cast<uint64_t *>(mxGetData(mx)));
}

inline mxArray *pack_handle(ffdas_handle_t h) {
    mxArray *mx = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t *>(mxGetData(mx)) = reinterpret_cast<uint64_t>(h);
    return mx;
}

inline mxClassID class_id_from_name(const char *name) {
    static const struct { 
        const char *name; mxClassID cls; 
    } map[] = {
        {"double", mxDOUBLE_CLASS},
        {"single", mxSINGLE_CLASS},
        {"int8", mxINT8_CLASS},
        {"uint8", mxUINT8_CLASS},
        {"int16", mxINT16_CLASS},
        {"uint16", mxUINT16_CLASS},
        {"int32", mxINT32_CLASS},
        {"uint32", mxUINT32_CLASS},
        {"int64", mxINT64_CLASS},
        {"uint64", mxUINT64_CLASS},
        {"logical", mxLOGICAL_CLASS},
        {"char", mxCHAR_CLASS},
    };
    for (int i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
        if (strcmp(name, map[i].name) == 0) 
            return map[i].cls;
    }
    return mxUNKNOWN_CLASS;
}

inline ffdas_datatype_t to_ffdas_dtype(mxClassID cls, mxComplexity cplx) {
    bool cx = (cplx == mxCOMPLEX);
    switch (cls) {
        case mxDOUBLE_CLASS: return cx ? FFDAS_C_64F : FFDAS_R_64F;
        case mxSINGLE_CLASS: return cx ? FFDAS_C_32F : FFDAS_R_32F;
        case mxINT32_CLASS: return cx ? FFDAS_C_32I : FFDAS_R_32I;
        case mxINT16_CLASS: return cx ? FFDAS_C_16I : FFDAS_R_16I;
        case mxUINT16_CLASS: return cx ? FFDAS_C_16F : FFDAS_R_16F;
        default:
            mexErrMsgIdAndTxt("ffdas:to_ffdas_dtype", "unsupported MATLAB class");
            return FFDAS_R_32F;
    }
}
 
inline void from_ffdas_dtype(ffdas_datatype_t dt, mxClassID &cls, mxComplexity &cplx) {
    switch (dt) {
        case FFDAS_R_64F: cls = mxDOUBLE_CLASS; cplx = mxREAL;    break;
        case FFDAS_C_64F: cls = mxDOUBLE_CLASS; cplx = mxCOMPLEX; break;
        case FFDAS_R_32F: cls = mxSINGLE_CLASS; cplx = mxREAL;    break;
        case FFDAS_C_32F: cls = mxSINGLE_CLASS; cplx = mxCOMPLEX; break;
        case FFDAS_R_32I: cls = mxINT32_CLASS;  cplx = mxREAL;    break;
        case FFDAS_C_32I: cls = mxINT32_CLASS;  cplx = mxCOMPLEX; break;
        case FFDAS_R_16I: cls = mxINT16_CLASS;  cplx = mxREAL;    break;
        case FFDAS_C_16I: cls = mxINT16_CLASS;  cplx = mxCOMPLEX; break;
        case FFDAS_R_16F: cls = mxUINT16_CLASS; cplx = mxREAL;    break;
        case FFDAS_C_16F: cls = mxUINT16_CLASS; cplx = mxCOMPLEX; break;
        default:
            mexErrMsgIdAndTxt("ffdas:from_ffdas_dtype", "unsupported ffdas datatype %d", dt);
    }
}

std::string array_to_string(const mxArray* x) {
    if (!mxIsChar(x))
        mexErrMsgIdAndTxt(
            "ffdas:array_to_string",
            "input must be a char array"
        );

    char* cstr = mxArrayToString(x);
    if (!cstr)
        mexErrMsgIdAndTxt(
            "ffdas:array_to_string",
            "could not convert mxArray to C string"
        );

    std::string out(cstr);
    mxFree(cstr);
    return out;
}

struct ScopedTensorDesc {
    ffdas_tensor_desc_t desc = nullptr;

    template<typename... Args>
    explicit ScopedTensorDesc(const ndarray::ndarray<Args...> &a) {
        check(ffdas_create_tensor_desc(
            &desc, a.ndim_val(), a.dims.data(), a.strides.data(),
            to_ffdas_dtype(a.class_id, a.complexity)));
    }
    ~ScopedTensorDesc() { if (desc) ffdas_destroy_tensor_desc(desc); }
    ScopedTensorDesc(const ScopedTensorDesc &) = delete;
    ScopedTensorDesc &operator=(const ScopedTensorDesc &) = delete;
};
