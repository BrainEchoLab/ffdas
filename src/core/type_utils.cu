#include "ffdas.h"
#include "type_utils.h"

// Function to get the size of the type at runtime
size_t ffdas_type_size(ffdas_datatype_t datatype) {
    switch (datatype) {
        case FFDAS_R_16F: 
            return builtin_traits<ffdas_traits<FFDAS_R_16F>::type>::size;
        case FFDAS_C_16F: 
            return builtin_traits<ffdas_traits<FFDAS_C_16F>::type>::size;
        case FFDAS_R_32F: 
            return builtin_traits<ffdas_traits<FFDAS_R_32F>::type>::size;
        case FFDAS_C_32F: 
            return builtin_traits<ffdas_traits<FFDAS_C_32F>::type>::size;
        case FFDAS_R_64F: 
            return builtin_traits<ffdas_traits<FFDAS_R_64F>::type>::size;
        case FFDAS_C_64F: 
            return builtin_traits<ffdas_traits<FFDAS_C_64F>::type>::size;
        case FFDAS_R_32I: 
            return builtin_traits<ffdas_traits<FFDAS_R_32I>::type>::size;
        case FFDAS_C_32I: 
            return builtin_traits<ffdas_traits<FFDAS_C_32I>::type>::size;
        case FFDAS_R_16I: 
            return builtin_traits<ffdas_traits<FFDAS_R_16I>::type>::size;
        case FFDAS_C_16I: 
            return builtin_traits<ffdas_traits<FFDAS_C_16I>::type>::size;
        default: 
            return 0;
    }
}
