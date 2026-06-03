#ifndef FFDAS_H_
#define FFDAS_H_

#if (defined(_WIN32) || defined(_WIN64)) && defined(FFDAS_BUILDING_LIBRARY)
    #define FFDAS_API __declspec(dllexport)
#else
    #define FFDAS_API
#endif

#include "ffdas_api.h"

#endif  // FFDAS_H_
