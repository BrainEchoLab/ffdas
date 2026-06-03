#pragma once

typedef enum {
    FFDAS_R_16F = 2,
    FFDAS_C_16F = 6,
    FFDAS_R_32F = 0,
    FFDAS_C_32F = 4,
    FFDAS_R_64F = 16,
    FFDAS_C_64F = 17,
    FFDAS_R_32I = 10,
    FFDAS_C_32I = 11,
    FFDAS_R_16I = 14,
    FFDAS_C_16I = 15
} ffdas_datatype_t;

typedef enum {
    FFDAS_COMPUTE_DEFAULT = 0,
    FFDAS_COMPUTE_16F = 1,
    FFDAS_COMPUTE_32F = 2,
    FFDAS_COMPUTE_64F = 3
} ffdas_compute_type_t;
