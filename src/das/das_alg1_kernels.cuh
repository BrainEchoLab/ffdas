#pragma once
#include "das/das_alg1.h"

#include <cuda_runtime.h>
#include <math.h>

#include "type_utils.h"
#include "das/das_common.cuh"
#include "das/das_device_utils.cuh"
#include "ffdas_math.cuh"


template<typename Tx, typename Ty, int tile_width, bool is_sparse, bool dir_check>
__global__ void das_alg1_kernel(
    int samples, 
    int seqlen,
    int channels,
    const float3 *xpos, 
    const float4 *xdir, 
    float wavenum,
    const Tx *x, 
    int ny, 
    int ystride, 
    const float3 *ypos, 
    const float *offsets, 
    const float *weights, 
    const int *sparse_indices, 
    Ty beta, 
    Ty *y, 
    int batch_size
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_base = blockIdx.y * tile_width;

    if (n >= ny || batch_base >= batch_size) 
        return;

    const float3 yp = ypos[n];

    // clamp the batch size for this block to the total batch size
    const int actual_batch_size = min(tile_width, batch_size - batch_base);

    Ty accum[tile_width]{};

    for (int i = 0; i < seqlen; i++) {
        int seqidx = i;

        if constexpr(is_sparse) {
            seqidx = sparse_indices[i * ystride];
        }

        const float ofs = offsets[n + seqidx * ystride];
        const float scl = weights[n + seqidx * ystride];

        for (int m = 0; m < channels; m++) {
            const float3 xp = xpos[m];
            const float dx = yp.x - xp.x;
            const float dy = yp.y - xp.y;
            const float dz = yp.z - xp.z;

            const float rsq = max(dx * dx + dy * dy + dz * dz, 1e-7);  // TODO: maybe take epsilon value as argument
            const float rinv = rsqrtf(rsq);
            const float phase = ofs + rsq * rinv;

            int k;
            const float fpart = modff_int(phase, &k);
            bool inbounds = (k >= 0) & (k < (samples - 1));

            if constexpr (dir_check) {
                float4 xd = xdir[m];
                float proj = (dx * xd.x + dy * xd.y + dz * xd.z) * rinv;
                inbounds &= (proj > xd.w);
            }

            Ty weight{};

            if (inbounds) {
                if constexpr (std::is_same_v<Ty, __half2> || std::is_same_v<Ty, float2> || std::is_same_v<Ty, double2>) { 
                    // complex type
                    float s, c;
                    __sincosf(wavenum * phase, &s, &c);
                    weight = {c * scl, s * scl};
                } else { 
                    // scalar type
                    weight = cast<float, Ty>(scl);
                }
            }

            for (int b = 0; b < actual_batch_size; b++) {
                int offs0 = (((batch_base + b) * channels + m) * seqlen + seqidx) * samples + k;
                accum[b] = cmadd(
                    weight, 
                    cast<Tx,Ty>(lerp(x[offs0], x[offs0+1], fpart)), 
                    accum[b]
                );
            }
        }
    }

    for (int b = 0; b < actual_batch_size; b++) {
        int idx = n + (batch_base + b) * ystride;
        accumulate_inplace(&y[idx], accum[b], beta);
    }
}
