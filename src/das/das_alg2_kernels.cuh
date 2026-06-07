#pragma once
#include "das_alg2.h"

#include <cuda_runtime.h>
#include <math.h>

#include "type_utils.h"
#include "das_common.cuh"
#include "das_device_utils.cuh"
#include "math/core.cuh"
#include "math/vector.cuh"
#include "math/bitops.cuh"


__constant__ float3 alg2_channel_pos[DAS_ALG2_MAX_CHANNELS];


template<typename Tx, typename Ty, int warps_per_block, int vec_size, int M, int N, bool dir_check>
__global__ void das_alg2_kernel(
    int samples, 
    int seqlen,
    int channels,
    const float4 *srcdir, 
    float wavenum,
    const Tx *x, 
    int ndst, 
    int outstride, 
    const float3 *dstpos, 
    const float *offsets, 
    const float *weights, 
    Ty beta, 
    Ty *out, 
    int batch_size
) {
    static_assert(M<=32, "M must be <= 32");
    static_assert(((M * N) % 32) == 0, "M*N must be a multiple of 32");

    // compute the maximum size (in elements) for the vector loads
    constexpr int elem_per_thread = (M * N) / 32;

    constexpr int N_bytes = N * sizeof(Tx);
    constexpr int loads_per_ld = ((N_bytes > 128) ? 128 : N_bytes) / (sizeof(Tx) * vec_size);

    // we distribute the 32 threads in a warp across 
    // a subtile by first filling one leading dimension block 
    // (or at most one cache line), then up to M rows,
    // and finally recompute the number of columns 
    // (multiple of the ld block), based on the actual M
    // so: for one channel, a subtile will contain
    // (rows_per_subtile, cols_per_subtile * vec_size) elements
    constexpr int max_rows_per_subtile = (loads_per_ld < 32) ? (32 / loads_per_ld) : 1;
    constexpr int rows_per_subtile = (M > max_rows_per_subtile) ? max_rows_per_subtile : M;
    constexpr int cols_per_subtile = 32 / rows_per_subtile;

    constexpr int chan_per_warp = 32 / M;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;

    const int target_base = (tid / 32) * M;
    const int batch_base = blockIdx.y * N;

    // ensure that vec_size divides batch_size
    if (batch_size % vec_size)
        return;

    if (target_base >= ndst || batch_base >= batch_size)
        return;  // must be same across warp

    // position of this thread within a subtile
    const int row = lane / cols_per_subtile;
    const int col = batch_base + (lane % cols_per_subtile) * vec_size;

    const float3 yp = dstpos[min(target_base + (lane % M), ndst-1)];
    __shared__ Ty shared_weight[warps_per_block][32];

    Ty accum[elem_per_thread]{};

    for (int o = 0; o < seqlen; o++) {
        const int to = min(target_base + (lane % M), ndst-1) + o * outstride;
        const float ofs = offsets[to];
        const float scl = weights[to];

        const int last_ch = channels - 1;

        for (int c = 0; c < channels; c += chan_per_warp) {
            const int lane_k = lane / M;
            const int src_ch_lane = c + lane_k;
            const int lane_valid = (src_ch_lane < channels);
            const int lane_ch = (lane_valid ? src_ch_lane : last_ch);


            const float3 xp = alg2_channel_pos[lane_ch];
            const float dx = yp.x - xp.x;
            const float dy = yp.y - xp.y;
            const float dz = yp.z - xp.z;

            const float rsq = max(dx * dx + dy * dy + dz * dz, 1e-7);  // TODO: maybe take epsilon value as argument
            const float rinv = rsqrtf(rsq);
            const float phase = ofs + rsq * rinv;

            int ipart;
            const float fpart = modff_int(phase, &ipart);

            bool inbounds = (ipart >= 0) & (ipart < (samples-1));

            if constexpr (dir_check) {
                float4 xd = srcdir[lane_ch];
                float proj = (dx * xd.x + dy * xd.y + dz * xd.z) * rinv;
                inbounds &= (proj > xd.w);
            }

            Ty weight{};

            if (lane_valid && inbounds) {
                if constexpr (std::is_same_v<Ty, __half2> || std::is_same_v<Ty, float2> || std::is_same_v<Ty, double2>) { 
                    // complex type
                    float s, c;
                    __sincosf(wavenum * phase, &s, &c);
                    weight = {c * scl, s * scl};
                } else { 
                    // scalar type
                    weight = scl;
                }
            }

            shared_weight[warp][lane] = weight;

            const __half2 index = make_half2(__short_as_half((short)ipart), __float2half_rn(fpart));

            #pragma unroll chan_per_warp
            for (int k = 0; k < chan_per_warp; k++) {
                Ty *acc_ptr = &accum[0];

                #pragma unroll (M / rows_per_subtile)
                for (int m = 0; m < M; m += rows_per_subtile) {
                    const Ty weight = shared_weight[warp][k * M + row + m];
                    const __half2 val = __shfl_sync(0xFFFFFFFFu, index, k * M + row + m);

                    const int ch_k = (c + k < channels) ? (c + k) : last_ch;
                    const Tx *x_ptr = x + (((ch_k * seqlen + o) * samples + (int)__half_as_short(val.x)) * batch_size);

                    #pragma unroll // N / (cols_per_subtile * vec_size)
                    for (int n = 0; n < N; n+=(cols_per_subtile*vec_size)) {
                        Tx v0[vec_size], v1[vec_size];

                        // clamp n such that the offset does not exceed batch_size
                        // the clamped samples will be skipped anyway in the output
                        vector_load<Tx, vec_size>(x_ptr + min(col+n, batch_size-vec_size), v0);
                        vector_load<Tx, vec_size>(x_ptr + min(col+n, batch_size-vec_size) + batch_size, v1);

                        #pragma unroll (vec_size)
                        for (int i = 0; i < vec_size; i++) {
                            cmadd_inplace(weight, cast<Tx,Ty>(lerp(v0[i], v1[i], val.y)), *acc_ptr++);
                        }
                    }
                }
            }
        }
    }

    Ty *acc_ptr = &accum[0];

#pragma unroll (M / rows_per_subtile)
    for (int m = row; m < M; m+=rows_per_subtile) {

        if (target_base + m < ndst) {
            #pragma unroll // (N / (cols_per_subtile*vec_size))
            for (int n = col; n < batch_size; n+=(cols_per_subtile*vec_size)) {
                int out_ofs = target_base + m + n * outstride;
                
                for (int i = 0; i < vec_size; i++) {
                    accumulate_inplace(&out[out_ofs + i*outstride], *acc_ptr++, beta);
                }
            }
        }
    }
}
