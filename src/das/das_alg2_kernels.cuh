#pragma once
#include "das_alg2.h"

#include <cuda_runtime.h>
#include <math.h>

#include "type_utils.h"
#include "das_device_utils.cuh"
#include "math/core.cuh"
#include "math/vector.cuh"
#include "math/bitops.cuh"


__constant__ float3 alg2_channel_pos[DAS_ALG2_MAX_CHANNELS];


// launched with warps_per_block * 32 = 1024 threads (das_alg2.cu): without the bound nvcc may give a
// variant more than 64 registers per thread, and its launch then fails ("too many resources requested
// for launch", seen on SM 7.5)
template<typename Tx, typename Ty, int warps_per_block, int vec_size, int M, int N, bool dir_check>
__global__ void __launch_bounds__(warps_per_block * 32) das_alg2_kernel(
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

    // each thread accumulates vec_size consecutive columns per (row, column step) of its tile. A subtile
    // can be wider than the tile (cols_per_subtile * vec_size > N, e.g. M = 2 rows, N = 16, vec_size 2):
    // then there is still one column step, so the count is not simply M * N / 32
    constexpr int col_step = cols_per_subtile * vec_size;
    constexpr int elem_per_thread = (M / rows_per_subtile) * ((N + col_step - 1) / col_step) * vec_size;

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

    // offset of the second interpolation tap; 0 for a one-sample recording, where every index is unused
    const int next_row = (samples > 1) ? batch_size : 0;

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

            // a sample outside [0, samples-2] (or rejected by directivity) is sent to the loading
            // threads as index 0 (rows 0 and 1, inside the data) with a fraction of -0, which tells
            // them to use 0 for the value (#2). A used sample's fraction is never negative
            const bool use = lane_valid && inbounds;

            Ty weight{};

            if (use) {
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

            const __half2 index = make_half2(__short_as_half((short)(use ? ipart : 0)),
                                             use ? __float2half_rn(fpart) : __ushort_as_half(0x8000));

            #pragma unroll chan_per_warp
            for (int k = 0; k < chan_per_warp; k++) {
                Ty *acc_ptr = &accum[0];

                #pragma unroll (M / rows_per_subtile)
                for (int m = 0; m < M; m += rows_per_subtile) {
                    const Ty weight = shared_weight[warp][k * M + row + m];
                    const __half2 val = __shfl_sync(0xFFFFFFFFu, index, k * M + row + m);

                    const bool use_s = !(__half_as_ushort(val.y) & 0x8000u);
                    const int ch_k = (c + k < channels) ? (c + k) : last_ch;
                    const Tx *x_ptr = x + (((ch_k * seqlen + o) * samples + (int)__half_as_short(val.x)) * batch_size);

                    #pragma unroll // N / col_step
                    for (int n = 0; n < N; n+=col_step) {
                        Tx v0[vec_size], v1[vec_size];

                        // clamp n such that the offset does not exceed batch_size
                        // the clamped samples will be skipped anyway in the output
                        vector_load<Tx, vec_size>(x_ptr + min(col+n, batch_size-vec_size), v0);
                        vector_load<Tx, vec_size>(x_ptr + min(col+n, batch_size-vec_size) + next_row, v1);

                        #pragma unroll (vec_size)
                        for (int i = 0; i < vec_size; i++) {
                            const Tx sample = lerp(v0[i], v1[i], val.y);
                            cmadd_inplace(weight, cast<Tx,Ty>(use_s ? sample : Tx{}), *acc_ptr++);
                        }
                    }
                }
            }
        }
    }

    // write back in the order the accumulators were filled, and only the columns of this block's tile:
    // a column past the tile belongs to the next block along y
    const int batch_end = min(batch_size, batch_base + N);
    Ty *acc_ptr = &accum[0];

#pragma unroll (M / rows_per_subtile)
    for (int m = row; m < M; m+=rows_per_subtile) {
        #pragma unroll
        for (int n = 0; n < N; n+=col_step) {
            #pragma unroll (vec_size)
            for (int i = 0; i < vec_size; i++, acc_ptr++) {
                const int b = col + n + i;
                if (target_base + m < ndst && b < batch_end) {
                    accumulate_inplace(&out[target_base + m + b * outstride], *acc_ptr, beta);
                }
            }
        }
    }
}
