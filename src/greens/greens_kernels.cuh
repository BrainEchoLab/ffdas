#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <mma.h>

#include "type_utils.h"


// Scalar fallback for SM 53-61. Each thread computes one (dst, batch) output
// element per frequency bin, looping over all source channels. Source positions
// are tiled through shared memory to reduce global memory traffic.
template<typename Tx, int TILE_DST, int TILE_BATCH, int SRC_CHUNK = 32>
__global__ void greens_kernel_sm53(
    int samples,
    int channels,
    const float3 *srcpos,
    const float *wavenums,
    const Tx *x,
    int ndst,
    const float3 *dstpos,
    float2 *out,
    int batch_size
) {
    const int dst_base = blockIdx.x * TILE_DST;
    const int batch_base = blockIdx.y * TILE_BATCH;
    const int k = blockIdx.z;

    const int dst_local = threadIdx.x / TILE_BATCH;
    const int batch_local = threadIdx.x % TILE_BATCH;
    const int dst_idx = dst_base + dst_local;
    const int batch_idx = batch_base + batch_local;

    const bool valid = (dst_idx < ndst) && (batch_idx < batch_size);
    const float wavenum = wavenums[k];

    float3 dp;
    if (valid) dp = dstpos[dst_idx];

    float acc_r = 0.0f, acc_i = 0.0f;

    __shared__ float3 srcpos_sh[SRC_CHUNK];

    for (int src_base = 0; src_base < channels; src_base += SRC_CHUNK) {
        int chunk_size = min(SRC_CHUNK, channels - src_base);

        for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
            srcpos_sh[i] = srcpos[src_base + i];
        }
        __syncthreads();

        if (valid) {
            for (int j = 0; j < chunk_size; j++) {
                float3 sp = srcpos_sh[j];
                float dx = dp.x - sp.x;
                float dy = dp.y - sp.y;
                float dz = dp.z - sp.z;
                float rsq = dx * dx + dy * dy + dz * dz;
                float rinv = rsqrtf(rsq);
                float r = rsq * rinv;

                float c, s;
                __sincosf(wavenum * r, &s, &c);
                float gr = c * rinv;
                float gi = s * rinv;

                float2 xv = cast<Tx, float2>(x[batch_idx * (channels * samples) + (src_base + j) * samples + k]);
                acc_r += gr * xv.x - gi * xv.y;
                acc_i += gr * xv.y + gi * xv.x;
            }
        }
        __syncthreads();
    }

    if (valid) {
        out[k * (ndst * batch_size) + dst_idx * batch_size + batch_idx] = make_float2(acc_r, acc_i);
    }
}


// Tensor-core kernel via four real WMMAs: (A_r + jA_i)(B_r + jB_i)
// = (A_r·B_r - A_i·B_i) + j(A_r·B_i + A_i·B_r)
// The function signature compiles on all architectures (needed for the
// host-side launch stub), but the WMMA device code is only emitted for SM 70+.
template<int warps_per_block, int M = 16, int N = 16, int K = 16>
__global__ void greens_kernel(
    int samples,
    int channels,
    const float3 *srcpos,
    const float *wavenums,
    const half2 *x,
    int ny,
    const float3 *dstpos,
    float2 *out,
    int batch_size
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    const int nbase = warp_id * M;
    const int batch = blockIdx.y * N;
    const int k = blockIdx.z;

    int smem_index = (threadIdx.x / warpSize)*M * K;

    int num_valid = min(M, ny - nbase);
    const int batch_valid = min(N, batch_size - batch);

    const float wavenum = wavenums[k];
    __shared__ float3 ypos_shared[warps_per_block * M];
    if (lane_id < M && nbase + lane_id < ny) {
        ypos_shared[(threadIdx.x / warpSize)*M + lane_id] = dstpos[nbase + lane_id];
    }

    __syncthreads();

    fragment<matrix_a, M, N, K, half, row_major> a_r, a_i;
    fragment<matrix_b, M, N, K, half, col_major> b_r, b_i;
    fragment<accumulator, M, N, K, float> acc_rr, acc_ii, acc_ri, acc_ir;

    fill_fragment(acc_rr, 0.0f);
    fill_fragment(acc_ii, 0.0f);
    fill_fragment(acc_ri, 0.0f);
    fill_fragment(acc_ir, 0.0f);

    __shared__ half A_r_sh[warps_per_block * M * K];
    __shared__ half A_i_sh[warps_per_block * M * K];
    __shared__ half B_r_sh[warps_per_block * K * N];
    __shared__ half B_i_sh[warps_per_block * K * N];

    for (int mbase = 0; mbase < channels; mbase += K) {
        int row = lane_id / K;
        int col = lane_id % K;
        int num_rows_per_warp = min(M, warpSize / K);
        int num_cols_per_warp = min(K, warpSize);

        for (int i = row; i < M; i+=num_rows_per_warp) {
            for (int j = col; j < K; j+=num_cols_per_warp) {
                if (nbase + i < ny && mbase + j < channels) {
                    float3 xp = srcpos[mbase + j];
                    float3 yp = ypos_shared[(threadIdx.x / warpSize)*M + i];

                    float dx = yp.x - xp.x;
                    float dy = yp.y - xp.y;
                    float dz = yp.z - xp.z;
                    float rsq = dx * dx + dy * dy + dz * dz;
                    float rinv = rsqrtf(rsq);
                    float phase = rsq * rinv;

                    float c, s;
                    __sincosf(wavenum * phase, &s, &c);

                    A_r_sh[smem_index + i * K + j] = __float2half(c * rinv);
                    A_i_sh[smem_index + i * K + j] = __float2half(s * rinv);
                } else {
                    A_r_sh[smem_index + i * K + j] = __float2half(0.0f);
                    A_i_sh[smem_index + i * K + j] = __float2half(0.0f);
                }
            }
        }
        
        row = lane_id % K;
        col = lane_id / K;
        num_cols_per_warp = min(M, warpSize / K);
        num_rows_per_warp = min(K, warpSize);

        for (int i = col; i < N; i+=num_cols_per_warp) {
            for (int j = row; j < K; j+=num_rows_per_warp) {
                if (i < batch_valid && mbase + j < channels) {
                    half2 v = x[(batch + i) * (channels * samples) + (mbase + j) * samples + k];
                    B_r_sh[smem_index + i * K + j] = v.x;
                    B_i_sh[smem_index + i * K + j] = v.y;
                } else {
                    B_r_sh[smem_index + i * K + j] = __float2half(0.0f);
                    B_i_sh[smem_index + i * K + j] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        load_matrix_sync(a_r, A_r_sh + smem_index, K);
        load_matrix_sync(a_i, A_i_sh + smem_index, K);
        load_matrix_sync(b_r, B_r_sh + smem_index, K);
        load_matrix_sync(b_i, B_i_sh + smem_index, K);

        mma_sync(acc_rr, a_r, b_r, acc_rr);
        mma_sync(acc_ii, a_i, b_i, acc_ii);
        mma_sync(acc_ri, a_r, b_i, acc_ri);
        mma_sync(acc_ir, a_i, b_r, acc_ir);
    }

    __shared__ float RR_sh[warps_per_block * M * N];
    __shared__ float II_sh[warps_per_block * M * N];
    __shared__ float RI_sh[warps_per_block * M * N];
    __shared__ float IR_sh[warps_per_block * M * N];

    store_matrix_sync(RR_sh + smem_index, acc_rr, N, mem_row_major);
    store_matrix_sync(II_sh + smem_index, acc_ii, N, mem_row_major);
    store_matrix_sync(RI_sh + smem_index, acc_ri, N, mem_row_major);
    store_matrix_sync(IR_sh + smem_index, acc_ir, N, mem_row_major);
    
    __syncthreads();

    int row = lane_id / K;
    int col = lane_id % K;
    int num_rows_per_warp = min(M, warpSize / K);
    int num_cols_per_warp = min(K, warpSize);

    for (int i = row; i < num_valid; i+=num_rows_per_warp) {
        for (int j = col; j < batch_valid; j+=num_cols_per_warp) {
            float2 v = make_float2(
                RR_sh[smem_index + i * N + j] - II_sh[smem_index + i * N + j],
                RI_sh[smem_index + i * N + j] + IR_sh[smem_index + i * N + j]
            );

            out[k * (ny * batch_size) + (nbase + i) * batch_size + batch + j] = v;
        }
    }
#endif  // __CUDA_ARCH__ >= 700
}
