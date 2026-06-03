#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <mma.h>

#include "type_utils.h"

using namespace nvcuda::wmma;


// Complex matrix multiply via four real WMMAs: (A_r + jA_i)(B_r + jB_i)
// = (A_r·B_r - A_i·B_i) + j(A_r·B_i + A_i·B_r)
template<int warps_per_block, int M = 16, int N = 16, int K = 16>
__global__ void greens_sum_kernel(
    int samples,  // S samples (frequency bins)
    int channels,  // M channels (sources)
    const float3 *xpos,  // [M]: positions of each source
    const float *wavenums,  // [S]: wave number k per sample
    const half2 *x,  // [M * S]: half2 samples (real, imag)
    int ny,
    const float3 *ypos,  // [N]: positions of each target point
    float2 *y,  // [N * S]: output complex float per target-sample
    int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    const int nbase = warp_id * M;  // start of the target block for this warp
    const int batch = blockIdx.y * N;
    const int k = blockIdx.z;

    int smem_index = (threadIdx.x / warpSize)*M * K;

    int num_valid = min(M, ny - nbase);
    const int batch_valid = min(N, batch_size - batch);

    const float wavenum = wavenums[k];
    __shared__ float3 ypos_shared[warps_per_block * M];
    if (lane_id < M && nbase + lane_id < ny) {
        ypos_shared[(threadIdx.x / warpSize)*M + lane_id] = ypos[nbase + lane_id];
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

    // for matrix a (kernel, row-major)
    // int row, col, num_rows_per_warp, num_cols_per_warp;

    for (int mbase = 0; mbase < channels; mbase += K) {
        // row major
        int row = lane_id / K;
        int col = lane_id % K;
        int num_rows_per_warp = min(M, warpSize / K);
        int num_cols_per_warp = min(K, warpSize);

        // build kernel tile
        for (int i = row; i < M; i+=num_rows_per_warp) {
            for (int j = col; j < K; j+=num_cols_per_warp) {
                if (nbase + i < ny && mbase + j < channels) {
                    float3 xp = xpos[mbase + j];
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
        
        // col major
        row = lane_id % K;
        col = lane_id / K;
        num_cols_per_warp = min(M, warpSize / K);
        num_rows_per_warp = min(K, warpSize);

        // build input tile
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

    // row major
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

            y[k * (ny * batch_size) + (nbase + i) * batch_size + batch + j] = v;
        }
    }
}
