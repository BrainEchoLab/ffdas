#pragma once
#include "das/das_alg4.h"

#include <cuda_runtime.h>
#include <math.h>

#include "type_utils.h"
#include "das/das_common.cuh"
#include "das/das_device_utils.cuh"
#include "ffdas_math.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  #define FFDAS_HAVE_SP_MMA 1
#else
  #define FFDAS_HAVE_SP_MMA 0
#endif


__constant__ float3 alg4_channel_pos[DAS_ALG4_MAX_CHANNELS];


#if FFDAS_HAVE_SP_MMA
__device__ inline void sp_mma_m16n8k16_f16(
    const __half2 a0,
    const __half2 a2,
    const __half2 b0,
    const __half2 b2,
    float2 &acc0,
    float2 &acc2,
    const uint32_t metadata
) {
    const uint32_t ra0 = *reinterpret_cast<const uint32_t*>(&a0);
    const uint32_t ra2 = *reinterpret_cast<const uint32_t*>(&a2);
    const uint32_t rb0 = *reinterpret_cast<const uint32_t*>(&b0);
    const uint32_t rb2 = *reinterpret_cast<const uint32_t*>(&b2);

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6, %7}, "
        "{%8, %9, %10, %11}, %12, %13;\n"
        : "=f"(acc0.x), "=f"(acc2.x), "=f"(acc0.y), "=f"(acc2.y)
        : "r"(ra0), "r"(ra2),
          "r"(rb0), "r"(rb2),
          "f"(acc0.x), "f"(acc2.x), "f"(acc0.y), "f"(acc2.y), "r"(metadata), "n"(0x0)
    );
}
#else
__device__ inline void sp_mma_m16n8k16_f16(
    const __half2, 
    const __half2, 
    const __half2, 
    const __half2,
    float2&, 
    float2&, 
    const uint32_t
) { }
#endif


#if FFDAS_HAVE_SP_MMA
__device__ inline void sp_mma_m16n8k16_tf32(
    const float2 a0,
    const float2 a2,
    const float2 b0,
    const float2 b2,
    float2 &acc0,
    float2 &acc2,
    const uint32_t metadata
) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, %16, %17;\n"
        : "=f"(acc0.x), "=f"(acc2.x), "=f"(acc0.y), "=f"(acc2.y)
        : "f"(a0.x), "f"(a2.x), "f"(a0.y), "f"(a2.y),
          "f"(b0.x), "f"(b2.x), "f"(b0.y), "f"(b2.y),
          "f"(acc0.x), "f"(acc2.x), "f"(acc0.y), "f"(acc2.y), "r"(metadata), "n"(0x0)
    );
}
#else
__device__ inline void sp_mma_m16n8k16_tf32(
    const float2 a0,
    const float2 a2,
    const float2 b0,
    const float2 b2,
    float2 &acc0,
    float2 &acc2,
    const uint32_t metadata
) { }
#endif


template<typename T>
__device__ inline void sp_mma_m16n8k16(
    const T a0,
    const T a2,
    const T b0,
    const T b2,
    float2 &acc0,
    float2 &acc2,
    const uint32_t metadata
) { }


template<>
__device__ inline void sp_mma_m16n8k16(
    const __half2 a0,
    const __half2 a2,
    const __half2 b0,
    const __half2 b2,
    float2 &acc0,
    float2 &acc2,
    const uint32_t metadata
) { 
    sp_mma_m16n8k16_f16(a0, a2, b0, b2, acc0, acc2, metadata);
}


template<>
__device__ inline void sp_mma_m16n8k16(
    const float2 a0,
    const float2 a2,
    const float2 b0,
    const float2 b2,
    float2 &acc0,
    float2 &acc2,
    const uint32_t metadata
) { 
    sp_mma_m16n8k16_tf32(a0, a2, b0, b2, acc0, acc2, metadata);
}

typedef enum {
    LEFT,
    RIGHT
} a_frag_side;

template<a_frag_side side, typename Tx>
__device__ __forceinline__
void load_a_frag(Tx weight, float t, Tx *a0, Tx *a2) {}

template<>
__device__ __forceinline__
void load_a_frag<LEFT>(__half2 weight, float t, __half2 *a0, __half2 *a2) {
    const __half2 t2 = __float2half2_rn(t);
    const __half2 tmp = __hfma2(__hneg2(t2), weight, weight);
    *a0 = make_half2(tmp.x, __hneg(tmp.y));
    *a2 = make_half2(tmp.y, tmp.x);
}

template<>
__device__ __forceinline__
void load_a_frag<LEFT>(float2 weight, float t, float2 *a0, float2 *a2) {
    float re = __fmaf_rn(-t, weight.x, weight.x);
    float im = __fmaf_rn(-t, weight.y, weight.y);
    *a0 = make_float2(re, -im);
    *a2 = make_float2(im, re);
}

template<>
__device__ __forceinline__
void load_a_frag<RIGHT>(__half2 weight, float t, __half2 *a0, __half2 *a2) {
    const __half2 t2 = __float2half2_rn(t);
    const __half2 tmp = __hmul2(t2, weight);
    *a0 = make_half2(tmp.x, __hneg(tmp.y));
    *a2 = make_half2(tmp.y, tmp.x);
}

template<>
__device__ __forceinline__
void load_a_frag<RIGHT>(float2 weight, float t, float2 *a0, float2 *a2) {
    float re = t * weight.x;
    float im = t * weight.y;
    *a0 = make_float2(re, -im);
    *a2 = make_float2(im, re);
}


template<int M, int vec_size, typename Tx>
__device__
void load_b_frag(
    const Tx* __restrict__ x, 
    int ldx,
    int row,  // local in B frag
    int rank, 
    const int* __restrict__ index_table,
    Tx *b
) {
    // compute the column this row maps to and find its offset in the count and index table
    const int src = (row / 4) * M + rank;
    int index = index_table[src];
    vector_load<Tx, vec_size>(&x[index * ldx], b);
}


__device__ __forceinline__
bool warp_count_unique(unsigned mask, int x, unsigned &count, unsigned &rank) {
    const unsigned lane = threadIdx.x & 31;

    const unsigned peers = __match_any_sync(mask, x);  // lanes with identical s within the strided group
    const int leader_lane = __ffs(peers) - 1;  // lowest lane holding this s

    unsigned is_leader = lane == (unsigned)leader_lane;
    const unsigned leaders = __ballot_sync(mask, is_leader);

    const unsigned prior = ((1u << leader_lane) - 1u) & mask;  // group lanes with lower lane ids
    const unsigned my_rank = __popc(leaders & prior);  // rank of this value among leaders
    rank = __shfl_sync(mask, my_rank, leader_lane);  // broadcast to all peers
    count = __popc(leaders);  // same for all lanes in the group

    return is_leader;
}

// Build the metadata register for the sparse MMA instruction. Each thread's
// unique rank (nz) selects which 2:4 column pair it occupies. The shfl_down
// operations gather the per-thread nibbles into a single 32-bit register
// shared by the 4-thread group.
template<typename Tx>
__device__ __forceinline__
unsigned pack_metadata(unsigned nz) {
    // if uniq_rank is even (have left pair) 01 {0100} else 23 {1110}
    unsigned metadata = (nz & 1) ? 0x0e : 0x04;
    metadata |= (__shfl_down_sync(0xFFFFFFFF, metadata, 1, 4) << 4);
    metadata |= (__shfl_down_sync(0xFFFFFFFF, metadata, 2, 4) << 8);
    metadata |= (metadata << 16);

    if constexpr(std::is_same<Tx, float2>::value) {
        // need to copy the same metadata to the second thread in each row
        metadata = __shfl_up_sync(0xFFFFFFFF, metadata, 1, 4);
    }

    return metadata;
}


template<typename Tx, typename Ty, int warps_per_block, int vec_size, int M, int N, bool dir_check>
__global__ void das_alg4_kernel(
    int samples, 
    int seqlen,
    int channels,
    const float4 *xdir, 
    float wavenum,
    const Tx *x, 
    int ny, 
    int ystride, 
    const float3 *ypos, 
    const float *offsets, 
    const float *weights, 
    Ty beta, 
    Ty *y, 
    int batch_size
) {
#if FFDAS_HAVE_SP_MMA
    constexpr int chan_per_tile = 4;

    static_assert(M==8, "M must be 8");
    static_assert(N==(chan_per_tile*vec_size*2), "invalid N");

    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x / 32;

    const int n = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * M + (lane / 4);  // per-thread
    const int batch = blockIdx.y * N + (lane / 4) * vec_size;  // per-thread (each thread is mapped to vec_size consecutive cols)

    // ensure that vec_size divides batch_size
    if (batch_size % vec_size)
        return;

    __shared__ int index_table[warps_per_block][32];

    float3 yp = ypos[min(n, ny-1)];
    float2 acc[2*vec_size]{};

    for (int o = 0; o < seqlen; o++) {
        const int to = min(n, ny-1) + o * ystride;
        const float ofs = offsets[to];
        const float scl = weights[to];

        for (int mbase = 0; mbase < channels; mbase += chan_per_tile) {
            int m = mbase + (lane % chan_per_tile);
            const int lane_valid = (m < channels);
            m = lane_valid ? m : channels-1;

            const float3 xp = alg4_channel_pos[m];
            const float dx = yp.x - xp.x;
            const float dy = yp.y - xp.y;
            const float dz = yp.z - xp.z;

            const float rsq = dx * dx + dy * dy + dz * dz;
            const float rinv = rsqrtf(rsq);
            const float phase = ofs + rsq * rinv;

            int k;
            const float fpart = modff_int(phase, &k);

            bool inbounds = (k >= 0) & (k < (samples-1));

            if constexpr (dir_check) {
                float4 xd = xdir[m];
                float proj = (dx * xd.x + dy * xd.y + dz * xd.z) * rinv;
                inbounds &= (proj > xd.w);
            }

            Tx weight{};

            if (lane_valid && inbounds) {
                if constexpr (std::is_same_v<Tx, __half2> || std::is_same_v<Tx, float2> || std::is_same_v<Ty, double2>) { 
                    // complex type
                    float s, c;
                    __sincosf(wavenum * phase, &s, &c);
                    weight = {c * scl, s * scl};
                } else { 
                    // scalar type
                    weight = scl;
                }
            }

            // find unique nonzeros for targets along the columns of A frag
            unsigned uniq_count, uniq_rank;
            const unsigned mask = 0x11111111u << (lane & 3);  // mask all threads in this col (stride 4)
            bool is_leader = warp_count_unique(mask, k, uniq_count, uniq_rank);

            if (is_leader)
                index_table[warp][(lane & 3)*M + uniq_rank] = k;

            // compute which lane holds the column of A frag that corresponds
            // to the rows of B that this thread holds
            int first_count = __shfl_sync(0xFFFFFFFF, uniq_count, (lane & 3) / 2);
            int second_count = __shfl_sync(0xFFFFFFFF, uniq_count, (lane & 3) / 2+2);

            // butterfly max reduction to determine minimum number of mma iterations
            int total = uniq_count;
            total = max(total, __shfl_xor_sync(0xFFFFFFFF, total, 1, 4));
            total = max(total, __shfl_xor_sync(0xFFFFFFFF, total, 2, 4));

            unsigned metadata = pack_metadata<Tx>(uniq_rank);

            // each mma can pack two unique nonzeros for each column of A
            for (int i=0; i < total; i+=2) {
                Tx b0[vec_size], b2[vec_size];
                Tx a0 = Tx{}, a2 = Tx{};

                if ((i + (lane&1)) < first_count && batch < batch_size) {
                    // choose one of the four columns (each has at most M elements) 
                    // and offset by current rank (i*2 + lane%2)

                    const int uniq_index = (lane&3)/2 * M + i + (lane&1);
                    const int s = index_table[warp][uniq_index];

                    const int chan = min(channels-1, mbase + (lane % 4) / 2);
                    const int idx = ((chan * seqlen + o) * samples + s) * batch_size + batch;

                    vector_load<Tx, vec_size>(&x[idx], b0);
                }

                if ((i + (lane&1)) < second_count && batch < batch_size) {
                    const int uniq_index = (2 + (lane&3)/2) * M + i + (lane&1);
                    const int s = index_table[warp][uniq_index];

                    const int chan = min(channels-1, mbase + 2 + (lane % 4) / 2);
                    const int idx = ((chan * seqlen + o) * samples + s) * batch_size + batch;

                    vector_load<Tx, vec_size>(&x[idx], b2);
                }

                if (((uniq_rank == i) || (uniq_rank == i+1))) {
                    load_a_frag<LEFT>(weight, fpart, &a0, &a2);
                }

                __syncwarp();

                #pragma unroll
                for (int j = 0; j < vec_size; j++) {
                    sp_mma_m16n8k16(a0, a2, b0[j], b2[j], acc[j*2], acc[j*2+1], metadata);
                }

                if ((i + (lane&1)) < first_count && batch < batch_size) {
                    // choose one of the four columns (each has at most M elements) 
                    // and offset by current rank (i*2 + lane%2)
                    const int uniq_index = (lane&3)/2 * M + i + (lane&1);
                    const int s = index_table[warp][uniq_index];

                    const int chan = min(channels-1, mbase + (lane % 4) / 2);
                    const int idx = ((chan * seqlen + o) * samples + s + 1) * batch_size + batch;

                    vector_load<Tx, vec_size>(&x[idx], b0);
                }

                if ((i + (lane&1)) < second_count && batch < batch_size) {
                    const int uniq_index = (2 + (lane&3)/2) * M + i + (lane&1);
                    const int s = index_table[warp][uniq_index];

                    const int chan = min(channels-1, mbase + 2 + (lane % 4) / 2);
                    const int idx = ((chan * seqlen + o) * samples + s + 1) * batch_size + batch;

                    vector_load<Tx, vec_size>(&x[idx], b2);
                }

                if ((uniq_rank == i) || (uniq_rank == i+1)) {
                    load_a_frag<RIGHT>(weight, fpart, &a0, &a2);
                }

                __syncwarp();

                #pragma unroll
                for (int j = 0; j < vec_size; j++) {
                    sp_mma_m16n8k16(a0, a2, b0[j], b2[j], acc[j*2], acc[j*2+1], metadata);
                }
            }
        }
    }

    if (n < ny) {
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
            const int b = (blockIdx.y * N) + (lane % 4) * vec_size * 2 + i;

            if (b < batch_size) {
                accumulate_inplace(&y[n + b*ystride], acc[i*2], beta);
            }

            if (b+vec_size < batch_size) {
                accumulate_inplace(&y[n + (b+vec_size)*ystride], acc[i*2+1], beta);
            }
        }
    }
#else
    __trap();
#endif
}
