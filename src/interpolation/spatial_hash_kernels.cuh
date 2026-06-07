#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#include "ffdas.h"
#include "error_checking.h"



__device__ inline uint32_t expand_bits_2(uint32_t v) {
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

__device__ inline uint32_t morton_2d(uint32_t x, uint32_t y) {
    return (expand_bits_2(x)) | 
           (expand_bits_2(y) << 1);
}


__device__ inline uint32_t expand_bits_3(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8 )) & 0x0300F00F;
    v = (v | (v << 4 )) & 0x030C30C3;
    v = (v | (v << 2 )) & 0x09249249;
    return v;
}


__device__ inline uint32_t morton_3d(uint32_t x, uint32_t y, uint32_t z) {
    return (expand_bits_3(x)) | 
           (expand_bits_3(y) << 1) | 
           (expand_bits_3(z) << 2);
}


__device__ inline float3 cross_3d(const float3 &a, const float3 &b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}


__device__ inline float dot_3d(const float3 &a, const float3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}


__device__ inline int float4_max_by_key(const float4 &v, const int4 &k) {
    int idx_xy = (v.x > v.y) ? k.x : k.y;
    float max_xy = (idx_xy == k.x) ? v.x : v.y;

    int idx_zw = (v.z > v.w) ? k.z : k.w;
    float max_zw = (idx_zw == k.z) ? v.z : v.w;

    return (max_xy > max_zw) ? idx_xy : idx_zw;
}


__device__ inline bool point_inside_simplex(
    const float3 &p,
    const float3 &v0,
    const float3 &v1,
    const float3 &v2,
    const float3 &v3,
    float4 &bary
) {
    float3 v10 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 v20 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 v30 = make_float3(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z);
    float3 vp  = make_float3(p.x - v0.x, p.y - v0.y, p.z - v0.z);
    float denom = dot_3d(v10, cross_3d(v20, v30));

    if (fabsf(denom) < 1e-12f)
        return false;

    bary.y = dot_3d(vp, cross_3d(v20, v30)) / denom;
    bary.z = dot_3d(v10, cross_3d(vp, v30)) / denom;
    bary.w = dot_3d(v10, cross_3d(v20, vp )) / denom;
    bary.x = 1.0f - bary.y - bary.z - bary.w;

    return (
        bary.x >= 0.0f && bary.y >= 0.0f &&
        bary.z >= 0.0f && bary.w >= 0.0f
    );
}


static __global__ void morton_transform_kernel(
    int num_points,
    const float3* __restrict__ points,
    uint32_t* __restrict__ codes, 
    int nx, int ny, int nz,
    float x0, float y0, float z0,
    float dx, float dy, float dz
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_points)
        return;

    float3 p = points[tid];

    int xi = __float2uint_rn((p.x - x0) * dx);
    int yi = __float2uint_rn((p.y - y0) * dy);
    int zi = __float2uint_rn((p.z - z0) * dz);

    if (xi < 0 || yi < 0 || zi < 0 || xi > nx-1 || yi > ny-1 || zi > nz-1) {
        codes[tid] = 0xFFFFFFFFu;
        return;
    }

    codes[tid] = morton_3d(xi, yi, zi);
}


static __global__ void quantize_points_kernel(
    int num_points,
    const float3* __restrict__ points,
    int3* __restrict__ quantized,
    float x0, float y0, float z0,
    float dx, float dy, float dz
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_points)
        return;

    float3 p = points[tid];

    quantized[tid] = make_int3(
        __float2uint_rn((p.x - x0) * dx),
        __float2uint_rn((p.y - y0) * dy),
        __float2uint_rn((p.z - z0) * dz)
    );
}


static __global__ void make_simplices_kernel(
    int nx, int ny, int nz,
    int4 *ids
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int cx = tid % (nx-1);
    int cy = (tid / (nx-1)) % (ny-1);
    int cz = (tid / ((nx-1)*(ny-1)));
    int num_cells = (nz-1)*(ny-1)*(nx-1);

    if (tid >= num_cells)
        return;

    int base = cx + cy * nx + cz * ny*nx;
    int v000 = base;
    int v100 = base + 1;
    int v010 = base + nx;
    int v110 = v010 + 1;
    int v001 = base + ny*nx;
    int v101 = v001 + 1;
    int v011 = v001 + nx;
    int v111 = v011 + 1;

    int T[6][4] = {
        {v000,v100,v010,v001},
        {v100,v110,v010,v111},
        {v100,v010,v001,v111},
        {v010,v001,v011,v111},
        {v100,v001,v101,v111},
        {v001,v011,v101,v111}
    };

    int id_offset = tid * 6;  // 6 tetrahedra per cell

#pragma unroll
    for (int t=0; t<6; t++) {
        ids[id_offset + t] = make_int4(
            T[t][0],
            T[t][1],
            T[t][2],
            T[t][3]
        );
    }
}


static __global__ void rasterize_simplices_kernel(
    const int3* __restrict__ quantized_points,
    int num_simplices,
    const int4* __restrict__ vertex_ids,
    int4* __restrict__ raster_info,
    int nx, int ny, int nz
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_simplices)
        return;

    int4 ids = vertex_ids[tid];

    int3 v0 = quantized_points[ids.x];
    int3 v1 = quantized_points[ids.y];
    int3 v2 = quantized_points[ids.z];
    int3 v3 = quantized_points[ids.w];

    int xmin = max(0, min(min(v0.x, v1.x), min(v2.x, v3.x)));
    int ymin = max(0, min(min(v0.y, v1.y), min(v2.y, v3.y)));
    int zmin = max(0, min(min(v0.z, v1.z), min(v2.z, v3.z)));
    int xmax = min(nx-1, max(max(v0.x, v1.x), max(v2.x, v3.x)));
    int ymax = min(ny-1, max(max(v0.y, v1.y), max(v2.y, v3.y)));
    int zmax = min(nz-1, max(max(v0.z, v1.z), max(v2.z, v3.z)));

    int lin_idx = xmin + ymin * nx + zmin * nx*ny;

    int xsize = xmax - xmin + 1;
    int ysize = ymax - ymin + 1;
    int zsize = zmax - zmin + 1;

    raster_info[tid] = make_int4(xsize, ysize, zsize, lin_idx);
}


static __global__ void hash_simplices_kernel(
    int num_simplices,
    const int4* __restrict__ raster_info,
    const int* __restrict__ simplex_offsets,
    uint32_t* __restrict__ hashes,
    int* __restrict__ hash_ids,
    int nx, int ny, int nz
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_simplices)
        return;

    int base = simplex_offsets[tid];
    int4 info = raster_info[tid];

    int x0 = info.w % nx;
    int y0 = (info.w / nx) % ny;
    int z0 = (info.w / (nx*ny));

    int idx = 0;

    for (int k = z0; k < z0+info.z; k++) {
        for (int j = y0; j < y0+info.y; j++) {
            for (int i = x0; i < x0+info.x; i++) {
                uint32_t code = morton_3d(i, j, k);
                hashes[base+idx] = code;
                hash_ids[base+idx] = tid;
                idx++;
            }
        }
    }
}


static __global__ void query_hash_kernel(
    int num_points,
    const float3* __restrict__ points,
    const uint32_t* __restrict__ hash,
    const int* __restrict__ bin_offsets,
    const int* __restrict__ hash_ids,
    const int4* __restrict__ simplex_vertex_ids,
    const float3* __restrict__ vertices,
    int* __restrict__ nearest_vertex_ids,
    int4* __restrict__ vertex_ids,
    float4* __restrict__ barycentric
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_points)
        return;

    uint32_t code = hash[tid];
    float3 p = points[tid];

    if (code == 0xFFFFFFFFu) {
        if (nearest_vertex_ids) nearest_vertex_ids[tid] = -1;
        if (vertex_ids) vertex_ids[tid] = make_int4(-1, -1, -1, -1);
        return;
    }

    int start = bin_offsets[code];
    int stop = bin_offsets[code+1];

    for (int i = start; i < stop; i++) {
        int simplex_id = hash_ids[i];
        int4 v_ids = simplex_vertex_ids[simplex_id];

        float3 v0 = vertices[v_ids.x];
        float3 v1 = vertices[v_ids.y];
        float3 v2 = vertices[v_ids.z];
        float3 v3 = vertices[v_ids.w];
        float4 bary;

        bool inside = point_inside_simplex(
            p,
            v0,
            v1,
            v2,
            v3,
            bary
        );

        if (inside) {
            if (nearest_vertex_ids) nearest_vertex_ids[tid] = float4_max_by_key(bary, v_ids);
            if (vertex_ids) vertex_ids[tid] = v_ids;
            if (barycentric) barycentric[tid] = bary;
            return;
        }
    }

    if (nearest_vertex_ids) nearest_vertex_ids[tid] = -1;
    if (vertex_ids) vertex_ids[tid] = make_int4(-1, -1, -1, -1);
}
