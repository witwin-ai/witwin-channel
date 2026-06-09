#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace raydn {

constexpr float kSmallEps = 1e-12f;
constexpr float kDistanceEps = 1e-20f;
constexpr float kRayTMin = 1e-5f;
constexpr float kRayBias = 1e-5f;
constexpr float kDfrRayBias = 1e-4f;
constexpr float kRayTMax = 1e8f;
constexpr float kPi = 3.14159265358979323846f;

__forceinline__ __host__ __device__ float3 make_f3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__forceinline__ __device__ float3 make_f3(const float *ptr) {
    return make_float3(ptr[0], ptr[1], ptr[2]);
}

__forceinline__ __host__ __device__ float3 f3_zero() {
    return make_float3(0.0f, 0.0f, 0.0f);
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __host__ __device__ float3 operator*(float s, float3 a) {
    return a * s;
}

__forceinline__ __host__ __device__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__forceinline__ __host__ __device__ float3 add3(float3 a, float3 b) {
    return a + b;
}

__forceinline__ __host__ __device__ float3 sub3(float3 a, float3 b) {
    return a - b;
}

__forceinline__ __host__ __device__ float3 mul3(float s, float3 a) {
    return s * a;
}

__forceinline__ __host__ __device__ float3 mul3(float3 a, float s) {
    return a * s;
}

__forceinline__ __host__ __device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __host__ __device__ float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__forceinline__ __host__ __device__ float3 cross(float3 a, float3 b) {
    return cross3(a, b);
}

__forceinline__ __host__ __device__ float norm2_3(float3 a) {
    return dot3(a, a);
}

__forceinline__ __host__ __device__ float squared_norm(float3 a) {
    return norm2_3(a);
}

__forceinline__ __host__ __device__ float norm3(float3 a) {
    return sqrtf(fmaxf(dot3(a, a), 0.0f));
}

__forceinline__ __host__ __device__ float length3(float3 a) {
    return norm3(a);
}

__forceinline__ __device__ float3 normalize3(float3 v, float eps = kSmallEps) {
    const float inv_len = rsqrtf(fmaxf(dot3(v, v), eps));
    return inv_len * v;
}

__forceinline__ __device__ void atomic_add3(float *base, int index, float3 value) {
    atomicAdd(&base[index * 3 + 0], value.x);
    atomicAdd(&base[index * 3 + 1], value.y);
    atomicAdd(&base[index * 3 + 2], value.z);
}

__forceinline__ __device__ float warp_sum_masked(unsigned int mask, float value) {
    float sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane) {
        if ((mask & (1u << lane)) != 0u)
            sum += __shfl_sync(mask, value, lane);
    }
    return sum;
}

__forceinline__ __device__ int warp_sum_masked(unsigned int mask, int value) {
    int sum = 0;
    for (int lane = 0; lane < 32; ++lane) {
        if ((mask & (1u << lane)) != 0u)
            sum += __shfl_sync(mask, value, lane);
    }
    return sum;
}

__forceinline__ __device__ bool warp_mask_leader(unsigned int mask) {
    return static_cast<int>(threadIdx.x & 31u) == (__ffs(mask) - 1);
}

struct WarpCellGroup {
    unsigned int peers = 0u;
    int count = 0;
    bool leader = false;
};

__forceinline__ __device__ WarpCellGroup warp_cell_group(int index) {
    WarpCellGroup group;
    group.peers = __match_any_sync(__activemask(), index);
    group.count = __popc(group.peers);
    group.leader = warp_mask_leader(group.peers);
    return group;
}

__forceinline__ __device__ void atomic_add_same_cell(float *base, int index, float value) {
    const unsigned int active = __activemask();
    const unsigned int peers = __match_any_sync(active, index);
    if (__popc(peers) == 1) {
        atomicAdd(base + index, value);
        return;
    }
    const float sum = warp_sum_masked(peers, value);
    if (warp_mask_leader(peers))
        atomicAdd(base + index, sum);
}

__forceinline__ __device__ void atomic_add_same_cell(
    float *base,
    int index,
    float value,
    WarpCellGroup group) {
    if (group.count == 1) {
        atomicAdd(base + index, value);
        return;
    }
    const float sum = warp_sum_masked(group.peers, value);
    if (group.leader)
        atomicAdd(base + index, sum);
}

__forceinline__ __device__ void atomic_add_same_cell(int *base, int index, int value) {
    const unsigned int active = __activemask();
    const unsigned int peers = __match_any_sync(active, index);
    if (__popc(peers) == 1) {
        atomicAdd(base + index, value);
        return;
    }
    const int sum = warp_sum_masked(peers, value);
    if (warp_mask_leader(peers))
        atomicAdd(base + index, sum);
}

__forceinline__ __device__ void atomic_add_same_cell(
    int *base,
    int index,
    int value,
    WarpCellGroup group) {
    if (group.count == 1) {
        atomicAdd(base + index, value);
        return;
    }
    const int sum = warp_sum_masked(group.peers, value);
    if (group.leader)
        atomicAdd(base + index, sum);
}

__forceinline__ __device__ void atomic_add_warp(float *base, float value) {
    const unsigned int active = __activemask();
    const float sum = warp_sum_masked(active, value);
    if (warp_mask_leader(active))
        atomicAdd(base, sum);
}

__forceinline__ __device__ void atomic_add_warp(int *base, int value) {
    const unsigned int active = __activemask();
    const int sum = warp_sum_masked(active, value);
    if (warp_mask_leader(active))
        atomicAdd(base, sum);
}

} // namespace raydn
