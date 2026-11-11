// Copyright Xingyu Chen.
// Declares the BDPT wavefront state and Kirchhoff scatter device contract shared by its kernels.

#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#include <rayd/scattering_table.cuh>

#include "math.cuh"
#endif

namespace channel::bdpt_wavefront {

constexpr int kMaskReflection = 2;
constexpr int kMaskTransmission = 8;
constexpr int kMaskScattering = 16;
// Transient bit: the subpath scattered at the current bounce and has a scatter
// vertex recorded for next-event estimation.
constexpr int kMaskScatterVertex = 32;

// Per-sample wavefront state of one chunk (structure of arrays).
struct State {
    float* origin;
    float* direction;
    float* field_real;
    float* field_imag;
    float* pdf_forward;
    float* path_length;
    // Post-scatter unpolarized amplitude carrier (|carrier|^2 is the power weight).
    float* carrier;
    uint8_t* component_mask;
    uint8_t* scatter_count;
    bool* valid;
    // Scatter vertex of the current bounce for next-event estimation.
    float* nee_position;
    float* nee_normal;
    float* nee_wi_local;
    float* nee_p_te;
    float* nee_p_tm;
    float* nee_p_scatter;
    float* nee_path_length;
    int* nee_material;
};

// Resident Kirchhoff tables stacked per rough-material slot, plus whether mixed
// reflection+transmission vertices are connected (the transmission component).
struct ScatterTables {
    bool emit_mixed;
    bool enabled;
    int max_order;
    float epsilon_floor;
    const int* material_slot;
    const int* scatter_model;
    const float* rough_sigma_h;
    const float* rough_axis_rad;
    const float* f_te;
    const float* f_tm;
    const float* marginal;
    const float* conditional;
    const float* density;
    const int64_t* table_offset;
    const int64_t* marginal_offset;
    const int* dims;
};

#if defined(__CUDACC__)

struct TableView {
    const float* f_te;
    const float* f_tm;
    const float* marginal;
    const float* conditional;
    const float* density;
    int nti;
    int npi;
    int nto;
    int npo;
};

__device__ __forceinline__ TableView table_view(const ScatterTables& tables, int slot) {
    const int64_t offset = tables.table_offset[slot];
    const int64_t marginal_offset = tables.marginal_offset[slot];
    const int* dims = tables.dims + slot * 4;
    return {
        tables.f_te + offset,
        tables.f_tm + offset,
        tables.marginal + marginal_offset,
        tables.conditional + offset,
        tables.density + offset,
        dims[0],
        dims[1],
        dims[2],
        dims[3],
    };
}

using channel::math::cross_values;
using channel::math::dot_values;

__device__ __forceinline__ void unit_values(float* v) {
    channel::math::normalize_values_div(v, 1.0e-12f);
}

// Tangent frame (t1, t2) of a unit normal: t1 is the stable perpendicular
// reference (z unless the normal is nearly vertical, then x) rotated by
// axis_rad about the normal, so it is the roughness principal axis.
__device__ __forceinline__ void local_frame(const float* normal, float axis_rad, float* t1, float* t2) {
    const float reference[3] = {
        fabsf(normal[2]) < 0.9f ? 0.0f : 1.0f, 0.0f, fabsf(normal[2]) < 0.9f ? 1.0f : 0.0f};
    float a[3];
    cross_values(reference, normal, a);
    unit_values(a);
    float b[3];
    cross_values(normal, a, b);
    const float c = cosf(axis_rad);
    const float s = sinf(axis_rad);
    for (int axis = 0; axis < 3; ++axis) {
        t1[axis] = c * a[axis] + s * b[axis];
        t2[axis] = -s * a[axis] + c * b[axis];
    }
}

// Restart offset max(|p|_inf * 1e-6, floor) with floor = max(1e-6 m, diagonal * 1e-6).
__device__ __forceinline__ float scale_epsilon(const float* position, float floor) {
    const float largest = fmaxf(fabsf(position[0]), fmaxf(fabsf(position[1]), fabsf(position[2])));
    return fmaxf(largest * 1.0e-6f, floor);
}

#endif  // defined(__CUDACC__)

}  // namespace channel::bdpt_wavefront
