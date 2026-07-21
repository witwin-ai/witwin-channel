#include "field_transport_ad_common.cuh"

#include <rayd/shared/rf/layer_stack.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr float kTransmissionCosFloor = 1.0e-6f;
constexpr float kDegenerateSinSq = 1.0e-12f;
constexpr int kThinSheetGeometryMode = 0;

struct ZeroLayerSeed {
    __device__ ad::LayerSeed operator()(int) const {
        return {0.0f, 0.0f, 0.0f};
    }
};

struct BasisLayerSeed {
    int slot;
    int parameter;

    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed seed{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (parameter == 0)
                seed.d_thickness = 1.0f;
            else if (parameter == 1)
                seed.d_eps = 1.0f;
            else
                seed.d_sigma = 1.0f;
        }
        return seed;
    }
};

struct OptionalFloatView {
    const float* data;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
};

__device__ __forceinline__ float optional_load1(
    OptionalFloatView view, int64_t index) {
    return view.data == nullptr ? 0.0f : view.data[index * view.stride0];
}

__device__ __forceinline__ float optional_load2(
    OptionalFloatView view, int64_t row, int64_t axis) {
    return view.data == nullptr
        ? 0.0f
        : view.data[row * view.stride0 + axis * view.stride1];
}

__device__ __forceinline__ float optional_load3(
    OptionalFloatView view, int64_t row, int64_t slot, int64_t axis) {
    return view.data == nullptr
        ? 0.0f
        : view.data[
              row * view.stride0 + slot * view.stride1 + axis * view.stride2];
}

struct TangentLayerSeed {
    OptionalFloatView thickness;
    OptionalFloatView eps;
    OptionalFloatView sigma;

    __device__ ad::LayerSeed operator()(int slot) const {
        return {
            optional_load1(thickness, slot),
            optional_load1(eps, slot),
            optional_load1(sigma, slot),
        };
    }
};

struct DualFloat {
    float value;
    float tangent;
};

struct DualVec3 {
    DualFloat x;
    DualFloat y;
    DualFloat z;
};

__device__ __forceinline__ DualFloat dual_add(DualFloat a, DualFloat b) {
    return {a.value + b.value, a.tangent + b.tangent};
}

__device__ __forceinline__ DualFloat dual_mul(DualFloat a, DualFloat b) {
    return {
        a.value * b.value,
        a.tangent * b.value + a.value * b.tangent,
    };
}

__device__ __forceinline__ DualFloat dual_div(DualFloat a, DualFloat b) {
    const float inverse = 1.0f / b.value;
    return {
        a.value * inverse,
        (a.tangent * b.value - a.value * b.tangent) * inverse * inverse,
    };
}

__device__ __forceinline__ DualFloat dual_square(DualFloat a) {
    return {a.value * a.value, 2.0f * a.value * a.tangent};
}

__device__ __forceinline__ DualVec3 dual_cross(DualVec3 a, DualVec3 b) {
    return {
        dual_add(dual_mul(a.y, b.z), {-a.z.value * b.y.value, -a.z.tangent * b.y.value - a.z.value * b.y.tangent}),
        dual_add(dual_mul(a.z, b.x), {-a.x.value * b.z.value, -a.x.tangent * b.z.value - a.x.value * b.z.tangent}),
        dual_add(dual_mul(a.x, b.y), {-a.y.value * b.x.value, -a.y.tangent * b.x.value - a.y.value * b.x.tangent}),
    };
}

__device__ __forceinline__ DualFloat dual_dot(DualVec3 a, DualVec3 b) {
    return dual_add(dual_add(dual_mul(a.x, b.x), dual_mul(a.y, b.y)), dual_mul(a.z, b.z));
}

__device__ __forceinline__ DualVec3 dual_scale(DualVec3 value, DualFloat scale) {
    return {
        dual_mul(value.x, scale),
        dual_mul(value.y, scale),
        dual_mul(value.z, scale),
    };
}

__device__ __forceinline__ DualVec3 load_dual_vec3_basis(
    const float* values,
    int64_t base,
    int axis) {
    return {
        {values[base], axis == 0 ? 1.0f : 0.0f},
        {values[base + 1], axis == 1 ? 1.0f : 0.0f},
        {values[base + 2], axis == 2 ? 1.0f : 0.0f},
    };
}

__device__ __forceinline__ DualVec3 load_dual_direction(
    const float* values,
    OptionalFloatView tangents,
    int64_t row) {
    const int64_t base = row * 3;
    return {
        {values[base], optional_load2(tangents, row, 0)},
        {values[base + 1], optional_load2(tangents, row, 1)},
        {values[base + 2], optional_load2(tangents, row, 2)},
    };
}

__device__ __forceinline__ DualVec3 load_dual_normal(
    const float* values,
    OptionalFloatView tangents,
    int64_t row,
    int64_t slot,
    int64_t hit_capacity) {
    const int64_t base = (row * hit_capacity + slot) * 3;
    return {
        {values[base], optional_load3(tangents, row, slot, 0)},
        {values[base + 1], optional_load3(tangents, row, slot, 1)},
        {values[base + 2], optional_load3(tangents, row, slot, 2)},
    };
}

__device__ __forceinline__ DualVec3 load_fixed_vec3(const float* values, int64_t base) {
    return {
        {values[base], 0.0f},
        {values[base + 1], 0.0f},
        {values[base + 2], 0.0f},
    };
}

__device__ __forceinline__ DualFloat incidence_cosine(DualVec3 direction, DualVec3 normal) {
    const DualFloat dot = dual_dot(direction, normal);
    const float absolute = fabsf(dot.value);
    const float absolute_tangent = dot.value < 0.0f ? -dot.tangent : dot.tangent;
    if (absolute < kTransmissionCosFloor)
        return {kTransmissionCosFloor, 0.0f};
    if (absolute > 1.0f)
        return {1.0f, 0.0f};
    return {absolute, absolute_tangent};
}

struct PolarizationFractions {
    DualFloat te;
    DualFloat tm;
};

__device__ PolarizationFractions polarization_fractions(
    DualVec3 direction,
    DualVec3 normal,
    const float* polarization,
    int64_t polarization_base) {
    DualVec3 s = dual_cross(direction, normal);
    const DualFloat norm_sq = dual_dot(s, s);
    if (norm_sq.value <= kDegenerateSinSq)
        return {{0.5f, 0.0f}, {0.5f, 0.0f}};

    const float inverse_norm = rsqrtf(fmaxf(norm_sq.value, 1.0e-30f));
    const float inverse_norm_tangent =
        -0.5f * inverse_norm * inverse_norm * inverse_norm * norm_sq.tangent;
    s = dual_scale(s, {inverse_norm, inverse_norm_tangent});
    const DualVec3 p = dual_cross(s, direction);
    const DualVec3 pol = load_fixed_vec3(polarization, polarization_base);
    const DualFloat te_power = dual_square(dual_dot(pol, s));
    const DualFloat tm_power = dual_square(dual_dot(pol, p));
    const DualFloat total = dual_add(te_power, tm_power);
    if (total.value <= kDegenerateSinSq)
        return {{0.5f, 0.0f}, {0.5f, 0.0f}};
    return {dual_div(te_power, total), dual_div(tm_power, total)};
}

template <typename SeedFn>
__device__ DualFloat wall_transmittance_dual(
    DualVec3 direction,
    DualVec3 normal,
    const float* polarization,
    int64_t polarization_base,
    const em::LayerView& layers,
    float frequency_hz,
    float tangent_frequency,
    SeedFn&& seed) {
    const DualFloat cosine = incidence_cosine(direction, normal);
    const PolarizationFractions fractions =
        polarization_fractions(direction, normal, polarization, polarization_base);
    const ad::DualStackRT te = ad::stack_rt_dual(
        cosine.value,
        layers,
        frequency_hz,
        cosine.tangent,
        tangent_frequency,
        em::kPolTE,
        seed);
    const ad::DualStackRT tm = ad::stack_rt_dual(
        cosine.value,
        layers,
        frequency_hz,
        cosine.tangent,
        tangent_frequency,
        em::kPolTM,
        seed);
    return {
        fractions.te.value * te.cap_t.v + fractions.tm.value * tm.cap_t.v,
        fractions.te.tangent * te.cap_t.v + fractions.te.value * te.cap_t.d +
            fractions.tm.tangent * tm.cap_t.v + fractions.tm.value * tm.cap_t.d,
    };
}

__device__ bool row_contract_valid(
    const bool* valid,
    const int* num_hits,
    int64_t row,
    int64_t hit_capacity,
    int* failure_state,
    int failure_bit) {
    const int count = num_hits[row];
    if (count < 0 || static_cast<int64_t>(count) > hit_capacity) {
        atomicOr(failure_state, failure_bit);
        return false;
    }
    const int64_t base = row * hit_capacity;
    for (int64_t slot = 0; slot < hit_capacity; ++slot) {
        if (valid[base + slot] != (slot < count)) {
            atomicOr(failure_state, failure_bit);
            return false;
        }
    }
    return true;
}

// >=0: material row; -1: ordinary non-penetrable material; -2: contract error.
__device__ int row_material(
    const int* primitive_id,
    int64_t primitive_slot,
    const int* face_material_id,
    int64_t face_count,
    const int* geometry_mode_id,
    const int* layer_offset,
    const int* layer_count,
    int64_t material_count,
    int64_t layer_total,
    int* failure_state,
    int failure_bit) {
    const int primitive = primitive_id[primitive_slot];
    if (primitive < 0 || static_cast<int64_t>(primitive) >= face_count) {
        atomicOr(failure_state, failure_bit);
        return -2;
    }
    const int material = face_material_id[primitive];
    if (material < 0)
        return -1;
    if (static_cast<int64_t>(material) >= material_count) {
        atomicOr(failure_state, failure_bit);
        return -2;
    }
    if (geometry_mode_id[material] != kThinSheetGeometryMode)
        return -1;
    const int offset = layer_offset[material];
    const int count = layer_count[material];
    if (offset < 0 || count <= 0 || static_cast<int64_t>(offset) > layer_total ||
        static_cast<int64_t>(count) > layer_total - static_cast<int64_t>(offset)) {
        atomicOr(failure_state, failure_bit);
        return -2;
    }
    return material;
}

// Validate every device-selected slot before any continuous payload is read.
// Ordinary blocking materials do not short-circuit later discrete contract
// checks; a malformed later slot must still poison the complete transaction.
__device__ int preflight_row_materials(
    const int* primitive_id,
    int64_t row_base,
    int count,
    const int* face_material_id,
    int64_t face_count,
    const int* geometry_mode_id,
    const int* layer_offset,
    const int* layer_count,
    int64_t material_count,
    int64_t layer_total,
    int* failure_state,
    int failure_bit) {
    bool all_eligible = true;
    for (int slot = 0; slot < count; ++slot) {
        const int material = row_material(
            primitive_id,
            row_base + slot,
            face_material_id,
            face_count,
            geometry_mode_id,
            layer_offset,
            layer_count,
            material_count,
            layer_total,
            failure_state,
            failure_bit);
        if (material == -2)
            return -2;
        if (material == -1)
            all_eligible = false;
    }
    return all_eligible ? 0 : -1;
}

#define CN_MC_WALL_PRODUCT_COMMON_KERNEL_PARAMS                               \
    int64_t row_count, int64_t hit_capacity, const bool* valid,               \
        const int* num_hits, const bool* reached_target,                      \
        const float* direction, const float* normal,                          \
        const int* primitive_id, const int* face_material_id,                 \
        int64_t face_count, const int* geometry_mode_id,                      \
        const int* layer_offset, const int* layer_count,                      \
        const float* layer_thickness, const float* layer_eps,                 \
        const float* layer_sigma, const float* layer_mu,                      \
        int64_t material_count, int64_t layer_total,                          \
        const float* polarization, const float* base_power,                   \
        float frequency_hz, int* failure_state, int failure_bit

__global__ void wall_product_kernel(
    CN_MC_WALL_PRODUCT_COMMON_KERNEL_PARAMS,
    float* scaled_power,
    float* transmittance,
    int* wall_count,
    bool* penetrated) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (*failure_state != 0 ||
            !row_contract_valid(valid, num_hits, row, hit_capacity, failure_state, failure_bit))
            continue;
        const int count = num_hits[row];
        wall_count[row] = count;
        if (!reached_target[row])
            continue;
        if (count == 0) {
            transmittance[row] = 1.0f;
            continue;
        }

        const int64_t row_base = row * hit_capacity;
        const int preflight = preflight_row_materials(
            primitive_id,
            row_base,
            count,
            face_material_id,
            face_count,
            geometry_mode_id,
            layer_offset,
            layer_count,
            material_count,
            layer_total,
            failure_state,
            failure_bit);
        if (preflight != 0 || *failure_state != 0)
            continue;
        const int64_t direction_base = row * 3;
        const DualVec3 ray_direction = load_fixed_vec3(direction, direction_base);
        float product = 1.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int64_t primitive_slot = row_base + slot;
            const int material = row_material(
                primitive_id,
                primitive_slot,
                face_material_id,
                face_count,
                geometry_mode_id,
                layer_offset,
                layer_count,
                material_count,
                layer_total,
                failure_state,
                failure_bit);
            em::LayerView layers{
                layer_offset,
                layer_count,
                layer_thickness,
                layer_eps,
                layer_sigma,
                layer_mu,
                material,
            };
            const int64_t normal_base = primitive_slot * 3;
            const DualFloat value = wall_transmittance_dual(
                ray_direction,
                load_fixed_vec3(normal, normal_base),
                polarization,
                direction_base,
                layers,
                frequency_hz,
                0.0f,
                ZeroLayerSeed{});
            product = product * value.value;
        }
        if (*failure_state != 0)
            continue;
        transmittance[row] = product;
        scaled_power[row] = base_power[row] * product;
        penetrated[row] = true;
    }
}

__global__ void wall_product_backward_kernel(
    CN_MC_WALL_PRODUCT_COMMON_KERNEL_PARAMS,
    OptionalFloatView grad_scaled,
    OptionalFloatView grad_transmittance,
    float* grad_direction,
    float* grad_normal,
    float* grad_base_power) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (*failure_state != 0 ||
            !row_contract_valid(valid, num_hits, row, hit_capacity, failure_state, failure_bit))
            continue;
        const int count = num_hits[row];
        if (!reached_target[row] || count == 0)
            continue;

        const int64_t row_base = row * hit_capacity;
        const int preflight = preflight_row_materials(
            primitive_id,
            row_base,
            count,
            face_material_id,
            face_count,
            geometry_mode_id,
            layer_offset,
            layer_count,
            material_count,
            layer_total,
            failure_state,
            failure_bit);
        if (preflight != 0 || *failure_state != 0)
            continue;
        const int64_t direction_base = row * 3;
        const DualVec3 ray_direction = load_fixed_vec3(direction, direction_base);
        float product = 1.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int64_t primitive_slot = row_base + slot;
            const int material = row_material(
                primitive_id,
                primitive_slot,
                face_material_id,
                face_count,
                geometry_mode_id,
                layer_offset,
                layer_count,
                material_count,
                layer_total,
                failure_state,
                failure_bit);
            em::LayerView layers{
                layer_offset, layer_count, layer_thickness, layer_eps,
                layer_sigma, layer_mu, material};
            const float wall_value = wall_transmittance_dual(
                ray_direction,
                load_fixed_vec3(normal, primitive_slot * 3),
                polarization,
                direction_base,
                layers,
                frequency_hz,
                0.0f,
                ZeroLayerSeed{}).value;
            product = product * wall_value;
        }
        if (*failure_state != 0)
            continue;

        const float g_scaled = optional_load1(grad_scaled, row);
        const float g_trans =
            optional_load1(grad_transmittance, row) +
            g_scaled * base_power[row];
        grad_base_power[row] = g_scaled * product;

        float direction_gradient[3] = {0.0f, 0.0f, 0.0f};
        for (int slot = 0; slot < count; ++slot) {
            float product_except = 1.0f;
            for (int other = 0; other < count; ++other) {
                if (other == slot)
                    continue;
                const int64_t other_primitive_slot = row_base + other;
                const int other_material = row_material(
                    primitive_id,
                    other_primitive_slot,
                    face_material_id,
                    face_count,
                    geometry_mode_id,
                    layer_offset,
                    layer_count,
                    material_count,
                    layer_total,
                    failure_state,
                    failure_bit);
                if (other_material < 0)
                    break;
                em::LayerView other_layers{
                    layer_offset, layer_count, layer_thickness, layer_eps,
                    layer_sigma, layer_mu, other_material};
                const float other_value = wall_transmittance_dual(
                    ray_direction,
                    load_fixed_vec3(normal, other_primitive_slot * 3),
                    polarization,
                    direction_base,
                    other_layers,
                    frequency_hz,
                    0.0f,
                    ZeroLayerSeed{}).value;
                product_except = product_except * other_value;
            }
            if (*failure_state != 0)
                break;
            const float g_wall = g_trans * product_except;
            const int64_t primitive_slot = row_base + slot;
            const int64_t normal_base = primitive_slot * 3;
            const int material = row_material(
                primitive_id,
                primitive_slot,
                face_material_id,
                face_count,
                geometry_mode_id,
                layer_offset,
                layer_count,
                material_count,
                layer_total,
                failure_state,
                failure_bit);
            if (material < 0)
                break;
            em::LayerView layers{
                layer_offset, layer_count, layer_thickness, layer_eps,
                layer_sigma, layer_mu, material};

            for (int axis = 0; axis < 3; ++axis) {
                const DualFloat derivative = wall_transmittance_dual(
                    load_dual_vec3_basis(direction, direction_base, axis),
                    load_fixed_vec3(normal, normal_base),
                    polarization,
                    direction_base,
                    layers,
                    frequency_hz,
                    0.0f,
                    ZeroLayerSeed{});
                direction_gradient[axis] += g_wall * derivative.tangent;

                const DualFloat normal_derivative = wall_transmittance_dual(
                    ray_direction,
                    load_dual_vec3_basis(normal, normal_base, axis),
                    polarization,
                    direction_base,
                    layers,
                    frequency_hz,
                    0.0f,
                    ZeroLayerSeed{});
                grad_normal[normal_base + axis] = g_wall * normal_derivative.tangent;
            }

        }
        grad_direction[direction_base] = direction_gradient[0];
        grad_direction[direction_base + 1] = direction_gradient[1];
        grad_direction[direction_base + 2] = direction_gradient[2];
    }
}

// One thread owns one shared layer derivative; one final thread owns frequency.
// Each owner adds in ascending row then slot order, so VJP bits are independent
// of CUDA block scheduling.
__global__ void wall_product_shared_backward_kernel(
    CN_MC_WALL_PRODUCT_COMMON_KERNEL_PARAMS,
    OptionalFloatView grad_scaled,
    OptionalFloatView grad_transmittance,
    float* grad_layer_thickness,
    float* grad_layer_eps,
    float* grad_layer_sigma,
    float* grad_frequency) {
    const int64_t output_count = layer_total * 3 + 1;
    for (int64_t output_index =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output_index < output_count;
         output_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const bool frequency_owner = output_index == layer_total * 3;
        const int64_t owned_layer = frequency_owner ? -1 : output_index / 3;
        const int owned_parameter = frequency_owner ? -1 : output_index % 3;
        float gradient = 0.0f;

        for (int64_t row = 0; row < row_count; ++row) {
            if (*failure_state != 0)
                return;
            if (!row_contract_valid(
                    valid, num_hits, row, hit_capacity, failure_state, failure_bit))
                return;
            const int count = num_hits[row];
            if (!reached_target[row] || count == 0)
                continue;
            const int64_t row_base = row * hit_capacity;
            const int preflight = preflight_row_materials(
                primitive_id,
                row_base,
                count,
                face_material_id,
                face_count,
                geometry_mode_id,
                layer_offset,
                layer_count,
                material_count,
                layer_total,
                failure_state,
                failure_bit);
            if (preflight == -2)
                return;
            if (preflight == -1)
                continue;
            if (*failure_state != 0)
                return;
            const int64_t direction_base = row * 3;
            const DualVec3 ray_direction = load_fixed_vec3(direction, direction_base);

            const float g_scaled = optional_load1(grad_scaled, row);
            const float g_trans =
                optional_load1(grad_transmittance, row) +
                g_scaled * base_power[row];
            for (int slot = 0; slot < count; ++slot) {
                const int64_t primitive_slot = row_base + slot;
                const int material = row_material(
                    primitive_id,
                    primitive_slot,
                    face_material_id,
                    face_count,
                    geometry_mode_id,
                    layer_offset,
                    layer_count,
                    material_count,
                    layer_total,
                    failure_state,
                    failure_bit);
                if (material < 0)
                    return;
                const int first = layer_offset[material];
                const int material_layers = layer_count[material];
                if (!frequency_owner &&
                    (owned_layer < static_cast<int64_t>(first) ||
                     owned_layer >= static_cast<int64_t>(first) +
                         static_cast<int64_t>(material_layers)))
                    continue;

                float product_except = 1.0f;
                for (int other = 0; other < count; ++other) {
                    if (other == slot)
                        continue;
                    const int64_t other_primitive_slot = row_base + other;
                    const int other_material = row_material(
                        primitive_id,
                        other_primitive_slot,
                        face_material_id,
                        face_count,
                        geometry_mode_id,
                        layer_offset,
                        layer_count,
                        material_count,
                        layer_total,
                        failure_state,
                        failure_bit);
                    if (other_material < 0)
                        return;
                    em::LayerView other_layers{
                        layer_offset, layer_count, layer_thickness, layer_eps,
                        layer_sigma, layer_mu, other_material};
                    const float other_value = wall_transmittance_dual(
                        ray_direction,
                        load_fixed_vec3(normal, other_primitive_slot * 3),
                        polarization,
                        direction_base,
                        other_layers,
                        frequency_hz,
                        0.0f,
                        ZeroLayerSeed{}).value;
                    product_except = product_except * other_value;
                }
                em::LayerView layers{
                    layer_offset, layer_count, layer_thickness, layer_eps,
                    layer_sigma, layer_mu, material};
                const DualFloat derivative = frequency_owner
                    ? wall_transmittance_dual(
                          ray_direction,
                          load_fixed_vec3(normal, primitive_slot * 3),
                          polarization,
                          direction_base,
                          layers,
                          frequency_hz,
                          1.0f,
                          ZeroLayerSeed{})
                    : wall_transmittance_dual(
                          ray_direction,
                          load_fixed_vec3(normal, primitive_slot * 3),
                          polarization,
                          direction_base,
                          layers,
                          frequency_hz,
                          0.0f,
                          BasisLayerSeed{
                              static_cast<int>(owned_layer), owned_parameter});
                gradient = gradient + g_trans * product_except * derivative.tangent;
            }
        }
        if (frequency_owner)
            grad_frequency[0] = gradient;
        else if (owned_parameter == 0)
            grad_layer_thickness[owned_layer] = gradient;
        else if (owned_parameter == 1)
            grad_layer_eps[owned_layer] = gradient;
        else
            grad_layer_sigma[owned_layer] = gradient;
    }
}

__global__ void wall_product_jvp_kernel(
    CN_MC_WALL_PRODUCT_COMMON_KERNEL_PARAMS,
    OptionalFloatView tangent_direction,
    OptionalFloatView tangent_normal,
    OptionalFloatView tangent_layer_thickness,
    OptionalFloatView tangent_layer_eps,
    OptionalFloatView tangent_layer_sigma,
    OptionalFloatView tangent_base_power,
    float tangent_frequency,
    float* tangent_scaled_power,
    float* tangent_transmittance) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (*failure_state != 0 ||
            !row_contract_valid(valid, num_hits, row, hit_capacity, failure_state, failure_bit))
            continue;
        const int count = num_hits[row];
        if (!reached_target[row] || count == 0)
            continue;

        const int64_t row_base = row * hit_capacity;
        const int preflight = preflight_row_materials(
            primitive_id,
            row_base,
            count,
            face_material_id,
            face_count,
            geometry_mode_id,
            layer_offset,
            layer_count,
            material_count,
            layer_total,
            failure_state,
            failure_bit);
        if (preflight != 0 || *failure_state != 0)
            continue;
        const int64_t direction_base = row * 3;
        const DualVec3 ray_direction =
            load_dual_direction(direction, tangent_direction, row);
        DualFloat product{1.0f, 0.0f};
        for (int slot = 0; slot < count; ++slot) {
            const int64_t primitive_slot = row_base + slot;
            const int material = row_material(
                primitive_id,
                primitive_slot,
                face_material_id,
                face_count,
                geometry_mode_id,
                layer_offset,
                layer_count,
                material_count,
                layer_total,
                failure_state,
                failure_bit);
            em::LayerView layers{
                layer_offset, layer_count, layer_thickness, layer_eps,
                layer_sigma, layer_mu, material};
            const DualFloat wall = wall_transmittance_dual(
                ray_direction,
                load_dual_normal(normal, tangent_normal, row, slot, hit_capacity),
                polarization,
                direction_base,
                layers,
                frequency_hz,
                tangent_frequency,
                TangentLayerSeed{
                    tangent_layer_thickness,
                    tangent_layer_eps,
                    tangent_layer_sigma});
            product = dual_mul(product, wall);
        }
        if (*failure_state != 0)
            continue;
        const float tangent_base = optional_load1(tangent_base_power, row);
        tangent_transmittance[row] = product.tangent;
        tangent_scaled_power[row] =
            tangent_base * product.value + base_power[row] * product.tangent;
    }
}

#undef CN_MC_WALL_PRODUCT_COMMON_KERNEL_PARAMS

__global__ void sanitize_primal_kernel(
    const int* failure_state,
    int64_t row_count,
    float* scaled_power,
    float* transmittance,
    int* wall_count,
    bool* penetrated) {
    if (*failure_state == 0)
        return;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        scaled_power[row] = 0.0f;
        transmittance[row] = 0.0f;
        wall_count[row] = 0;
        penetrated[row] = false;
    }
}

__global__ void sanitize_backward_kernel(
    const int* failure_state,
    int64_t row_count,
    int64_t hit_capacity,
    int64_t layer_total,
    float* grad_direction,
    float* grad_normal,
    float* grad_layer_thickness,
    float* grad_layer_eps,
    float* grad_layer_sigma,
    float* grad_base_power,
    float* grad_frequency) {
    if (*failure_state == 0)
        return;
    const int64_t direction_total = row_count * 3;
    const int64_t normal_total = row_count * hit_capacity * 3;
    const int64_t vector_total = direction_total > normal_total ? direction_total : normal_total;
    const int64_t scalar_total = layer_total > row_count ? layer_total : row_count;
    const int64_t total = vector_total > scalar_total ? vector_total : scalar_total;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (index < direction_total)
            grad_direction[index] = 0.0f;
        if (index < normal_total)
            grad_normal[index] = 0.0f;
        if (index < layer_total) {
            grad_layer_thickness[index] = 0.0f;
            grad_layer_eps[index] = 0.0f;
            grad_layer_sigma[index] = 0.0f;
        }
        if (index < row_count)
            grad_base_power[index] = 0.0f;
        if (index == 0)
            grad_frequency[0] = 0.0f;
    }
}

__global__ void sanitize_jvp_kernel(
    const int* failure_state,
    int64_t row_count,
    float* tangent_scaled_power,
    float* tangent_transmittance) {
    if (*failure_state == 0)
        return;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < row_count;
         row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        tangent_scaled_power[row] = 0.0f;
        tangent_transmittance[row] = 0.0f;
    }
}

void check_tensor(
    const at::Tensor& tensor,
    const char* name,
    c10::ScalarType dtype,
    int64_t rank,
    const at::Tensor* reference = nullptr) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == rank, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    if (reference != nullptr)
        TORCH_CHECK(tensor.get_device() == reference->get_device(), name, " must share the input device");
}

struct WallProductInputs {
    int64_t rows;
    int64_t hit_capacity;
    int64_t materials;
    int64_t layers;
};

WallProductInputs check_inputs(
    const at::Tensor& valid,
    const at::Tensor& num_hits,
    const at::Tensor& reached_target,
    const at::Tensor& direction,
    const at::Tensor& normal,
    const at::Tensor& primitive_id,
    const at::Tensor& face_material_id,
    const at::Tensor& geometry_mode_id,
    const at::Tensor& layer_offset,
    const at::Tensor& layer_count,
    const at::Tensor& layer_thickness,
    const at::Tensor& layer_eps,
    const at::Tensor& layer_sigma,
    const at::Tensor& layer_mu,
    const at::Tensor& polarization,
    const at::Tensor& base_power,
    const at::Tensor& failure_state,
    int64_t failure_bit,
    double frequency_hz) {
    check_tensor(valid, "valid", at::kBool, 2);
    const int64_t rows = valid.size(0);
    const int64_t capacity = valid.size(1);
    check_tensor(num_hits, "num_hits", at::kInt, 1, &valid);
    check_tensor(reached_target, "reached_target", at::kBool, 1, &valid);
    check_tensor(direction, "direction", at::kFloat, 2, &valid);
    check_tensor(normal, "normal", at::kFloat, 3, &valid);
    check_tensor(primitive_id, "global_primitive_id", at::kInt, 2, &valid);
    check_tensor(face_material_id, "face_material_id", at::kInt, 1, &valid);
    check_tensor(geometry_mode_id, "geometry_mode_id", at::kInt, 1, &valid);
    check_tensor(layer_offset, "layer_offset", at::kInt, 1, &valid);
    check_tensor(layer_count, "layer_count", at::kInt, 1, &valid);
    check_tensor(layer_thickness, "layer_thickness_m", at::kFloat, 1, &valid);
    check_tensor(layer_eps, "layer_eps_r", at::kFloat, 1, &valid);
    check_tensor(layer_sigma, "layer_sigma_e", at::kFloat, 1, &valid);
    check_tensor(layer_mu, "layer_mu_r", at::kFloat, 1, &valid);
    check_tensor(polarization, "pair_polarization", at::kFloat, 2, &valid);
    check_tensor(base_power, "base_power", at::kFloat, 1, &valid);
    check_tensor(failure_state, "capacity_failure_state", at::kInt, 1, &valid);
    TORCH_CHECK(num_hits.size(0) == rows, "num_hits must match valid rows");
    TORCH_CHECK(reached_target.size(0) == rows, "reached_target must match valid rows");
    TORCH_CHECK(direction.sizes() == at::IntArrayRef({rows, 3}), "direction must have shape (N, 3)");
    TORCH_CHECK(normal.sizes() == at::IntArrayRef({rows, capacity, 3}), "normal must have shape (N, D, 3)");
    TORCH_CHECK(primitive_id.sizes() == valid.sizes(), "global_primitive_id must match valid");
    TORCH_CHECK(polarization.sizes() == at::IntArrayRef({rows, 3}), "pair_polarization must have shape (N, 3)");
    TORCH_CHECK(base_power.size(0) == rows, "base_power must match valid rows");
    const int64_t materials = layer_offset.size(0);
    TORCH_CHECK(layer_count.size(0) == materials, "layer_count must match layer_offset");
    TORCH_CHECK(geometry_mode_id.size(0) == materials, "geometry_mode_id must match material rows");
    const int64_t layers = layer_thickness.size(0);
    TORCH_CHECK(layer_eps.size(0) == layers && layer_sigma.size(0) == layers && layer_mu.size(0) == layers,
                "layer property tensors must have one shared length");
    TORCH_CHECK(failure_state.numel() == 1, "capacity_failure_state must have shape (1,)");
    TORCH_CHECK(failure_bit > 0 && failure_bit <= std::numeric_limits<int>::max() &&
                    (failure_bit & (failure_bit - 1)) == 0,
                "failure_bit must be one positive int32 bit");
    TORCH_CHECK(std::isfinite(frequency_hz) && frequency_hz > 0.0,
                "frequency_hz must be finite and positive");
    return {rows, capacity, materials, layers};
}

void zero_tensor(const at::Tensor& tensor, cudaStream_t stream) {
    if (tensor.numel() != 0)
        C10_CUDA_CHECK(cudaMemsetAsync(
            tensor.data_ptr(), 0,
            static_cast<size_t>(tensor.numel()) * tensor.element_size(), stream));
}

OptionalFloatView optional_tensor_view(
    pybind11::object value,
    at::Tensor& storage,
    const char* name,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (value.is_none())
        return {nullptr, 0, 0, 0};
    storage = value.cast<at::Tensor>();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == at::kFloat, name, " has the wrong dtype");
    TORCH_CHECK(storage.dim() == sizes.size(), name, " has the wrong rank");
    TORCH_CHECK(storage.get_device() == reference.get_device(), name, " must share the input device");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    return {
        storage.data_ptr<float>(),
        storage.dim() > 0 ? storage.stride(0) : 0,
        storage.dim() > 1 ? storage.stride(1) : 0,
        storage.dim() > 2 ? storage.stride(2) : 0,
    };
}

}  // namespace

pybind11::dict cn_mc_transmission_wall_product(
    at::Tensor valid,
    at::Tensor num_hits,
    at::Tensor reached_target,
    at::Tensor direction,
    at::Tensor normal,
    at::Tensor global_primitive_id,
    at::Tensor face_material_id,
    at::Tensor geometry_mode_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    at::Tensor pair_polarization,
    at::Tensor base_power,
    double frequency_hz,
    at::Tensor capacity_failure_state,
    int64_t failure_bit) {
    const WallProductInputs shape = check_inputs(
        valid, num_hits, reached_target, direction, normal, global_primitive_id,
        face_material_id, geometry_mode_id, layer_offset, layer_count,
        layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r,
        pair_polarization, base_power, capacity_failure_state, failure_bit,
        frequency_hz);
    auto scaled_power = at::empty({shape.rows}, base_power.options());
    auto transmittance = at::empty({shape.rows}, base_power.options());
    auto wall_count = at::empty({shape.rows}, num_hits.options());
    auto penetrated = at::empty({shape.rows}, valid.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    zero_tensor(scaled_power, stream);
    zero_tensor(transmittance, stream);
    zero_tensor(wall_count, stream);
    zero_tensor(penetrated, stream);
    if (shape.rows > 0) {
        const int blocks = launch_blocks(shape.rows);
        wall_product_kernel<<<blocks, kBlockSize, 0, stream>>>(
            shape.rows, shape.hit_capacity, valid.data_ptr<bool>(), num_hits.data_ptr<int>(),
            reached_target.data_ptr<bool>(), direction.data_ptr<float>(), normal.data_ptr<float>(),
            global_primitive_id.data_ptr<int>(), face_material_id.data_ptr<int>(), face_material_id.numel(),
            geometry_mode_id.data_ptr<int>(), layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(), layer_sigma_e.data_ptr<float>(),
            layer_mu_r.data_ptr<float>(), shape.materials, shape.layers,
            pair_polarization.data_ptr<float>(),
            base_power.data_ptr<float>(), static_cast<float>(frequency_hz),
            capacity_failure_state.data_ptr<int>(), static_cast<int>(failure_bit),
            scaled_power.data_ptr<float>(), transmittance.data_ptr<float>(), wall_count.data_ptr<int>(),
            penetrated.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        sanitize_primal_kernel<<<blocks, kBlockSize, 0, stream>>>(
            capacity_failure_state.data_ptr<int>(), shape.rows, scaled_power.data_ptr<float>(),
            transmittance.data_ptr<float>(), wall_count.data_ptr<int>(), penetrated.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["scaled_power"] = scaled_power;
    out["transmittance"] = transmittance;
    out["wall_count"] = wall_count;
    out["penetrated"] = penetrated;
    return out;
}

pybind11::tuple cn_mc_transmission_wall_product_backward(
    at::Tensor valid,
    at::Tensor num_hits,
    at::Tensor reached_target,
    at::Tensor direction,
    at::Tensor normal,
    at::Tensor global_primitive_id,
    at::Tensor face_material_id,
    at::Tensor geometry_mode_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    at::Tensor pair_polarization,
    at::Tensor base_power,
    double frequency_hz,
    at::Tensor capacity_failure_state,
    int64_t failure_bit,
    pybind11::object grad_scaled_power,
    pybind11::object grad_transmittance) {
    const WallProductInputs shape = check_inputs(
        valid, num_hits, reached_target, direction, normal, global_primitive_id,
        face_material_id, geometry_mode_id, layer_offset, layer_count,
        layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r,
        pair_polarization, base_power, capacity_failure_state, failure_bit,
        frequency_hz);
    at::Tensor grad_scaled_storage;
    at::Tensor grad_trans_storage;
    const OptionalFloatView grad_scaled = optional_tensor_view(
        grad_scaled_power, grad_scaled_storage, "grad_scaled_power", {shape.rows}, valid);
    const OptionalFloatView grad_trans = optional_tensor_view(
        grad_transmittance, grad_trans_storage, "grad_transmittance", {shape.rows}, valid);
    auto grad_direction = at::empty_like(direction);
    auto grad_normal = at::empty_like(normal);
    auto grad_layer_thickness = at::empty_like(layer_thickness_m);
    auto grad_layer_eps = at::empty_like(layer_eps_r);
    auto grad_layer_sigma = at::empty_like(layer_sigma_e);
    auto grad_base_power = at::empty_like(base_power);
    auto grad_frequency = at::empty({1}, base_power.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    for (const auto& tensor : {grad_direction, grad_normal, grad_layer_thickness,
                               grad_layer_eps, grad_layer_sigma, grad_base_power,
                               grad_frequency})
        zero_tensor(tensor, stream);
    if (shape.rows > 0) {
        const int blocks = launch_blocks(shape.rows);
        wall_product_backward_kernel<<<blocks, kBlockSize, 0, stream>>>(
            shape.rows, shape.hit_capacity, valid.data_ptr<bool>(), num_hits.data_ptr<int>(),
            reached_target.data_ptr<bool>(), direction.data_ptr<float>(), normal.data_ptr<float>(),
            global_primitive_id.data_ptr<int>(), face_material_id.data_ptr<int>(), face_material_id.numel(),
            geometry_mode_id.data_ptr<int>(), layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(), layer_sigma_e.data_ptr<float>(),
            layer_mu_r.data_ptr<float>(), shape.materials, shape.layers,
            pair_polarization.data_ptr<float>(),
            base_power.data_ptr<float>(), static_cast<float>(frequency_hz),
            capacity_failure_state.data_ptr<int>(), static_cast<int>(failure_bit), grad_scaled, grad_trans,
            grad_direction.data_ptr<float>(), grad_normal.data_ptr<float>(),
            grad_base_power.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const int64_t shared_outputs = shape.layers * 3 + 1;
        wall_product_shared_backward_kernel<<<
            launch_blocks(shared_outputs), kBlockSize, 0, stream>>>(
            shape.rows, shape.hit_capacity, valid.data_ptr<bool>(), num_hits.data_ptr<int>(),
            reached_target.data_ptr<bool>(), direction.data_ptr<float>(), normal.data_ptr<float>(),
            global_primitive_id.data_ptr<int>(), face_material_id.data_ptr<int>(), face_material_id.numel(),
            geometry_mode_id.data_ptr<int>(), layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(), layer_sigma_e.data_ptr<float>(),
            layer_mu_r.data_ptr<float>(), shape.materials, shape.layers,
            pair_polarization.data_ptr<float>(), base_power.data_ptr<float>(),
            static_cast<float>(frequency_hz), capacity_failure_state.data_ptr<int>(),
            static_cast<int>(failure_bit), grad_scaled, grad_trans,
            grad_layer_thickness.data_ptr<float>(), grad_layer_eps.data_ptr<float>(),
            grad_layer_sigma.data_ptr<float>(), grad_frequency.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const int64_t total = std::max(
            std::max(shape.rows * 3, shape.rows * shape.hit_capacity * 3),
            std::max(shape.layers, shape.rows));
        sanitize_backward_kernel<<<launch_blocks(std::max<int64_t>(total, 1)), kBlockSize, 0, stream>>>(
            capacity_failure_state.data_ptr<int>(), shape.rows, shape.hit_capacity, shape.layers,
            grad_direction.data_ptr<float>(), grad_normal.data_ptr<float>(),
            grad_layer_thickness.data_ptr<float>(), grad_layer_eps.data_ptr<float>(),
            grad_layer_sigma.data_ptr<float>(), grad_base_power.data_ptr<float>(),
            grad_frequency.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::tuple out(7);
    out[0] = grad_direction;
    out[1] = grad_normal;
    out[2] = grad_layer_thickness;
    out[3] = grad_layer_eps;
    out[4] = grad_layer_sigma;
    out[5] = grad_base_power;
    out[6] = grad_frequency;
    return out;
}

pybind11::dict cn_mc_transmission_wall_product_jvp(
    at::Tensor valid,
    at::Tensor num_hits,
    at::Tensor reached_target,
    at::Tensor direction,
    at::Tensor normal,
    at::Tensor global_primitive_id,
    at::Tensor face_material_id,
    at::Tensor geometry_mode_id,
    at::Tensor layer_offset,
    at::Tensor layer_count,
    at::Tensor layer_thickness_m,
    at::Tensor layer_eps_r,
    at::Tensor layer_sigma_e,
    at::Tensor layer_mu_r,
    at::Tensor pair_polarization,
    at::Tensor base_power,
    double frequency_hz,
    at::Tensor capacity_failure_state,
    int64_t failure_bit,
    pybind11::object tangent_direction_value,
    pybind11::object tangent_normal_value,
    pybind11::object tangent_layer_thickness_value,
    pybind11::object tangent_layer_eps_value,
    pybind11::object tangent_layer_sigma_value,
    pybind11::object tangent_base_power_value,
    double tangent_frequency) {
    const WallProductInputs shape = check_inputs(
        valid, num_hits, reached_target, direction, normal, global_primitive_id,
        face_material_id, geometry_mode_id, layer_offset, layer_count,
        layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r,
        pair_polarization, base_power, capacity_failure_state, failure_bit,
        frequency_hz);
    at::Tensor tangent_direction_storage, tangent_normal_storage;
    at::Tensor tangent_layer_thickness_storage, tangent_layer_eps_storage;
    at::Tensor tangent_layer_sigma_storage, tangent_base_power_storage;
    const OptionalFloatView tangent_direction = optional_tensor_view(
        tangent_direction_value, tangent_direction_storage, "tangent_direction", direction.sizes(), valid);
    const OptionalFloatView tangent_normal = optional_tensor_view(
        tangent_normal_value, tangent_normal_storage, "tangent_normal", normal.sizes(), valid);
    const OptionalFloatView tangent_layer_thickness = optional_tensor_view(
        tangent_layer_thickness_value, tangent_layer_thickness_storage,
        "tangent_layer_thickness_m", layer_thickness_m.sizes(), valid);
    const OptionalFloatView tangent_layer_eps = optional_tensor_view(
        tangent_layer_eps_value, tangent_layer_eps_storage, "tangent_layer_eps_r", layer_eps_r.sizes(), valid);
    const OptionalFloatView tangent_layer_sigma = optional_tensor_view(
        tangent_layer_sigma_value, tangent_layer_sigma_storage, "tangent_layer_sigma_e", layer_sigma_e.sizes(), valid);
    const OptionalFloatView tangent_base_power = optional_tensor_view(
        tangent_base_power_value, tangent_base_power_storage, "tangent_base_power", base_power.sizes(), valid);
    auto tangent_scaled_power = at::empty_like(base_power);
    auto tangent_transmittance = at::empty_like(base_power);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(valid.get_device()).stream();
    zero_tensor(tangent_scaled_power, stream);
    zero_tensor(tangent_transmittance, stream);
    if (shape.rows > 0) {
        const int blocks = launch_blocks(shape.rows);
        wall_product_jvp_kernel<<<blocks, kBlockSize, 0, stream>>>(
            shape.rows, shape.hit_capacity, valid.data_ptr<bool>(), num_hits.data_ptr<int>(),
            reached_target.data_ptr<bool>(), direction.data_ptr<float>(), normal.data_ptr<float>(),
            global_primitive_id.data_ptr<int>(), face_material_id.data_ptr<int>(), face_material_id.numel(),
            geometry_mode_id.data_ptr<int>(), layer_offset.data_ptr<int>(), layer_count.data_ptr<int>(),
            layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(), layer_sigma_e.data_ptr<float>(),
            layer_mu_r.data_ptr<float>(), shape.materials, shape.layers,
            pair_polarization.data_ptr<float>(),
            base_power.data_ptr<float>(), static_cast<float>(frequency_hz),
            capacity_failure_state.data_ptr<int>(), static_cast<int>(failure_bit),
            tangent_direction, tangent_normal, tangent_layer_thickness, tangent_layer_eps,
            tangent_layer_sigma, tangent_base_power, static_cast<float>(tangent_frequency),
            tangent_scaled_power.data_ptr<float>(), tangent_transmittance.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        sanitize_jvp_kernel<<<blocks, kBlockSize, 0, stream>>>(
            capacity_failure_state.data_ptr<int>(), shape.rows,
            tangent_scaled_power.data_ptr<float>(), tangent_transmittance.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["scaled_power"] = tangent_scaled_power;
    out["transmittance"] = tangent_transmittance;
    return out;
}
