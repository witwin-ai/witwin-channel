// Copyright Xingyu Chen.
// Shares the small vector and complex operations used by Channel CUDA kernels.

#pragma once

#include <cuda_runtime.h>
#include <rayd/utd.h>

#include <cmath>
#include <cstdint>

namespace channel::math {

using Vec3 = float3;

struct Complex {
    float r;
    float i;
};

struct Complex3 {
    Complex x;
    Complex y;
    Complex z;
};

__device__ __forceinline__ Vec3 vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ __forceinline__ Vec3 load_vec3(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return vec3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ rayd::shared::diffraction::float3a load_field_vec3(
    const float* values,
    int64_t index) {
    const int64_t base = index * 3;
    return rayd::shared::diffraction::make_f3(
        values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ Vec3 load_sequence_vec3(
    const float* values,
    int64_t index,
    int64_t position,
    int64_t depth) {
    const int64_t base = (index * depth + position) * 3;
    return vec3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ void store_vec3(float* values, int64_t index, Vec3 value) {
    const int64_t base = index * 3;
    values[base] = value.x;
    values[base + 1] = value.y;
    values[base + 2] = value.z;
}

__device__ __forceinline__ Vec3 add(Vec3 a, Vec3 b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Vec3 sub(Vec3 a, Vec3 b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3 scale(Vec3 value, float factor) {
    return vec3(value.x * factor, value.y * factor, value.z * factor);
}

__device__ __forceinline__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float dot_rn_xzy(Vec3 a, Vec3 b) {
    const float xz = __fadd_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.z, b.z));
    return __fadd_rn(xz, __fmul_rn(a.y, b.y));
}

__device__ __forceinline__ float dot_values(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ float length_values(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

__device__ __forceinline__ Vec3 cross(Vec3 a, Vec3 b) {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float length(Vec3 value) {
    return sqrtf(fmaxf(dot(value, value), 0.0f));
}

__device__ __forceinline__ float length_rn_xzy(Vec3 value) {
    return sqrtf(dot_rn_xzy(value, value));
}

__device__ __forceinline__ Vec3 normalize_min_length(Vec3 value, float minimum_length) {
    const float inverse = 1.0f / fmaxf(length(value), minimum_length);
    return scale(value, inverse);
}

__device__ __forceinline__ Vec3 load_normalized_min_length(
    const float* values,
    int64_t index,
    float minimum_length) {
    Vec3 value = load_vec3(values, index);
    float value_length = sqrtf(
        value.x * value.x + value.y * value.y + value.z * value.z);
    value_length = fmaxf(value_length, minimum_length);
    return vec3(
        value.x / value_length,
        value.y / value_length,
        value.z / value_length);
}

__device__ __forceinline__ Vec3 normalize_rsqrt_min_squared(
    Vec3 value,
    float minimum_squared_length) {
    const float inverse = rsqrtf(fmaxf(dot(value, value), minimum_squared_length));
    return scale(value, inverse);
}

__device__ __forceinline__ Vec3 normalize_rsqrt_safe(Vec3 value) {
    return normalize_rsqrt_min_squared(value, 1.0e-30f);
}

__device__ __forceinline__ Vec3 normalize_rn_xzy(Vec3 value, float minimum_length) {
    const float denominator = fmaxf(length_rn_xzy(value), minimum_length);
    return vec3(
        __fdiv_rn(value.x, denominator),
        __fdiv_rn(value.y, denominator),
        __fdiv_rn(value.z, denominator));
}

__device__ __forceinline__ Vec3 normalize_or_zero(Vec3 value, float epsilon) {
    const float value_length = length(value);
    return value_length > epsilon
        ? scale(value, 1.0f / value_length)
        : vec3(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ Vec3 normalize_or(
    Vec3 value,
    float epsilon,
    Vec3 fallback) {
    const float value_length = length(value);
    return value_length > epsilon ? scale(value, 1.0f / value_length) : fallback;
}

__device__ __forceinline__ Complex complex(float real, float imag) {
    return {real, imag};
}

__device__ __forceinline__ Complex complex_add(Complex a, Complex b) {
    return {a.r + b.r, a.i + b.i};
}

__device__ __forceinline__ Complex complex_sub(Complex a, Complex b) {
    return {a.r - b.r, a.i - b.i};
}

__device__ __forceinline__ Complex complex_mul(Complex a, Complex b) {
    return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}

__device__ __forceinline__ Complex complex_scale(Complex value, float factor) {
    return {value.r * factor, value.i * factor};
}

__device__ __forceinline__ Complex complex_div_floor(
    Complex numerator,
    Complex denominator,
    float minimum_denominator) {
    const float divisor = fmaxf(
        denominator.r * denominator.r + denominator.i * denominator.i,
        minimum_denominator);
    return {
        (numerator.r * denominator.r + numerator.i * denominator.i) / divisor,
        (numerator.i * denominator.r - numerator.r * denominator.i) / divisor};
}

__device__ __forceinline__ Complex complex_sqrt_passive(Complex value) {
    const float magnitude = hypotf(value.r, value.i);
    const float real = sqrtf(fmaxf(0.0f, 0.5f * (magnitude + value.r)));
    const float imag_sign = value.i < 0.0f ? -1.0f : 1.0f;
    const float imag = imag_sign * sqrtf(fmaxf(0.0f, 0.5f * (magnitude - value.r)));
    return {real, imag};
}

__device__ __forceinline__ float complex_abs2(Complex value) {
    return value.r * value.r + value.i * value.i;
}

__device__ __forceinline__ Complex3 complex3_zero() {
    return {
        complex(0.0f, 0.0f),
        complex(0.0f, 0.0f),
        complex(0.0f, 0.0f)};
}

__device__ __forceinline__ Complex3 complex3_from_real(Vec3 value) {
    return {
        complex(value.x, 0.0f),
        complex(value.y, 0.0f),
        complex(value.z, 0.0f)};
}

__device__ __forceinline__ Complex3 complex3_add(Complex3 a, Complex3 b) {
    return {
        complex_add(a.x, b.x),
        complex_add(a.y, b.y),
        complex_add(a.z, b.z)};
}

__device__ __forceinline__ Complex3 complex3_axis(Vec3 axis, Complex value) {
    return {
        complex_scale(value, axis.x),
        complex_scale(value, axis.y),
        complex_scale(value, axis.z)};
}

__device__ __forceinline__ Complex complex3_dot_real(Complex3 value, Vec3 vector) {
    return complex(
        value.x.r * vector.x + value.y.r * vector.y + value.z.r * vector.z,
        value.x.i * vector.x + value.y.i * vector.y + value.z.i * vector.z);
}

__device__ __forceinline__ float complex3_power(Complex3 value) {
    return complex_abs2(value.x) + complex_abs2(value.y) + complex_abs2(value.z);
}

}  // namespace channel::math
