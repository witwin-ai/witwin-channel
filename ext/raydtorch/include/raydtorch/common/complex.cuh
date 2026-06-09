#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace raydtorch {

struct Complex {
    float r;
    float i;
};

struct Complex3 {
    Complex x;
    Complex y;
    Complex z;
};

__forceinline__ __device__ Complex c_make(float r, float i = 0.f) {
    Complex z;
    z.r = r;
    z.i = i;
    return z;
}

__forceinline__ __device__ Complex c_add(Complex a, Complex b) {
    return c_make(a.r + b.r, a.i + b.i);
}

__forceinline__ __device__ Complex c_sub(Complex a, Complex b) {
    return c_make(a.r - b.r, a.i - b.i);
}

__forceinline__ __device__ Complex c_mul(Complex a, Complex b) {
    return c_make(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);
}

__forceinline__ __device__ Complex c_scale(Complex a, float s) {
    return c_make(a.r * s, a.i * s);
}

__forceinline__ __device__ Complex c_mul_real(Complex a, float s) {
    return c_scale(a, s);
}

__forceinline__ __device__ Complex c_div(Complex a, Complex b) {
    const float denom = fmaxf(b.r * b.r + b.i * b.i, 1e-20f);
    return c_make((a.r * b.r + a.i * b.i) / denom,
                  (a.i * b.r - a.r * b.i) / denom);
}

__forceinline__ __device__ float c_abs2(Complex z) {
    return z.r * z.r + z.i * z.i;
}

__forceinline__ __device__ Complex c_sqrt(Complex z) {
    const float r = hypotf(z.r, z.i);
    if (r <= 0.f) {
        return c_make(0.f, 0.f);
    }
    const float real_mag = sqrtf(fmaxf(0.f, 0.5f * (r + z.r)));
    const float imag_mag = sqrtf(fmaxf(0.f, 0.5f * (r - z.r)));
    const float imag = copysignf(imag_mag, z.i);
    return c_make(real_mag, imag);
}

__forceinline__ __device__ Complex c_exp_neg_i(float phase) {
    float s;
    float c;
    sincosf(phase, &s, &c);
    return c_make(c, -s);
}

__forceinline__ __device__ Complex3 c3_zero() {
    Complex3 v;
    v.x = c_make(0.f, 0.f);
    v.y = c_make(0.f, 0.f);
    v.z = c_make(0.f, 0.f);
    return v;
}

__forceinline__ __device__ Complex3 c3_from_real(float3 value) {
    Complex3 v;
    v.x = c_make(value.x, 0.f);
    v.y = c_make(value.y, 0.f);
    v.z = c_make(value.z, 0.f);
    return v;
}

__forceinline__ __device__ Complex3 c3_add(Complex3 a, Complex3 b) {
    Complex3 v;
    v.x = c_add(a.x, b.x);
    v.y = c_add(a.y, b.y);
    v.z = c_add(a.z, b.z);
    return v;
}

__forceinline__ __device__ Complex3 c3_scale_complex(float3 basis, Complex coeff) {
    Complex3 v;
    v.x = c_mul_real(coeff, basis.x);
    v.y = c_mul_real(coeff, basis.y);
    v.z = c_mul_real(coeff, basis.z);
    return v;
}

__forceinline__ __device__ Complex3 c3_mul_complex(Complex3 value, Complex coeff) {
    Complex3 v;
    v.x = c_mul(value.x, coeff);
    v.y = c_mul(value.y, coeff);
    v.z = c_mul(value.z, coeff);
    return v;
}

__forceinline__ __device__ Complex c3_dot_real(Complex3 value, float3 basis) {
    return c_add(c_add(c_mul_real(value.x, basis.x),
                       c_mul_real(value.y, basis.y)),
                 c_mul_real(value.z, basis.z));
}

__forceinline__ __device__ float c3_power(Complex3 value) {
    return c_abs2(value.x) + c_abs2(value.y) + c_abs2(value.z);
}

__forceinline__ __device__ bool finite_complex3(Complex3 value) {
    return isfinite(value.x.r) && isfinite(value.x.i) &&
           isfinite(value.y.r) && isfinite(value.y.i) &&
           isfinite(value.z.r) && isfinite(value.z.i);
}

} // namespace raydtorch
