#pragma once

#include <cmath>

#include <common/device.h>

namespace witwin::channel::native_ext::common {

struct Vec3f {
    float x;
    float y;
    float z;
};

struct Complexf {
    float re;
    float im;
};

WITWIN_KERNEL_DINLINE Vec3f make_vec3(float x, float y, float z) {
    return Vec3f{x, y, z};
}

WITWIN_KERNEL_DINLINE Vec3f add(Vec3f a, Vec3f b) {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

WITWIN_KERNEL_DINLINE Vec3f sub(Vec3f a, Vec3f b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

WITWIN_KERNEL_DINLINE Vec3f mul(Vec3f a, float s) {
    return make_vec3(a.x * s, a.y * s, a.z * s);
}

WITWIN_KERNEL_DINLINE Vec3f neg(Vec3f a) {
    return make_vec3(-a.x, -a.y, -a.z);
}

WITWIN_KERNEL_DINLINE float dot(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

WITWIN_KERNEL_DINLINE Vec3f cross(Vec3f a, Vec3f b) {
    return make_vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

WITWIN_KERNEL_DINLINE float norm(Vec3f a) {
    return sqrtf(fmaxf(dot(a, a), 0.0f));
}

WITWIN_KERNEL_HD_INLINE int ceil_div_int(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

WITWIN_KERNEL_DINLINE Vec3f safe_unit(Vec3f a, float eps) {
    float n = norm(a);
    if (n <= eps) {
        return make_vec3(0.0f, 0.0f, 0.0f);
    }
    return mul(a, 1.0f / n);
}

WITWIN_KERNEL_DINLINE Complexf make_complex(float re, float im) {
    return Complexf{re, im};
}

WITWIN_KERNEL_DINLINE Complexf cadd(Complexf a, Complexf b) {
    return make_complex(a.re + b.re, a.im + b.im);
}

WITWIN_KERNEL_DINLINE Complexf cmul(Complexf a, Complexf b) {
    return make_complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

WITWIN_KERNEL_DINLINE Complexf cscale(Complexf a, float s) {
    return make_complex(a.re * s, a.im * s);
}

} // namespace witwin::channel::native_ext::common
