#pragma once

#include <ATen/ATen.h>

#include <raydn/scene/cache.h>

namespace raydn {

struct VisibilityForwardOutputs {
    at::Tensor visible;
    at::Tensor tape_prim_id;
    at::Tensor tape_t;
};

VisibilityForwardOutputs visibility_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &start,
    const at::Tensor &end,
    const at::Tensor &active);

struct ReflectionBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_ray_o;
    at::Tensor grad_ray_d;
    at::Tensor grad_ray_tmax;
};

ReflectionBackwardOutputs reflection_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t);

struct ReflectionJvpOutputs {
    at::Tensor tangent_t;
    at::Tensor tangent_image_sources;
};

ReflectionJvpOutputs reflection_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_ray_o,
    const at::Tensor &tangent_ray_d,
    const at::Tensor &image_sources);

ReflectionBackwardOutputs reflection_chain_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_hit_points,
    const at::Tensor &tape_normals,
    const at::Tensor &image_sources,
    const at::Tensor *grad_t,
    const at::Tensor *grad_image_sources);

ReflectionJvpOutputs reflection_chain_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_hit_points,
    const at::Tensor &tape_normals,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_ray_o,
    const at::Tensor *tangent_ray_d,
    const at::Tensor &image_sources);

struct ReflEpcBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_source;
    at::Tensor grad_receiver;
};

ReflEpcBackwardOutputs refl_epc_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor *grad_field_real,
    const at::Tensor *grad_field_imag,
    const at::Tensor *grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver);

struct ReflEpcJvpOutputs {
    at::Tensor tangent_field_real;
    at::Tensor tangent_field_imag;
    at::Tensor tangent_path_length;
};

ReflEpcJvpOutputs refl_epc_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor *tangent_vertices,
    const at::Tensor *tangent_source,
    const at::Tensor *tangent_receiver);

} // namespace raydn
