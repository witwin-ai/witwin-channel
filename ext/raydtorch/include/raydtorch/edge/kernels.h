#pragma once

#include <ATen/ATen.h>

#include <raydtorch/scene/cache.h>

namespace raydtorch {

struct EdgeForwardOutputs {
    at::Tensor distance;
    at::Tensor edge_point;
    at::Tensor edge_t;
    at::Tensor shape_id;
    at::Tensor edge_id;
    at::Tensor global_edge_id;
    at::Tensor tape_edge_id;
    at::Tensor tape_s;
    at::Tensor tape_d;
};

EdgeForwardOutputs edge_forward_cuda(const SceneCache &scene, const at::Tensor &point);

struct EdgeForwardPublicOutputs {
    at::Tensor distance;
    at::Tensor edge_point;
    at::Tensor edge_t;
    at::Tensor shape_id;
    at::Tensor edge_id;
    at::Tensor global_edge_id;
};

EdgeForwardPublicOutputs edge_forward_noad_cuda(const SceneCache &scene, const at::Tensor &point);

struct EdgeRayForwardOutputs {
    at::Tensor distance;
    at::Tensor ray_t;
    at::Tensor point;
    at::Tensor edge_t;
    at::Tensor edge_point;
    at::Tensor shape_id;
    at::Tensor edge_id;
    at::Tensor global_edge_id;
    at::Tensor tape_edge_id;
};

EdgeRayForwardOutputs edge_ray_forward_cuda(
    const SceneCache &scene,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active);

void compute_edge_optix_aabbs_cuda(
    int64_t edge_count,
    const at::Tensor &edge_p0_x,
    const at::Tensor &edge_p0_y,
    const at::Tensor &edge_p0_z,
    const at::Tensor &edge_e1_x,
    const at::Tensor &edge_e1_y,
    const at::Tensor &edge_e1_z,
    float radius,
    at::Tensor &out_aabbs);

struct EdgeBackwardOutputs {
    at::Tensor grad_vertices;
    at::Tensor grad_point;
};

EdgeBackwardOutputs edge_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor &grad_distance,
    const at::Tensor &grad_edge_point,
    const at::Tensor &grad_edge_t);

struct EdgeJvpOutputs {
    at::Tensor tangent_distance;
    at::Tensor tangent_edge_point;
    at::Tensor tangent_edge_t;
};

EdgeJvpOutputs edge_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &edge_v0,
    const at::Tensor &edge_v1,
    const at::Tensor &point,
    const at::Tensor &tape_edge_id,
    const at::Tensor &tape_s,
    const at::Tensor &tape_d,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_point);

} // namespace raydtorch
