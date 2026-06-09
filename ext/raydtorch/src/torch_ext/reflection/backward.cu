#include <raydtorch/scene/geometry_kernels.h>
#include <raydtorch/reflection/kernels.h>
#include <raydtorch/common/math.cuh>

#include <vector>

namespace raydtorch {

namespace {

at::Tensor zero_vec3_like_batch(const at::Tensor &like, int64_t count) {
    return at::zeros({count, 3}, like.options());
}

at::Tensor zero_uv_like_batch(const at::Tensor &like, int64_t count) {
    return at::zeros({count, 2}, like.options());
}

at::Tensor zero_bary_like_batch(const at::Tensor &like, int64_t count) {
    return at::zeros({count, 3}, like.options());
}

at::Tensor select_bounce(const at::Tensor &value, int64_t bounce) {
    return value.select(1, bounce).contiguous();
}

at::Tensor reflect_vec(const at::Tensor &direction, const at::Tensor &normal) {
    at::Tensor dot = at::sum(direction * normal, 1, true);
    return (direction - 2.0f * dot * normal).contiguous();
}

at::Tensor raw_normals_for_prim(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &prim_id) {
    at::Tensor valid = prim_id.ge(0);
    at::Tensor safe_prim = at::where(valid, prim_id, at::zeros_like(prim_id)).to(at::kLong);
    at::Tensor tri = faces.index_select(0, safe_prim);
    at::Tensor i0 = tri.select(1, 0).to(at::kLong);
    at::Tensor i1 = tri.select(1, 1).to(at::kLong);
    at::Tensor i2 = tri.select(1, 2).to(at::kLong);
    at::Tensor v0 = vertices.index_select(0, i0);
    at::Tensor v1 = vertices.index_select(0, i1);
    at::Tensor v2 = vertices.index_select(0, i2);
    at::Tensor fn = at::cross(v1 - v0, v2 - v0, 1);
    at::Tensor len = at::sqrt(at::clamp_min(at::sum(fn * fn, 1, true), kDistanceEps));
    return (fn / len).contiguous();
}

at::Tensor normal_sign(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &prim_id,
    const at::Tensor &signed_normal) {
    at::Tensor raw = raw_normals_for_prim(vertices, faces, prim_id);
    at::Tensor same_side = at::sum(raw * signed_normal, 1, true).ge(0);
    return at::where(same_side, at::ones_like(signed_normal), -at::ones_like(signed_normal));
}

at::Tensor active_for_bounce(const at::Tensor &active, const at::Tensor &prim_id) {
    return at::logical_and(active, prim_id.ge(0)).contiguous();
}

at::Tensor mask_vec(const at::Tensor &value, const at::Tensor &active_bounce) {
    return (value * active_bounce.to(value.dtype()).reshape({value.size(0), 1})).contiguous();
}

} // namespace

ReflectionBackwardOutputs reflection_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &ray_o,
    const at::Tensor &ray_d,
    const at::Tensor &ray_tmax,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &grad_t) {
    const int64_t ray_count = ray_o.size(0);
    at::Tensor zeros_vec3 = zero_vec3_like_batch(ray_o, ray_count);
    at::Tensor zeros_uv = zero_uv_like_batch(ray_o, ray_count);
    at::Tensor zeros_bary = zero_bary_like_batch(ray_o, ray_count);
    IntersectBackwardOutputs hit_grad = intersect_backward_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t.reshape({ray_count}).contiguous(),
        zeros_vec3,
        zeros_vec3,
        zeros_vec3,
        zeros_uv,
        zeros_bary);
    return {
        hit_grad.grad_vertices,
        hit_grad.grad_ray_o,
        hit_grad.grad_ray_d,
        hit_grad.grad_ray_tmax,
    };
}

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
    const at::Tensor &grad_t,
    const at::Tensor &grad_image_sources) {
    (void)ray_tmax;
    const int64_t ray_count = ray_o.size(0);
    const int64_t max_bounces = tape_prim_id.size(1);

    std::vector<at::Tensor> origins(static_cast<size_t>(max_bounces));
    std::vector<at::Tensor> directions(static_cast<size_t>(max_bounces));
    std::vector<at::Tensor> image_states(static_cast<size_t>(max_bounces));
    origins[0] = ray_o.contiguous();
    directions[0] = ray_d.contiguous();
    image_states[0] = ray_o.contiguous();
    for (int64_t bounce = 0; bounce + 1 < max_bounces; ++bounce) {
        at::Tensor hit = select_bounce(tape_hit_points, bounce);
        at::Tensor normal = select_bounce(tape_normals, bounce);
        at::Tensor next_direction = reflect_vec(directions[static_cast<size_t>(bounce)], normal);
        origins[static_cast<size_t>(bounce + 1)] =
            (hit + static_cast<float>(kRayBias) * next_direction).contiguous();
        directions[static_cast<size_t>(bounce + 1)] = next_direction;
        image_states[static_cast<size_t>(bounce + 1)] =
            select_bounce(image_sources, bounce);
    }

    at::Tensor grad_vertices = at::zeros_like(vertices);
    at::Tensor grad_ray_tmax = at::zeros({ray_count}, ray_o.options());
    at::Tensor grad_origin_next = zero_vec3_like_batch(ray_o, ray_count);
    at::Tensor grad_direction_next = zero_vec3_like_batch(ray_o, ray_count);
    at::Tensor grad_image_next = zero_vec3_like_batch(ray_o, ray_count);
    at::Tensor grad_ray_o = zero_vec3_like_batch(ray_o, ray_count);
    at::Tensor grad_ray_d = zero_vec3_like_batch(ray_o, ray_count);
    at::Tensor zeros_uv = zero_uv_like_batch(ray_o, ray_count);
    at::Tensor zeros_bary = zero_bary_like_batch(ray_o, ray_count);

    for (int64_t bounce = max_bounces - 1; bounce >= 0; --bounce) {
        at::Tensor prim = select_bounce(tape_prim_id, bounce);
        at::Tensor active_b = active_for_bounce(active, prim);
        at::Tensor normal = select_bounce(tape_normals, bounce);
        at::Tensor hit = select_bounce(tape_hit_points, bounce);
        at::Tensor direction = directions[static_cast<size_t>(bounce)];
        at::Tensor image_before = image_states[static_cast<size_t>(bounce)];

        at::Tensor grad_p = zero_vec3_like_batch(ray_o, ray_count);
        at::Tensor grad_signed_n = zero_vec3_like_batch(ray_o, ray_count);
        at::Tensor grad_direction_current = zero_vec3_like_batch(ray_o, ray_count);

        at::Tensor grad_image_out = select_bounce(grad_image_sources, bounce) + grad_image_next;
        grad_image_out = mask_vec(grad_image_out, active_b);
        at::Tensor image_delta = image_before - hit;
        at::Tensor image_dist = at::sum(image_delta * normal, 1, true);
        at::Tensor image_gdotn = at::sum(grad_image_out * normal, 1, true);
        at::Tensor grad_image_prev =
            grad_image_out - 2.0f * image_gdotn * normal;
        grad_p = grad_p + 2.0f * image_gdotn * normal;
        grad_signed_n =
            grad_signed_n -
            2.0f * (image_gdotn * image_delta + image_dist * grad_image_out);

        at::Tensor grad_origin_out = mask_vec(grad_origin_next, active_b);
        grad_p = grad_p + grad_origin_out;
        at::Tensor grad_reflected =
            mask_vec(grad_direction_next + static_cast<float>(kRayBias) * grad_origin_out, active_b);
        at::Tensor dir_dot_n = at::sum(direction * normal, 1, true);
        at::Tensor refl_gdotn = at::sum(grad_reflected * normal, 1, true);
        grad_direction_current =
            grad_direction_current + grad_reflected - 2.0f * refl_gdotn * normal;
        grad_signed_n =
            grad_signed_n -
            2.0f * (refl_gdotn * direction + dir_dot_n * grad_reflected);

        at::Tensor sign = normal_sign(vertices, faces, prim, normal);
        at::Tensor grad_raw_n = grad_signed_n * sign;
        at::Tensor grad_t_b = select_bounce(grad_t, bounce);
        at::Tensor bary_b = select_bounce(tape_barycentric, bounce);
        IntersectBackwardOutputs hit_grad = intersect_backward_cuda(
            vertices,
            faces,
            origins[static_cast<size_t>(bounce)],
            direction,
            at::ones({ray_count}, ray_o.options()),
            active_b,
            prim,
            bary_b,
            grad_t_b.contiguous(),
            grad_p.contiguous(),
            grad_raw_n.contiguous(),
            at::zeros_like(grad_raw_n),
            zeros_uv,
            zeros_bary);
        grad_vertices = grad_vertices + hit_grad.grad_vertices;
        grad_origin_next = hit_grad.grad_ray_o;
        grad_direction_next = hit_grad.grad_ray_d + grad_direction_current;
        grad_image_next = grad_image_prev;

        if (bounce == 0) {
            grad_ray_o = grad_origin_next + grad_image_next;
            grad_ray_d = grad_direction_next;
        }
    }

    return {grad_vertices.contiguous(), grad_ray_o.contiguous(), grad_ray_d.contiguous(), grad_ray_tmax};
}

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
    const at::Tensor &image_sources) {
    const int64_t ray_count = ray_o.size(0);
    IntersectJvpOutputs hit_jvp = intersect_jvp_cuda(
        vertices,
        faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d);
    return {
        hit_jvp.tangent_t.reshape({ray_count, 1}),
        at::zeros_like(image_sources),
    };
}

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
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_ray_o,
    const at::Tensor &tangent_ray_d,
    const at::Tensor &image_sources) {
    const int64_t ray_count = ray_o.size(0);
    const int64_t max_bounces = tape_prim_id.size(1);
    at::Tensor tangent_t = at::zeros({ray_count, max_bounces}, ray_o.options());
    at::Tensor tangent_image_sources = at::zeros_like(image_sources);

    at::Tensor origin = ray_o.contiguous();
    at::Tensor direction = ray_d.contiguous();
    at::Tensor tangent_origin = tangent_ray_o.contiguous();
    at::Tensor tangent_direction = tangent_ray_d.contiguous();
    at::Tensor image_state = ray_o.contiguous();
    at::Tensor tangent_image_state = tangent_ray_o.contiguous();

    for (int64_t bounce = 0; bounce < max_bounces; ++bounce) {
        at::Tensor prim = select_bounce(tape_prim_id, bounce);
        at::Tensor active_b = active_for_bounce(active, prim);
        at::Tensor bary_b = select_bounce(tape_barycentric, bounce);
        IntersectJvpOutputs hit_jvp = intersect_jvp_cuda(
            vertices,
            faces,
            origin,
            direction,
            active_b,
            prim,
            bary_b,
            tangent_vertices,
            tangent_origin,
            tangent_direction);

        at::Tensor normal = select_bounce(tape_normals, bounce);
        at::Tensor sign = normal_sign(vertices, faces, prim, normal);
        at::Tensor tangent_normal = hit_jvp.tangent_n * sign;
        at::Tensor hit = select_bounce(tape_hit_points, bounce);
        at::Tensor tangent_hit = hit_jvp.tangent_p;
        tangent_t.select(1, bounce).copy_(hit_jvp.tangent_t);

        at::Tensor image_delta = image_state - hit;
        at::Tensor tangent_image_delta = tangent_image_state - tangent_hit;
        at::Tensor image_dist = at::sum(image_delta * normal, 1, true);
        at::Tensor tangent_image_dist =
            at::sum(tangent_image_delta * normal + image_delta * tangent_normal, 1, true);
        at::Tensor next_image_state =
            (image_state - 2.0f * image_dist * normal).contiguous();
        at::Tensor next_tangent_image_state =
            (tangent_image_state -
             2.0f * (tangent_image_dist * normal + image_dist * tangent_normal))
                .contiguous();
        next_tangent_image_state = mask_vec(next_tangent_image_state, active_b);
        tangent_image_sources.select(1, bounce).copy_(next_tangent_image_state);

        at::Tensor dir_dot_n = at::sum(direction * normal, 1, true);
        at::Tensor tangent_dir_dot_n =
            at::sum(tangent_direction * normal + direction * tangent_normal, 1, true);
        at::Tensor next_direction =
            (direction - 2.0f * dir_dot_n * normal).contiguous();
        at::Tensor next_tangent_direction =
            (tangent_direction -
             2.0f * (tangent_dir_dot_n * normal + dir_dot_n * tangent_normal))
                .contiguous();
        at::Tensor next_origin =
            (hit + static_cast<float>(kRayBias) * next_direction).contiguous();
        at::Tensor next_tangent_origin =
            (tangent_hit + static_cast<float>(kRayBias) * next_tangent_direction).contiguous();

        origin = next_origin;
        direction = next_direction;
        tangent_origin = mask_vec(next_tangent_origin, active_b);
        tangent_direction = mask_vec(next_tangent_direction, active_b);
        image_state = next_image_state;
        tangent_image_state = next_tangent_image_state;
    }

    return {tangent_t.contiguous(), tangent_image_sources.contiguous()};
}

ReflEpcBackwardOutputs refl_epc_backward_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor &grad_field_real,
    const at::Tensor &grad_field_imag,
    const at::Tensor &grad_path_length) {
    const int64_t ray_count = source.size(0);
    at::Tensor ray_d = (receiver - source).contiguous();
    at::Tensor ray_tmax = at::ones({ray_count}, source.options());
    at::Tensor denom = 1.f + tape_t;
    at::Tensor inv_denom = 1.f / denom;
    at::Tensor real_dt =
        -at::sin(tape_t) * inv_denom - at::cos(tape_t) * inv_denom * inv_denom;
    at::Tensor imag_dt =
        at::cos(tape_t) * inv_denom - at::sin(tape_t) * inv_denom * inv_denom;
    at::Tensor grad_t =
        grad_path_length.reshape({ray_count}) +
        grad_field_real.reshape({ray_count}) * real_dt +
        grad_field_imag.reshape({ray_count}) * imag_dt;
    ReflectionBackwardOutputs hit_grad = reflection_backward_cuda(
        vertices,
        faces,
        source,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t.contiguous());
    return {
        hit_grad.grad_vertices,
        hit_grad.grad_ray_o - hit_grad.grad_ray_d,
        hit_grad.grad_ray_d,
    };
}

ReflEpcJvpOutputs refl_epc_jvp_cuda(
    const at::Tensor &vertices,
    const at::Tensor &faces,
    const at::Tensor &source,
    const at::Tensor &receiver,
    const at::Tensor &active,
    const at::Tensor &tape_prim_id,
    const at::Tensor &tape_barycentric,
    const at::Tensor &tape_t,
    const at::Tensor &tangent_vertices,
    const at::Tensor &tangent_source,
    const at::Tensor &tangent_receiver) {
    const int64_t ray_count = source.size(0);
    at::Tensor ray_d = (receiver - source).contiguous();
    at::Tensor tangent_ray_d = (tangent_receiver - tangent_source).contiguous();
    ReflectionJvpOutputs hit_jvp = reflection_jvp_cuda(
        vertices,
        faces,
        source,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_source,
        tangent_ray_d,
        at::zeros({ray_count, 1, 3}, source.options()));
    at::Tensor tangent_t = hit_jvp.tangent_t.reshape({ray_count});
    at::Tensor denom = 1.f + tape_t;
    at::Tensor inv_denom = 1.f / denom;
    at::Tensor real_dt =
        -at::sin(tape_t) * inv_denom - at::cos(tape_t) * inv_denom * inv_denom;
    at::Tensor imag_dt =
        at::cos(tape_t) * inv_denom - at::sin(tape_t) * inv_denom * inv_denom;
    return {
        (real_dt * tangent_t).contiguous(),
        (imag_dt * tangent_t).contiguous(),
        tangent_t.contiguous(),
    };
}

} // namespace raydtorch
