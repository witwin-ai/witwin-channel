#include "resource.h"

#include <cstdint>
#include <stdexcept>

pybind11::tuple cn_rayd_trace_reflections_forward(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces) {
    rayd::torch::ReflectionTraceResult out = rayd::torch::trace_reflections_forward(
        scene.resource(),
        rayd::torch::ReflectionTraceRequest{
            {ray_o, ray_d, ray_tmax, optional_tensor(active)}, max_bounces});
    return pybind11::make_tuple(out.valid, out.t, out.prim_ids);
}

pybind11::tuple cn_rayd_reflection_epc_paths_forward(
    RayDSceneResource &scene,
    torch::Tensor source,
    torch::Tensor receiver,
    pybind11::object active,
    torch::Tensor expected_prim_ids,
    torch::Tensor direct_plane_points,
    torch::Tensor direct_plane_normals,
    torch::Tensor surface_group_id,
    torch::Tensor surface_group_size,
    torch::Tensor surface_group_members,
    int64_t max_bounces,
    int64_t visibility_ignore_mode) {
    rayd::torch::ReflectionEpcRequest request;
    request.source = source;
    request.receiver = receiver;
    request.active = optional_tensor(active);
    request.expected_prim_ids = expected_prim_ids;
    request.direct_plane_points = direct_plane_points;
    request.direct_plane_normals = direct_plane_normals;
    request.surface_group_id = surface_group_id;
    request.surface_group_size = surface_group_size;
    request.surface_group_members = surface_group_members;
    request.max_bounces = max_bounces;
    request.visibility_ignore_mode = visibility_ignore_mode;
    request.plane_tolerance = 1.0e-3;
    rayd::torch::ReflectionEpcResult out =
        rayd::torch::reflection_epc_paths_forward(scene.resource(), request);
    return pybind11::make_tuple(
        out.valid,
        out.path_length,
        out.resolved_prim_ids,
        out.surface_group_ids,
        out.hit_positions,
        out.normals);
}

pybind11::tuple cn_rayd_trace_reflections_forward_tape(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces) {
    rayd::torch::ReflectionTraceTapeResult out =
        rayd::torch::trace_reflections_forward_tape(
            scene.resource(),
            rayd::torch::ReflectionTraceRequest{
                {ray_o, ray_d, ray_tmax, optional_tensor(active)}, max_bounces});
    return pybind11::make_tuple(
        out.valid,
        out.t,
        out.image_sources,
        out.prim_ids,
        out.tape_prim_id,
        out.tape_barycentric,
        out.tape_hit_points,
        out.tape_normals,
        out.active_ctx);
}

pybind11::tuple cn_rayd_trace_reflections_backward(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_hit_points,
    torch::Tensor tape_normals,
    torch::Tensor image_sources,
    pybind11::object grad_t,
    pybind11::object grad_image_sources) {
    rayd::torch::ReflectionTraceBackwardRequest request;
    request.rays = {ray_o, ray_d, ray_tmax, optional_tensor(active)};
    request.tape_prim_id = tape_prim_id;
    request.tape_barycentric = tape_barycentric;
    request.tape_hit_points = tape_hit_points;
    request.tape_normals = tape_normals;
    request.image_sources = image_sources;
    request.grad_t = optional_tensor(grad_t);
    request.grad_image_sources = optional_tensor(grad_image_sources);
    rayd::torch::ReflectionTraceBackwardResult out =
        rayd::torch::trace_reflections_backward(scene.resource(), request);
    return pybind11::make_tuple(
        tensor_or_none(out.grad_vertices),
        tensor_or_none(out.grad_ray_o),
        tensor_or_none(out.grad_ray_d),
        tensor_or_none(out.grad_ray_tmax));
}

pybind11::tuple cn_rayd_trace_reflections_jvp(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    pybind11::object active,
    torch::Tensor tape_prim_id,
    torch::Tensor tape_barycentric,
    torch::Tensor tape_hit_points,
    torch::Tensor tape_normals,
    pybind11::object tangent_vertices,
    pybind11::object tangent_ray_o,
    pybind11::object tangent_ray_d,
    torch::Tensor image_sources) {
    rayd::torch::ReflectionTraceJvpRequest request;
    request.ray_o = ray_o;
    request.ray_d = ray_d;
    request.active = optional_tensor(active);
    request.tape_prim_id = tape_prim_id;
    request.tape_barycentric = tape_barycentric;
    request.tape_hit_points = tape_hit_points;
    request.tape_normals = tape_normals;
    request.tangent_vertices = optional_tensor(tangent_vertices);
    request.tangent_ray_o = optional_tensor(tangent_ray_o);
    request.tangent_ray_d = optional_tensor(tangent_ray_d);
    request.image_sources = image_sources;
    rayd::torch::ReflectionTraceJvpResult out =
        rayd::torch::trace_reflections_jvp(scene.resource(), request);
    return pybind11::make_tuple(out.tangent_t, out.tangent_image_sources);
}

pybind11::tuple cn_rayd_reflection_epc_paths_backward(
    RayDSceneResource &scene,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor sequence,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor valid,
    torch::Tensor bounce_count,
    pybind11::object grad_points,
    pybind11::object grad_normals,
    pybind11::object grad_path_length,
    bool need_grad_vertices,
    bool need_grad_source,
    bool need_grad_receiver) {
    rayd::torch::ReflectionEpcBackwardRequest request;
    request.source = source;
    request.receiver = receiver;
    request.sequence = sequence;
    request.plane_points = plane_points;
    request.plane_normals = plane_normals;
    request.valid = valid;
    request.bounce_count = bounce_count;
    request.grad_points = optional_tensor(grad_points);
    request.grad_normals = optional_tensor(grad_normals);
    request.grad_path_length = optional_tensor(grad_path_length);
    request.need_grad_vertices = need_grad_vertices;
    request.need_grad_source = need_grad_source;
    request.need_grad_receiver = need_grad_receiver;
    rayd::torch::ReflectionEpcBackwardResult out =
        rayd::torch::reflection_epc_paths_backward(scene.resource(), request);
    return pybind11::make_tuple(
        tensor_or_none(out.grad_vertices),
        tensor_or_none(out.grad_source),
        tensor_or_none(out.grad_receiver));
}

pybind11::tuple cn_rayd_reflection_epc_paths_jvp(
    RayDSceneResource &scene,
    torch::Tensor source,
    torch::Tensor receiver,
    torch::Tensor sequence,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor valid,
    torch::Tensor bounce_count,
    pybind11::object tangent_vertices,
    pybind11::object tangent_source,
    pybind11::object tangent_receiver) {
    rayd::torch::ReflectionEpcJvpRequest request;
    request.source = source;
    request.receiver = receiver;
    request.sequence = sequence;
    request.plane_points = plane_points;
    request.plane_normals = plane_normals;
    request.valid = valid;
    request.bounce_count = bounce_count;
    request.tangent_vertices = optional_tensor(tangent_vertices);
    request.tangent_source = optional_tensor(tangent_source);
    request.tangent_receiver = optional_tensor(tangent_receiver);
    rayd::torch::ReflectionEpcJvpResult out =
        rayd::torch::reflection_epc_paths_jvp(scene.resource(), request);
    return pybind11::make_tuple(
        out.tangent_points, out.tangent_normals, out.tangent_path_length);
}

at::Tensor cn_rayd_scene_face_normals_backward(
    RayDSceneResource &scene,
    torch::Tensor grad_face_normals) {
    return rayd::torch::scene_face_normals_backward(
        scene.resource(), grad_face_normals);
}

at::Tensor cn_rayd_scene_face_normals_jvp(
    RayDSceneResource &scene,
    torch::Tensor tangent_vertices) {
    return rayd::torch::scene_face_normals_jvp(
        scene.resource(), tangent_vertices);
}

pybind11::tuple cn_bdpt_reflection_accumulation_forward(
    RayDSceneResource &scene,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    torch::Tensor active,
    torch::Tensor tx,
    torch::Tensor tx_pol,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t max_bounces,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double wavelength,
    double solid_angle_per_ray,
    bool collect_wedges,
    bool collect_wedge_prefixes,
    int64_t wedge_capacity,
    int64_t wedge_sample_stride,
    int64_t accumulation_strategy,
    int64_t compact_min_samples,
    int64_t staged_min_samples_per_cell,
    int64_t procedural_sample_count,
    bool streaming_los_enabled) {
    if (accumulation_strategy == 4) {
        if (procedural_sample_count <= 0)
            throw std::runtime_error("streaming_planar requires a positive procedural_sample_count");
        if (tx.dim() != 2 || tx.size(0) != 1 || tx.size(1) != 3)
            throw std::runtime_error("streaming_planar expects one transmitter position");

        const auto dopts = ray_o.options().dtype(at::kDouble);
        at::Tensor index = at::arange(procedural_sample_count, dopts);
        at::Tensor azimuth = index / 1.618033988749894848204586834365638118;
        at::Tensor azimuth_u = azimuth - at::floor(azimuth);
        at::Tensor phi = azimuth_u * (2.0 * 3.141592653589793238462643383279502884);
        at::Tensor elevation = procedural_sample_count == 1
            ? at::zeros_like(index)
            : index / static_cast<double>(procedural_sample_count - 1);
        at::Tensor z = 1.0 - 2.0 * elevation;
        at::Tensor radial = at::sqrt(at::clamp_min(1.0 - z * z, 0.0));
        at::Tensor x = radial * at::cos(phi);
        at::Tensor y = radial * at::sin(phi);
        ray_d = at::stack({x, y, z}, 1).to(ray_o.scalar_type()).contiguous();

        at::Tensor pole = radial <= 1.0e-6;
        at::Tensor safe_radial = at::clamp_min(radial, 1.0e-6);
        at::Tensor pole_x = at::where(z >= 0.0, at::ones_like(z), -at::ones_like(z));
        at::Tensor pol_x = at::where(pole, pole_x, z * x / safe_radial);
        at::Tensor pol_y = at::where(pole, at::zeros_like(z), z * y / safe_radial);
        at::Tensor pol_z = at::where(pole, at::zeros_like(z), -radial);
        tx_pol = at::stack({pol_x, pol_y, pol_z}, 1).to(ray_o.scalar_type()).contiguous();

        ray_o = tx.select(0, 0).reshape({1, 3})
                    .expand({procedural_sample_count, 3}).contiguous();
        tx = ray_o;
        ray_tmax = at::empty({0}, ray_o.options());
        active = at::ones({procedural_sample_count}, ray_o.options().dtype(at::kBool));
        accumulation_strategy = 1;
        procedural_sample_count = 0;
    }
    rayd::torch::ReflectionAccumulationConfig config;
    config.rays = {ray_o, ray_d, ray_tmax, active};
    config.tx = tx;
    config.tx_pol = tx_pol;
    config.material = {
        material_eta_r, material_sigma, material_mu_r, material_gain, material_valid};
    config.max_bounces = max_bounces;
    config.grid = {
        grid_axis,
        grid_position,
        grid_coord0_min,
        grid_coord0_max,
        grid_coord1_min,
        grid_coord1_max,
        grid_resolution0,
        grid_resolution1,
        0.0};
    config.wavelength = wavelength;
    config.solid_angle_per_ray = solid_angle_per_ray;
    config.collect_wedges = collect_wedges;
    config.collect_wedge_prefixes = collect_wedge_prefixes;
    config.wedge_capacity = wedge_capacity;
    config.wedge_sample_stride = wedge_sample_stride;
    config.accumulation_strategy = accumulation_strategy;
    config.compact_min_samples = compact_min_samples;
    config.staged_min_samples_per_cell = staged_min_samples_per_cell;
    config.procedural_sample_count = procedural_sample_count;
    config.include_los = streaming_los_enabled;
    rayd::torch::ReflectionAccumulationResult out =
        rayd::torch::reflection_accumulation_forward(scene.resource(), config);
    return pybind11::make_tuple(
        out.power,
        out.field_x_re,
        out.field_x_im,
        out.field_y_re,
        out.field_y_im,
        out.field_z_re,
        out.field_z_im,
        out.reflection_count,
        out.wedge_count,
        out.wedge_ray_index,
        out.wedge_hit,
        out.wedge_normal,
        out.wedge_prim_id,
        out.wedge_direction,
        out.wedge_source,
        out.wedge_source_power,
        out.wedge_initial_direction,
        out.wedge_bounce_depth);
}
