#include "bridge.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace {

using channel_native::rayd_bridge::optional_tensor;
using channel_native::rayd_bridge::raydn_reflection_accumulation_forward_fn;
using channel_native::rayd_bridge::raydn_reflection_epc_paths_backward_fn;
using channel_native::rayd_bridge::raydn_reflection_epc_paths_forward_fn;
using channel_native::rayd_bridge::raydn_reflection_epc_paths_jvp_fn;
using channel_native::rayd_bridge::raydn_scene_face_normals_backward_fn;
using channel_native::rayd_bridge::raydn_scene_face_normals_jvp_fn;
using channel_native::rayd_bridge::raydn_trace_reflections_backward_fn;
using channel_native::rayd_bridge::raydn_trace_reflections_forward_fn;
using channel_native::rayd_bridge::raydn_trace_reflections_forward_tape_fn;
using channel_native::rayd_bridge::raydn_trace_reflections_jvp_fn;
using channel_native::rayd_bridge::tensor_or_none;

}  // namespace

pybind11::tuple cn_raydn_trace_reflections_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 3;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_trace_reflections_forward_fn(0)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        max_bounces,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN reflection trace returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_raydn_reflection_epc_paths_forward(
    int64_t scene_handle,
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
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 6;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_epc_paths_forward_fn(0)(
        scene_handle,
        &source,
        &receiver,
        active_ptr,
        &expected_prim_ids,
        &direct_plane_points,
        &direct_plane_normals,
        &surface_group_id,
        &surface_group_size,
        &surface_group_members,
        max_bounces,
        visibility_ignore_mode,
        1.0e-3,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN reflection EPC path export returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_raydn_trace_reflections_forward_tape(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t max_bounces) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 9;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_trace_reflections_forward_tape_fn(0)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        max_bounces,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN tape reflection trace returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_raydn_trace_reflections_backward(
    int64_t scene_handle,
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
    at::Tensor active_storage;
    at::Tensor grad_t_storage;
    at::Tensor grad_image_sources_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *grad_t_ptr = optional_tensor(std::move(grad_t), grad_t_storage);
    const at::Tensor *grad_image_sources_ptr =
        optional_tensor(std::move(grad_image_sources), grad_image_sources_storage);
    constexpr int64_t kOutputCount = 4;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_trace_reflections_backward_fn(0)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        &tape_prim_id,
        &tape_barycentric,
        &tape_hit_points,
        &tape_normals,
        &image_sources,
        grad_t_ptr,
        grad_image_sources_ptr,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN reflection backward returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = tensor_or_none(outputs[static_cast<size_t>(i)]);
    return result;
}

pybind11::tuple cn_raydn_trace_reflections_jvp(
    int64_t scene_handle,
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
    at::Tensor active_storage;
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_ray_o_storage;
    at::Tensor tangent_ray_d_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *tangent_vertices_ptr =
        optional_tensor(std::move(tangent_vertices), tangent_vertices_storage);
    const at::Tensor *tangent_ray_o_ptr =
        optional_tensor(std::move(tangent_ray_o), tangent_ray_o_storage);
    const at::Tensor *tangent_ray_d_ptr =
        optional_tensor(std::move(tangent_ray_d), tangent_ray_d_storage);
    constexpr int64_t kOutputCount = 2;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_trace_reflections_jvp_fn(0)(
        scene_handle,
        &ray_o,
        &ray_d,
        active_ptr,
        &tape_prim_id,
        &tape_barycentric,
        &tape_hit_points,
        &tape_normals,
        tangent_vertices_ptr,
        tangent_ray_o_ptr,
        tangent_ray_d_ptr,
        &image_sources,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN reflection jvp returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_raydn_reflection_epc_paths_backward(
    int64_t scene_handle,
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
    at::Tensor grad_points_storage;
    at::Tensor grad_normals_storage;
    at::Tensor grad_path_length_storage;
    const at::Tensor *grad_points_ptr =
        optional_tensor(std::move(grad_points), grad_points_storage);
    const at::Tensor *grad_normals_ptr =
        optional_tensor(std::move(grad_normals), grad_normals_storage);
    const at::Tensor *grad_path_length_ptr =
        optional_tensor(std::move(grad_path_length), grad_path_length_storage);
    constexpr int64_t kOutputCount = 3;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_epc_paths_backward_fn(0)(
        scene_handle,
        &source,
        &receiver,
        &sequence,
        &plane_points,
        &plane_normals,
        &valid,
        &bounce_count,
        grad_points_ptr,
        grad_normals_ptr,
        grad_path_length_ptr,
        need_grad_vertices,
        need_grad_source,
        need_grad_receiver,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN reflection EPC paths backward returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = tensor_or_none(outputs[static_cast<size_t>(i)]);
    return result;
}

pybind11::tuple cn_raydn_reflection_epc_paths_jvp(
    int64_t scene_handle,
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
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_source_storage;
    at::Tensor tangent_receiver_storage;
    const at::Tensor *tangent_vertices_ptr =
        optional_tensor(std::move(tangent_vertices), tangent_vertices_storage);
    const at::Tensor *tangent_source_ptr =
        optional_tensor(std::move(tangent_source), tangent_source_storage);
    const at::Tensor *tangent_receiver_ptr =
        optional_tensor(std::move(tangent_receiver), tangent_receiver_storage);
    constexpr int64_t kOutputCount = 3;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_epc_paths_jvp_fn(0)(
        scene_handle,
        &source,
        &receiver,
        &sequence,
        &plane_points,
        &plane_normals,
        &valid,
        &bounce_count,
        tangent_vertices_ptr,
        tangent_source_ptr,
        tangent_receiver_ptr,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN reflection EPC paths jvp returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

at::Tensor cn_raydn_scene_face_normals_backward(
    int64_t scene_handle,
    torch::Tensor grad_face_normals) {
    constexpr int64_t kOutputCount = 1;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_scene_face_normals_backward_fn(0)(
        scene_handle,
        &grad_face_normals,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN scene face normals backward returned an invalid output count");
    return outputs[0];
}

at::Tensor cn_raydn_scene_face_normals_jvp(
    int64_t scene_handle,
    torch::Tensor tangent_vertices) {
    constexpr int64_t kOutputCount = 1;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_scene_face_normals_jvp_fn(0)(
        scene_handle,
        &tangent_vertices,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN scene face normals jvp returned an invalid output count");
    return outputs[0];
}

pybind11::tuple cn_bdpt_reflection_accumulation_forward(
    int64_t scene_handle,
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
    constexpr int64_t kOutputCount = 18;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_accumulation_forward_fn(0)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        &active,
        &tx,
        &tx_pol,
        &material_eta_r,
        &material_sigma,
        &material_mu_r,
        &material_gain,
        &material_valid,
        max_bounces,
        grid_axis,
        grid_position,
        grid_coord0_min,
        grid_coord0_max,
        grid_coord1_min,
        grid_coord1_max,
        grid_resolution0,
        grid_resolution1,
        wavelength,
        solid_angle_per_ray,
        collect_wedges,
        collect_wedge_prefixes,
        wedge_capacity,
        wedge_sample_stride,
        accumulation_strategy,
        compact_min_samples,
        staged_min_samples_per_cell,
        procedural_sample_count,
        streaming_los_enabled,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN reflection accumulation returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}
