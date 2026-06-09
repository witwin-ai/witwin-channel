#include <raydn/scene/cache.h>

#include <torch/extension.h>
#include <torch/library.h>

#include <vector>

namespace raydn {

namespace {

using ScenePtr = c10::intrusive_ptr<SceneHandle>;
using OptionalTensor = c10::optional<at::Tensor>;
using OptionalTensorList = std::vector<OptionalTensor>;

py::object to_py_optional(const OptionalTensor &value) {
    if (!value.has_value())
        return py::none();
    return py::cast(*value);
}

OptionalTensorList tuple_to_optional_tensor_list(const py::tuple &tuple, size_t start = 0) {
    OptionalTensorList out;
    out.reserve(py::len(tuple) - start);
    for (size_t i = start; i < py::len(tuple); ++i) {
        py::handle item = tuple[i];
        if (item.is_none())
            out.emplace_back(c10::nullopt);
        else
            out.emplace_back(item.cast<at::Tensor>());
    }
    return out;
}

OptionalTensorList tensors(std::initializer_list<at::Tensor> values) {
    OptionalTensorList out;
    out.reserve(values.size());
    for (const at::Tensor &value : values)
        out.emplace_back(value);
    return out;
}

int64_t handle(const ScenePtr &scene) {
    return scene->handle;
}

} // namespace

// Existing pybind entry points kept as compatibility wrappers during migration.
py::tuple intersect_forward_op(int64_t, at::Tensor, at::Tensor, at::Tensor, py::object);
py::tuple intersect_forward_flags_op(int64_t, at::Tensor, at::Tensor, at::Tensor, py::object, int64_t);
at::Tensor intersect_forward_t_op(int64_t, at::Tensor, at::Tensor, at::Tensor, py::object);
py::tuple intersect_forward_ad_flags_op(int64_t, at::Tensor, at::Tensor, at::Tensor, py::object, int64_t);
at::Tensor intersect_ad_t_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object);
py::tuple intersect_ad_flags_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, int64_t);
py::tuple intersection_empty_fields_op(int64_t, at::Tensor);
py::tuple intersect_backward_optional_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, py::object, py::object, py::object, py::object, py::object, bool, bool, bool, bool);
py::tuple intersect_backward_t_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool, bool, bool, bool);
py::tuple intersect_jvp_optional_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, py::object, py::object, int64_t);

py::tuple nearest_edge_forward_op(int64_t, at::Tensor);
py::tuple nearest_edge_forward_noad_op(int64_t, at::Tensor);
py::tuple nearest_edge_ray_forward_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
py::tuple nearest_edge_backward_optional_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, py::object, py::object, py::object);
py::tuple nearest_edge_jvp_optional_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, py::object);

py::tuple visibility_forward_op(int64_t, at::Tensor, at::Tensor, py::object);
py::tuple trace_reflections_forward_op(int64_t, at::Tensor, at::Tensor, at::Tensor, py::object, int64_t);
py::tuple trace_reflections_forward_noad_op(int64_t, at::Tensor, at::Tensor, at::Tensor, py::object, int64_t);
py::tuple trace_reflections_forward_reduced_op(int64_t, at::Tensor, at::Tensor, at::Tensor, py::object, int64_t);
py::tuple trace_reflections_backward_optional_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, py::object);
py::tuple trace_reflections_jvp_optional_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, py::object, py::object, at::Tensor);
py::tuple trace_refl_epc_field_forward_op(int64_t, at::Tensor, at::Tensor, py::object, int64_t);
py::tuple trace_refl_epc_field_backward_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, py::object, py::object, bool, bool, bool);
py::tuple trace_refl_epc_field_jvp_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, py::object, py::object, py::object);
py::tuple reflection_dedup_forward_op(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, double);
py::tuple reflection_accumulation_forward_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, double, double, double, double, double, int64_t, int64_t, double);

py::tuple diffraction_paths_order1_forward_op(int64_t, at::Tensor, at::Tensor, OptionalTensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, double);
py::tuple diffraction_accumulation_forward_op(int64_t, OptionalTensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, OptionalTensor, OptionalTensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, double, double, double, double, double, int64_t, int64_t, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, int64_t);
py::tuple diffraction_accumulation_direct_backward_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, OptionalTensor, at::Tensor, at::Tensor, int64_t, int64_t, double, double, double, double, double, int64_t, int64_t, double, double, int64_t, int64_t, int64_t, int64_t, OptionalTensor, OptionalTensor);
py::tuple diffraction_accumulation_direct_jvp_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, OptionalTensor, at::Tensor, at::Tensor, int64_t, int64_t, double, double, double, double, double, int64_t, int64_t, double, double, int64_t, int64_t, int64_t, int64_t, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor);
py::tuple diffraction_accumulation_chain_backward_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, double, double, double, double, double, int64_t, int64_t, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, OptionalTensor, OptionalTensor);
py::tuple diffraction_accumulation_chain_jvp_op(int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, double, double, double, double, double, int64_t, int64_t, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor, OptionalTensor);
py::tuple diffraction_coherent_accumulation_forward_op(int64_t, OptionalTensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, OptionalTensor, OptionalTensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, double, double, double, double, double, int64_t, int64_t, double, double, bool, bool);

py::tuple reflection_trace_stats_op(at::Tensor, at::Tensor);
py::tuple diffraction_path_stats_op(at::Tensor, at::Tensor, at::Tensor);
py::tuple default_dfr_material_op(int64_t, at::Tensor);
at::Tensor intersection_valid_op(at::Tensor, at::Tensor);
at::Tensor camera_sample_to_world_op(at::Tensor, double, double, double);
at::Tensor camera_sample_to_world_backward_op(at::Tensor, int64_t, double, double, double);
at::Tensor camera_world_to_sample_op(at::Tensor, double, double);
at::Tensor camera_world_to_sample_backward_op(at::Tensor, at::Tensor, double, double);
py::tuple camera_sample_ray_op(at::Tensor, double, double);
at::Tensor camera_sample_ray_backward_op(at::Tensor, OptionalTensor, double, double);

OptionalTensorList intersect_forward_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(intersect_forward_op(handle(scene), ray_o, ray_d, ray_tmax, to_py_optional(active)));
}

OptionalTensorList intersect_forward_flags_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active, int64_t flags) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(intersect_forward_flags_op(handle(scene), ray_o, ray_d, ray_tmax, to_py_optional(active), flags));
}

at::Tensor intersect_forward_t_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active) {
    py::gil_scoped_acquire gil;
    return intersect_forward_t_op(handle(scene), ray_o, ray_d, ray_tmax, to_py_optional(active));
}

OptionalTensorList intersect_forward_ad_flags_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active, int64_t flags) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(intersect_forward_ad_flags_op(handle(scene), ray_o, ray_d, ray_tmax, to_py_optional(active), flags));
}

at::Tensor intersect_ad_t_dispatch(ScenePtr scene, at::Tensor vertices, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active) {
    py::gil_scoped_acquire gil;
    return intersect_ad_t_op(handle(scene), vertices, ray_o, ray_d, ray_tmax, to_py_optional(active));
}

OptionalTensorList intersect_ad_flags_dispatch(ScenePtr scene, at::Tensor vertices, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active, int64_t flags) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(intersect_ad_flags_op(handle(scene), vertices, ray_o, ray_d, ray_tmax, to_py_optional(active), flags));
}

OptionalTensorList intersection_empty_fields_dispatch(ScenePtr scene, at::Tensor like) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(intersection_empty_fields_op(handle(scene), like));
}

OptionalTensorList intersect_backward_optional_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, at::Tensor active, at::Tensor tape_prim_id, at::Tensor tape_barycentric, OptionalTensor grad_t, OptionalTensor grad_p, OptionalTensor grad_n, OptionalTensor grad_geo_n, OptionalTensor grad_uv, OptionalTensor grad_barycentric, bool need_grad_vertices, bool need_grad_ray_o, bool need_grad_ray_d, bool need_grad_ray_tmax) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(intersect_backward_optional_op(handle(scene), ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, to_py_optional(grad_t), to_py_optional(grad_p), to_py_optional(grad_n), to_py_optional(grad_geo_n), to_py_optional(grad_uv), to_py_optional(grad_barycentric), need_grad_vertices, need_grad_ray_o, need_grad_ray_d, need_grad_ray_tmax));
}

OptionalTensorList intersect_backward_t_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor active, at::Tensor tape_prim_id, at::Tensor tape_barycentric, at::Tensor grad_t, bool need_grad_vertices, bool need_grad_ray_o, bool need_grad_ray_d, bool need_grad_ray_tmax) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(intersect_backward_t_op(handle(scene), ray_o, ray_d, active, tape_prim_id, tape_barycentric, grad_t, need_grad_vertices, need_grad_ray_o, need_grad_ray_d, need_grad_ray_tmax));
}

OptionalTensorList intersect_jvp_optional_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor active, at::Tensor tape_prim_id, at::Tensor tape_barycentric, OptionalTensor tangent_vertices, OptionalTensor tangent_ray_o, OptionalTensor tangent_ray_d, int64_t flags) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(intersect_jvp_optional_op(handle(scene), ray_o, ray_d, active, tape_prim_id, tape_barycentric, to_py_optional(tangent_vertices), to_py_optional(tangent_ray_o), to_py_optional(tangent_ray_d), flags));
}

OptionalTensorList nearest_edge_forward_dispatch(ScenePtr scene, at::Tensor point) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(nearest_edge_forward_op(handle(scene), point));
}

OptionalTensorList nearest_edge_forward_noad_dispatch(ScenePtr scene, at::Tensor point) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(nearest_edge_forward_noad_op(handle(scene), point));
}

OptionalTensorList nearest_edge_ray_forward_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, at::Tensor active) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(nearest_edge_ray_forward_op(handle(scene), ray_o, ray_d, ray_tmax, active));
}

OptionalTensorList nearest_edge_backward_optional_dispatch(ScenePtr scene, at::Tensor point, at::Tensor tape_edge_id, at::Tensor tape_s, at::Tensor tape_d, OptionalTensor grad_distance, OptionalTensor grad_edge_point, OptionalTensor grad_edge_t, OptionalTensor grad_edge_t_alias) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(nearest_edge_backward_optional_op(handle(scene), point, tape_edge_id, tape_s, tape_d, to_py_optional(grad_distance), to_py_optional(grad_edge_point), to_py_optional(grad_edge_t), to_py_optional(grad_edge_t_alias)));
}

OptionalTensorList nearest_edge_jvp_optional_dispatch(ScenePtr scene, at::Tensor point, at::Tensor tape_edge_id, at::Tensor tape_s, at::Tensor tape_d, OptionalTensor tangent_vertices, OptionalTensor tangent_point) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(nearest_edge_jvp_optional_op(handle(scene), point, tape_edge_id, tape_s, tape_d, to_py_optional(tangent_vertices), to_py_optional(tangent_point)));
}

OptionalTensorList visibility_forward_dispatch(ScenePtr scene, at::Tensor start, at::Tensor end, OptionalTensor active) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(visibility_forward_op(handle(scene), start, end, to_py_optional(active)));
}

OptionalTensorList trace_reflections_forward_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active, int64_t max_bounces) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(trace_reflections_forward_op(handle(scene), ray_o, ray_d, ray_tmax, to_py_optional(active), max_bounces));
}

OptionalTensorList trace_reflections_forward_noad_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active, int64_t max_bounces) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(trace_reflections_forward_noad_op(handle(scene), ray_o, ray_d, ray_tmax, to_py_optional(active), max_bounces));
}

OptionalTensorList trace_reflections_forward_reduced_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, OptionalTensor active, int64_t max_bounces) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(trace_reflections_forward_reduced_op(handle(scene), ray_o, ray_d, ray_tmax, to_py_optional(active), max_bounces));
}

OptionalTensorList trace_reflections_backward_optional_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, at::Tensor active, at::Tensor tape_prim_id, at::Tensor tape_barycentric, at::Tensor tape_hit_points, at::Tensor tape_normals, at::Tensor image_sources, OptionalTensor grad_t, OptionalTensor grad_image_sources) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(trace_reflections_backward_optional_op(handle(scene), ray_o, ray_d, ray_tmax, active, tape_prim_id, tape_barycentric, tape_hit_points, tape_normals, image_sources, to_py_optional(grad_t), to_py_optional(grad_image_sources)));
}

OptionalTensorList trace_reflections_jvp_optional_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor active, at::Tensor tape_prim_id, at::Tensor tape_barycentric, at::Tensor tape_hit_points, at::Tensor tape_normals, OptionalTensor tangent_vertices, OptionalTensor tangent_ray_o, OptionalTensor tangent_ray_d, at::Tensor image_sources) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(trace_reflections_jvp_optional_op(handle(scene), ray_o, ray_d, active, tape_prim_id, tape_barycentric, tape_hit_points, tape_normals, to_py_optional(tangent_vertices), to_py_optional(tangent_ray_o), to_py_optional(tangent_ray_d), image_sources));
}

OptionalTensorList trace_refl_epc_field_forward_dispatch(ScenePtr scene, at::Tensor source, at::Tensor receiver, OptionalTensor active, int64_t max_bounces) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(trace_refl_epc_field_forward_op(handle(scene), source, receiver, to_py_optional(active), max_bounces));
}

OptionalTensorList trace_refl_epc_field_backward_dispatch(ScenePtr scene, at::Tensor source, at::Tensor receiver, at::Tensor active, at::Tensor tape_prim_id, at::Tensor tape_barycentric, at::Tensor tape_t, OptionalTensor grad_field_real, OptionalTensor grad_field_imag, OptionalTensor grad_path_length, bool need_grad_vertices, bool need_grad_source, bool need_grad_receiver) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(trace_refl_epc_field_backward_op(handle(scene), source, receiver, active, tape_prim_id, tape_barycentric, tape_t, to_py_optional(grad_field_real), to_py_optional(grad_field_imag), to_py_optional(grad_path_length), need_grad_vertices, need_grad_source, need_grad_receiver));
}

OptionalTensorList trace_refl_epc_field_jvp_dispatch(ScenePtr scene, at::Tensor source, at::Tensor receiver, at::Tensor active, at::Tensor tape_prim_id, at::Tensor tape_barycentric, at::Tensor tape_t, OptionalTensor tangent_vertices, OptionalTensor tangent_source, OptionalTensor tangent_receiver) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(trace_refl_epc_field_jvp_op(handle(scene), source, receiver, active, tape_prim_id, tape_barycentric, tape_t, to_py_optional(tangent_vertices), to_py_optional(tangent_source), to_py_optional(tangent_receiver)));
}

OptionalTensorList reflection_dedup_forward_dispatch(at::Tensor bounce_count, at::Tensor shape_ids, at::Tensor prim_ids, at::Tensor t, at::Tensor bary_u, at::Tensor bary_v, at::Tensor hit_x, at::Tensor hit_y, at::Tensor hit_z, at::Tensor norm_x, at::Tensor norm_y, at::Tensor norm_z, at::Tensor img_x, at::Tensor img_y, at::Tensor img_z, int64_t max_bounces, double image_source_tolerance) {
    py::gil_scoped_acquire gil;
    py::tuple result = reflection_dedup_forward_op(bounce_count, shape_ids, prim_ids, t, bary_u, bary_v, hit_x, hit_y, hit_z, norm_x, norm_y, norm_z, img_x, img_y, img_z, max_bounces, image_source_tolerance);
    OptionalTensorList out = tuple_to_optional_tensor_list(result, 1);
    out.insert(out.begin(), at::scalar_tensor(result[0].cast<int64_t>(), bounce_count.options().dtype(at::kLong)));
    return out;
}

OptionalTensorList reflection_accumulation_forward_dispatch(ScenePtr scene, at::Tensor ray_o, at::Tensor ray_d, at::Tensor ray_tmax, at::Tensor active, at::Tensor tx, at::Tensor tx_pol, int64_t max_bounces, int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double wavelength) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(reflection_accumulation_forward_op(handle(scene), ray_o, ray_d, ray_tmax, active, tx, tx_pol, max_bounces, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min, grid_coord1_max, grid_resolution0, grid_resolution1, wavelength));
}

OptionalTensorList reflection_trace_stats_dispatch(at::Tensor valid, at::Tensor t) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(reflection_trace_stats_op(valid, t));
}

OptionalTensorList diffraction_path_stats_dispatch(at::Tensor count, at::Tensor valid, at::Tensor delay) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(diffraction_path_stats_op(count, valid, delay));
}

OptionalTensorList default_dfr_material_dispatch(int64_t count, at::Tensor like) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(default_dfr_material_op(count, like));
}

OptionalTensorList camera_sample_ray_dispatch(at::Tensor sample, double tan_x, double tan_y) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(camera_sample_ray_op(sample, tan_x, tan_y));
}

OptionalTensor pack_scene_vertex_tangents_dispatch(
    ScenePtr scene,
    std::vector<OptionalTensor> tangents) {
    at::Tensor packed = pack_scene_vertex_tangents(scene, std::move(tangents));
    if (!packed.defined())
        return c10::nullopt;
    return packed;
}

// Diffraction wrappers keep c10::optional arguments directly; only the scene handle is bridged.
OptionalTensorList diffraction_paths_order1_forward_dispatch(ScenePtr scene, at::Tensor tx_pos, at::Tensor rx_pos, OptionalTensor active, at::Tensor state_edge_index, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_n0, at::Tensor state_n1, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power, at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit, int64_t capacity, double wavelength) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(diffraction_paths_order1_forward_op(handle(scene), tx_pos, rx_pos, active, state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min, state_edge_t_max, state_n0, state_n1, state_prim0, state_prim1, state_exterior_angle, state_src, state_src_power, material_gain, material_valid, state_limit, capacity, wavelength));
}

OptionalTensorList diffraction_accumulation_forward_dispatch(ScenePtr scene, OptionalTensor active, at::Tensor state_edge_index, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_n0, at::Tensor state_n1, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power, OptionalTensor state_wi, OptionalTensor state_d0, at::Tensor material_eta_r, at::Tensor material_sigma, at::Tensor material_mu_r, at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit, int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area, double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed, int64_t max_order, int64_t recursive_state_limit, OptionalTensor recursive_active, OptionalTensor recursive_state_edge_index, OptionalTensor recursive_state_edge_pos, OptionalTensor recursive_state_edge_dir, OptionalTensor recursive_state_edge_t_min, OptionalTensor recursive_state_edge_t_max, OptionalTensor recursive_state_n0, OptionalTensor recursive_state_n1, OptionalTensor recursive_state_prim0, OptionalTensor recursive_state_prim1, OptionalTensor recursive_state_exterior_angle, int64_t export_tape) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(diffraction_accumulation_forward_op(handle(scene), active, state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min, state_edge_t_max, state_n0, state_n1, state_prim0, state_prim1, state_exterior_angle, state_src, state_src_power, state_wi, state_d0, material_eta_r, material_sigma, material_mu_r, material_gain, material_valid, state_limit, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min, grid_coord1_max, grid_resolution0, grid_resolution1, grid_cell_area, wavelength, direct_samples, keller_samples, suffix_samples, seed, max_order, recursive_state_limit, recursive_active, recursive_state_edge_index, recursive_state_edge_pos, recursive_state_edge_dir, recursive_state_edge_t_min, recursive_state_edge_t_max, recursive_state_n0, recursive_state_n1, recursive_state_prim0, recursive_state_prim1, recursive_state_exterior_angle, export_tape));
}

OptionalTensorList diffraction_accumulation_direct_backward_dispatch(ScenePtr scene, at::Tensor tape_active, at::Tensor tape_state_idx, at::Tensor tape_cell, at::Tensor tape_material_idx, at::Tensor tape_edge_u, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power, OptionalTensor state_wi, at::Tensor material_gain, at::Tensor material_valid, int64_t state_count, int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area, double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed, OptionalTensor grad_power, OptionalTensor grad_field_x_re) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(diffraction_accumulation_direct_backward_op(handle(scene), tape_active, tape_state_idx, tape_cell, tape_material_idx, tape_edge_u, state_edge_pos, state_edge_dir, state_edge_t_min, state_edge_t_max, state_prim0, state_prim1, state_exterior_angle, state_src, state_src_power, state_wi, material_gain, material_valid, state_count, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min, grid_coord1_max, grid_resolution0, grid_resolution1, grid_cell_area, wavelength, direct_samples, keller_samples, suffix_samples, seed, grad_power, grad_field_x_re));
}

OptionalTensorList diffraction_accumulation_direct_jvp_dispatch(ScenePtr scene, at::Tensor tape_active, at::Tensor tape_state_idx, at::Tensor tape_cell, at::Tensor tape_material_idx, at::Tensor tape_edge_u, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power, OptionalTensor state_wi, at::Tensor material_gain, at::Tensor material_valid, int64_t state_count, int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area, double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed, OptionalTensor dot_state_edge_pos, OptionalTensor dot_state_edge_dir, OptionalTensor dot_state_edge_t_min, OptionalTensor dot_state_edge_t_max, OptionalTensor dot_state_exterior_angle, OptionalTensor dot_state_src, OptionalTensor dot_state_src_power, OptionalTensor dot_state_wi, OptionalTensor dot_material_gain) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(diffraction_accumulation_direct_jvp_op(handle(scene), tape_active, tape_state_idx, tape_cell, tape_material_idx, tape_edge_u, state_edge_pos, state_edge_dir, state_edge_t_min, state_edge_t_max, state_prim0, state_prim1, state_exterior_angle, state_src, state_src_power, state_wi, material_gain, material_valid, state_count, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min, grid_coord1_max, grid_resolution0, grid_resolution1, grid_cell_area, wavelength, direct_samples, keller_samples, suffix_samples, seed, dot_state_edge_pos, dot_state_edge_dir, dot_state_edge_t_min, dot_state_edge_t_max, dot_state_exterior_angle, dot_state_src, dot_state_src_power, dot_state_wi, dot_material_gain));
}

OptionalTensorList diffraction_accumulation_chain_backward_dispatch(ScenePtr scene, at::Tensor tape_active, at::Tensor tape_cell, at::Tensor state_edge_index, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power, at::Tensor recursive_state_edge_index, at::Tensor recursive_state_edge_pos, at::Tensor recursive_state_edge_dir, at::Tensor recursive_state_edge_t_min, at::Tensor recursive_state_edge_t_max, at::Tensor recursive_state_prim0, at::Tensor recursive_state_prim1, at::Tensor recursive_state_exterior_angle, at::Tensor material_gain, at::Tensor material_valid, int64_t state_count, int64_t recursive_state_count, int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area, double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed, int64_t max_order, OptionalTensor grad_power, OptionalTensor grad_field_x_re) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(diffraction_accumulation_chain_backward_op(handle(scene), tape_active, tape_cell, state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min, state_edge_t_max, state_prim0, state_prim1, state_exterior_angle, state_src, state_src_power, recursive_state_edge_index, recursive_state_edge_pos, recursive_state_edge_dir, recursive_state_edge_t_min, recursive_state_edge_t_max, recursive_state_prim0, recursive_state_prim1, recursive_state_exterior_angle, material_gain, material_valid, state_count, recursive_state_count, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min, grid_coord1_max, grid_resolution0, grid_resolution1, grid_cell_area, wavelength, direct_samples, keller_samples, suffix_samples, seed, max_order, grad_power, grad_field_x_re));
}

OptionalTensorList diffraction_accumulation_chain_jvp_dispatch(ScenePtr scene, at::Tensor tape_active, at::Tensor tape_cell, at::Tensor state_edge_index, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power, at::Tensor recursive_state_edge_index, at::Tensor recursive_state_edge_pos, at::Tensor recursive_state_edge_dir, at::Tensor recursive_state_edge_t_min, at::Tensor recursive_state_edge_t_max, at::Tensor recursive_state_prim0, at::Tensor recursive_state_prim1, at::Tensor recursive_state_exterior_angle, at::Tensor material_gain, at::Tensor material_valid, int64_t state_count, int64_t recursive_state_count, int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area, double wavelength, int64_t direct_samples, int64_t keller_samples, int64_t suffix_samples, int64_t seed, int64_t max_order, OptionalTensor dot_state_edge_pos, OptionalTensor dot_state_edge_dir, OptionalTensor dot_state_edge_t_min, OptionalTensor dot_state_edge_t_max, OptionalTensor dot_state_exterior_angle, OptionalTensor dot_state_src, OptionalTensor dot_state_src_power, OptionalTensor dot_recursive_state_edge_pos, OptionalTensor dot_recursive_state_edge_dir, OptionalTensor dot_recursive_state_edge_t_min, OptionalTensor dot_recursive_state_edge_t_max, OptionalTensor dot_recursive_state_exterior_angle, OptionalTensor dot_material_gain) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(diffraction_accumulation_chain_jvp_op(handle(scene), tape_active, tape_cell, state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min, state_edge_t_max, state_prim0, state_prim1, state_exterior_angle, state_src, state_src_power, recursive_state_edge_index, recursive_state_edge_pos, recursive_state_edge_dir, recursive_state_edge_t_min, recursive_state_edge_t_max, recursive_state_prim0, recursive_state_prim1, recursive_state_exterior_angle, material_gain, material_valid, state_count, recursive_state_count, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min, grid_coord1_max, grid_resolution0, grid_resolution1, grid_cell_area, wavelength, direct_samples, keller_samples, suffix_samples, seed, max_order, dot_state_edge_pos, dot_state_edge_dir, dot_state_edge_t_min, dot_state_edge_t_max, dot_state_exterior_angle, dot_state_src, dot_state_src_power, dot_recursive_state_edge_pos, dot_recursive_state_edge_dir, dot_recursive_state_edge_t_min, dot_recursive_state_edge_t_max, dot_recursive_state_exterior_angle, dot_material_gain));
}

OptionalTensorList diffraction_coherent_accumulation_forward_dispatch(ScenePtr scene, OptionalTensor active, at::Tensor state_edge_index, at::Tensor state_edge_pos, at::Tensor state_edge_dir, at::Tensor state_edge_t_min, at::Tensor state_edge_t_max, at::Tensor state_n0, at::Tensor state_n1, at::Tensor state_prim0, at::Tensor state_prim1, at::Tensor state_exterior_angle, at::Tensor state_src, at::Tensor state_src_power, OptionalTensor state_wi, OptionalTensor state_d0, at::Tensor material_eta_r, at::Tensor material_sigma, at::Tensor material_mu_r, at::Tensor material_gain, at::Tensor material_valid, int64_t state_limit, int64_t grid_axis, double grid_position, double grid_coord0_min, double grid_coord0_max, double grid_coord1_min, double grid_coord1_max, int64_t grid_resolution0, int64_t grid_resolution1, double grid_cell_area, double wavelength, bool select_diffraction_point, bool prefilter_visibility) {
    py::gil_scoped_acquire gil;
    return tuple_to_optional_tensor_list(diffraction_coherent_accumulation_forward_op(handle(scene), active, state_edge_index, state_edge_pos, state_edge_dir, state_edge_t_min, state_edge_t_max, state_n0, state_n1, state_prim0, state_prim1, state_exterior_angle, state_src, state_src_power, state_wi, state_d0, material_eta_r, material_sigma, material_mu_r, material_gain, material_valid, state_limit, grid_axis, grid_position, grid_coord0_min, grid_coord0_max, grid_coord1_min, grid_coord1_max, grid_resolution0, grid_resolution1, grid_cell_area, wavelength, select_diffraction_point, prefilter_visibility));
}

#define RAYDN_SCHEMA_SCENE "__torch__.torch.classes.raydn.Scene"

TORCH_LIBRARY(raydn, m) {
    m.class_<SceneHandle>("Scene")
        .def(torch::init([](
            std::vector<at::Tensor> vertices,
            std::vector<at::Tensor> faces,
            std::vector<at::Tensor> uv,
            std::vector<at::Tensor> face_uv,
            std::vector<at::Tensor> to_world_left,
            std::vector<at::Tensor> to_world_right,
            std::vector<int64_t> mesh_flags) {
            return create_scene_cache_from_flat(
                std::move(vertices),
                std::move(faces),
                std::move(uv),
                std::move(face_uv),
                std::move(to_world_left),
                std::move(to_world_right),
                std::move(mesh_flags));
        }))
        .def("update_vertices", [](const ScenePtr &scene, int64_t mesh_id, at::Tensor vertices) {
            update_mesh_vertices(scene, mesh_id, vertices);
        })
        .def("sync", [](const ScenePtr &scene) {
            sync_scene(scene);
        })
        .def("version", [](const ScenePtr &scene) {
            return scene_version(scene);
        })
        .def("num_meshes", [](const ScenePtr &scene) {
            return scene_num_meshes(scene);
        })
        .def("edge_count", [](const ScenePtr &scene) {
            return scene_edge_count(scene);
        })
        .def("handle", [](const ScenePtr &scene) {
            return scene->handle;
        });
}

TORCH_LIBRARY_FRAGMENT(raydn, m) {
    m.def("split_scene_vertex_grad(" RAYDN_SCHEMA_SCENE " scene, Tensor grad_vertices) -> Tensor[]");
    m.def("pack_scene_vertex_tangents(" RAYDN_SCHEMA_SCENE " scene, Tensor?[] tangents) -> Tensor?");
    m.def("intersect_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active) -> Tensor?[]");
    m.def("intersect_forward_flags(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active, int flags) -> Tensor?[]");
    m.def("intersect_forward_t(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active) -> Tensor");
    m.def("intersect_forward_ad_flags(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active, int flags) -> Tensor?[]");
    m.def("intersect_ad_t(" RAYDN_SCHEMA_SCENE " scene, Tensor vertices, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active) -> Tensor");
    m.def("intersect_ad_flags(" RAYDN_SCHEMA_SCENE " scene, Tensor vertices, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active, int flags) -> Tensor?[]");
    m.def("intersection_empty_fields(" RAYDN_SCHEMA_SCENE " scene, Tensor like) -> Tensor?[]");
    m.def("intersect_backward_optional(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor active, Tensor tape_prim_id, Tensor tape_barycentric, Tensor? grad_t, Tensor? grad_p, Tensor? grad_n, Tensor? grad_geo_n, Tensor? grad_uv, Tensor? grad_barycentric, bool need_grad_vertices, bool need_grad_ray_o, bool need_grad_ray_d, bool need_grad_ray_tmax) -> Tensor?[]");
    m.def("intersect_backward_t(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor active, Tensor tape_prim_id, Tensor tape_barycentric, Tensor grad_t, bool need_grad_vertices, bool need_grad_ray_o, bool need_grad_ray_d, bool need_grad_ray_tmax) -> Tensor?[]");
    m.def("intersect_jvp_optional(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor active, Tensor tape_prim_id, Tensor tape_barycentric, Tensor? tangent_vertices, Tensor? tangent_ray_o, Tensor? tangent_ray_d, int flags) -> Tensor?[]");

    m.def("nearest_edge_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor point) -> Tensor?[]");
    m.def("nearest_edge_forward_noad(" RAYDN_SCHEMA_SCENE " scene, Tensor point) -> Tensor?[]");
    m.def("nearest_edge_ray_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor active) -> Tensor?[]");
    m.def("nearest_edge_backward_optional(" RAYDN_SCHEMA_SCENE " scene, Tensor point, Tensor tape_edge_id, Tensor tape_s, Tensor tape_d, Tensor? grad_distance, Tensor? grad_edge_point, Tensor? grad_edge_t, Tensor? grad_edge_t_alias) -> Tensor?[]");
    m.def("nearest_edge_jvp_optional(" RAYDN_SCHEMA_SCENE " scene, Tensor point, Tensor tape_edge_id, Tensor tape_s, Tensor tape_d, Tensor? tangent_vertices, Tensor? tangent_point) -> Tensor?[]");

    m.def("visibility_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor start, Tensor end, Tensor? active) -> Tensor?[]");
    m.def("trace_reflections_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active, int max_bounces) -> Tensor?[]");
    m.def("trace_reflections_forward_noad(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active, int max_bounces) -> Tensor?[]");
    m.def("trace_reflections_forward_reduced(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor? active, int max_bounces) -> Tensor?[]");
    m.def("trace_reflections_backward_optional(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor active, Tensor tape_prim_id, Tensor tape_barycentric, Tensor tape_hit_points, Tensor tape_normals, Tensor image_sources, Tensor? grad_t, Tensor? grad_image_sources) -> Tensor?[]");
    m.def("trace_reflections_jvp_optional(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor active, Tensor tape_prim_id, Tensor tape_barycentric, Tensor tape_hit_points, Tensor tape_normals, Tensor? tangent_vertices, Tensor? tangent_ray_o, Tensor? tangent_ray_d, Tensor image_sources) -> Tensor?[]");
    m.def("trace_refl_epc_field_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor source, Tensor receiver, Tensor? active, int max_bounces) -> Tensor?[]");
    m.def("trace_refl_epc_field_backward(" RAYDN_SCHEMA_SCENE " scene, Tensor source, Tensor receiver, Tensor active, Tensor tape_prim_id, Tensor tape_barycentric, Tensor tape_t, Tensor? grad_field_real, Tensor? grad_field_imag, Tensor? grad_path_length, bool need_grad_vertices, bool need_grad_source, bool need_grad_receiver) -> Tensor?[]");
    m.def("trace_refl_epc_field_jvp(" RAYDN_SCHEMA_SCENE " scene, Tensor source, Tensor receiver, Tensor active, Tensor tape_prim_id, Tensor tape_barycentric, Tensor tape_t, Tensor? tangent_vertices, Tensor? tangent_source, Tensor? tangent_receiver) -> Tensor?[]");
    m.def("reflection_dedup_forward(Tensor bounce_count, Tensor shape_ids, Tensor prim_ids, Tensor t, Tensor bary_u, Tensor bary_v, Tensor hit_x, Tensor hit_y, Tensor hit_z, Tensor norm_x, Tensor norm_y, Tensor norm_z, Tensor img_x, Tensor img_y, Tensor img_z, int max_bounces, float image_source_tolerance) -> Tensor?[]");
    m.def("reflection_accumulation_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor ray_o, Tensor ray_d, Tensor ray_tmax, Tensor active, Tensor tx, Tensor tx_pol, int max_bounces, int grid_axis, float grid_position, float grid_coord0_min, float grid_coord0_max, float grid_coord1_min, float grid_coord1_max, int grid_resolution0, int grid_resolution1, float wavelength) -> Tensor?[]");

    m.def("reflection_trace_stats(Tensor valid, Tensor t) -> Tensor?[]");
    m.def("diffraction_path_stats(Tensor count, Tensor valid, Tensor delay) -> Tensor?[]");
    m.def("default_dfr_material(int count, Tensor like) -> Tensor?[]");
    m.def("intersection_valid(Tensor t, Tensor shape_id) -> Tensor");
    m.def("camera_sample_to_world(Tensor sample, float tan_x, float tan_y, float depth) -> Tensor");
    m.def("camera_sample_to_world_backward(Tensor grad_world, int sample_count, float tan_x, float tan_y, float depth) -> Tensor");
    m.def("camera_world_to_sample(Tensor point, float tan_x, float tan_y) -> Tensor");
    m.def("camera_world_to_sample_backward(Tensor point, Tensor grad_sample, float tan_x, float tan_y) -> Tensor");
    m.def("camera_sample_ray(Tensor sample, float tan_x, float tan_y) -> Tensor?[]");
    m.def("camera_sample_ray_backward(Tensor sample, Tensor? grad_direction, float tan_x, float tan_y) -> Tensor");

    m.def("diffraction_paths_order1_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor tx_pos, Tensor rx_pos, Tensor? active, Tensor state_edge_index, Tensor state_edge_pos, Tensor state_edge_dir, Tensor state_edge_t_min, Tensor state_edge_t_max, Tensor state_n0, Tensor state_n1, Tensor state_prim0, Tensor state_prim1, Tensor state_exterior_angle, Tensor state_src, Tensor state_src_power, Tensor material_gain, Tensor material_valid, int state_limit, int capacity, float wavelength) -> Tensor?[]");
    m.def("diffraction_accumulation_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor? active, Tensor state_edge_index, Tensor state_edge_pos, Tensor state_edge_dir, Tensor state_edge_t_min, Tensor state_edge_t_max, Tensor state_n0, Tensor state_n1, Tensor state_prim0, Tensor state_prim1, Tensor state_exterior_angle, Tensor state_src, Tensor state_src_power, Tensor? state_wi, Tensor? state_d0, Tensor material_eta_r, Tensor material_sigma, Tensor material_mu_r, Tensor material_gain, Tensor material_valid, int state_limit, int grid_axis, float grid_position, float grid_coord0_min, float grid_coord0_max, float grid_coord1_min, float grid_coord1_max, int grid_resolution0, int grid_resolution1, float grid_cell_area, float wavelength, int direct_samples, int keller_samples, int suffix_samples, int seed, int max_order, int recursive_state_limit, Tensor? recursive_active, Tensor? recursive_state_edge_index, Tensor? recursive_state_edge_pos, Tensor? recursive_state_edge_dir, Tensor? recursive_state_edge_t_min, Tensor? recursive_state_edge_t_max, Tensor? recursive_state_n0, Tensor? recursive_state_n1, Tensor? recursive_state_prim0, Tensor? recursive_state_prim1, Tensor? recursive_state_exterior_angle, int export_tape) -> Tensor?[]");
    m.def("diffraction_accumulation_direct_backward(" RAYDN_SCHEMA_SCENE " scene, Tensor tape_active, Tensor tape_state_idx, Tensor tape_cell, Tensor tape_material_idx, Tensor tape_edge_u, Tensor state_edge_pos, Tensor state_edge_dir, Tensor state_edge_t_min, Tensor state_edge_t_max, Tensor state_prim0, Tensor state_prim1, Tensor state_exterior_angle, Tensor state_src, Tensor state_src_power, Tensor? state_wi, Tensor material_gain, Tensor material_valid, int state_count, int grid_axis, float grid_position, float grid_coord0_min, float grid_coord0_max, float grid_coord1_min, float grid_coord1_max, int grid_resolution0, int grid_resolution1, float grid_cell_area, float wavelength, int direct_samples, int keller_samples, int suffix_samples, int seed, Tensor? grad_power, Tensor? grad_field_x_re) -> Tensor?[]");
    m.def("diffraction_accumulation_direct_jvp(" RAYDN_SCHEMA_SCENE " scene, Tensor tape_active, Tensor tape_state_idx, Tensor tape_cell, Tensor tape_material_idx, Tensor tape_edge_u, Tensor state_edge_pos, Tensor state_edge_dir, Tensor state_edge_t_min, Tensor state_edge_t_max, Tensor state_prim0, Tensor state_prim1, Tensor state_exterior_angle, Tensor state_src, Tensor state_src_power, Tensor? state_wi, Tensor material_gain, Tensor material_valid, int state_count, int grid_axis, float grid_position, float grid_coord0_min, float grid_coord0_max, float grid_coord1_min, float grid_coord1_max, int grid_resolution0, int grid_resolution1, float grid_cell_area, float wavelength, int direct_samples, int keller_samples, int suffix_samples, int seed, Tensor? dot_state_edge_pos, Tensor? dot_state_edge_dir, Tensor? dot_state_edge_t_min, Tensor? dot_state_edge_t_max, Tensor? dot_state_exterior_angle, Tensor? dot_state_src, Tensor? dot_state_src_power, Tensor? dot_state_wi, Tensor? dot_material_gain) -> Tensor?[]");
    m.def("diffraction_accumulation_chain_backward(" RAYDN_SCHEMA_SCENE " scene, Tensor tape_active, Tensor tape_cell, Tensor state_edge_index, Tensor state_edge_pos, Tensor state_edge_dir, Tensor state_edge_t_min, Tensor state_edge_t_max, Tensor state_prim0, Tensor state_prim1, Tensor state_exterior_angle, Tensor state_src, Tensor state_src_power, Tensor recursive_state_edge_index, Tensor recursive_state_edge_pos, Tensor recursive_state_edge_dir, Tensor recursive_state_edge_t_min, Tensor recursive_state_edge_t_max, Tensor recursive_state_prim0, Tensor recursive_state_prim1, Tensor recursive_state_exterior_angle, Tensor material_gain, Tensor material_valid, int state_count, int recursive_state_count, int grid_axis, float grid_position, float grid_coord0_min, float grid_coord0_max, float grid_coord1_min, float grid_coord1_max, int grid_resolution0, int grid_resolution1, float grid_cell_area, float wavelength, int direct_samples, int keller_samples, int suffix_samples, int seed, int max_order, Tensor? grad_power, Tensor? grad_field_x_re) -> Tensor?[]");
    m.def("diffraction_accumulation_chain_jvp(" RAYDN_SCHEMA_SCENE " scene, Tensor tape_active, Tensor tape_cell, Tensor state_edge_index, Tensor state_edge_pos, Tensor state_edge_dir, Tensor state_edge_t_min, Tensor state_edge_t_max, Tensor state_prim0, Tensor state_prim1, Tensor state_exterior_angle, Tensor state_src, Tensor state_src_power, Tensor recursive_state_edge_index, Tensor recursive_state_edge_pos, Tensor recursive_state_edge_dir, Tensor recursive_state_edge_t_min, Tensor recursive_state_edge_t_max, Tensor recursive_state_prim0, Tensor recursive_state_prim1, Tensor recursive_state_exterior_angle, Tensor material_gain, Tensor material_valid, int state_count, int recursive_state_count, int grid_axis, float grid_position, float grid_coord0_min, float grid_coord0_max, float grid_coord1_min, float grid_coord1_max, int grid_resolution0, int grid_resolution1, float grid_cell_area, float wavelength, int direct_samples, int keller_samples, int suffix_samples, int seed, int max_order, Tensor? dot_state_edge_pos, Tensor? dot_state_edge_dir, Tensor? dot_state_edge_t_min, Tensor? dot_state_edge_t_max, Tensor? dot_state_exterior_angle, Tensor? dot_state_src, Tensor? dot_state_src_power, Tensor? dot_recursive_state_edge_pos, Tensor? dot_recursive_state_edge_dir, Tensor? dot_recursive_state_edge_t_min, Tensor? dot_recursive_state_edge_t_max, Tensor? dot_recursive_state_exterior_angle, Tensor? dot_material_gain) -> Tensor?[]");
    m.def("diffraction_coherent_accumulation_forward(" RAYDN_SCHEMA_SCENE " scene, Tensor? active, Tensor state_edge_index, Tensor state_edge_pos, Tensor state_edge_dir, Tensor state_edge_t_min, Tensor state_edge_t_max, Tensor state_n0, Tensor state_n1, Tensor state_prim0, Tensor state_prim1, Tensor state_exterior_angle, Tensor state_src, Tensor state_src_power, Tensor? state_wi, Tensor? state_d0, Tensor material_eta_r, Tensor material_sigma, Tensor material_mu_r, Tensor material_gain, Tensor material_valid, int state_limit, int grid_axis, float grid_position, float grid_coord0_min, float grid_coord0_max, float grid_coord1_min, float grid_coord1_max, int grid_resolution0, int grid_resolution1, float grid_cell_area, float wavelength, bool select_diffraction_point, bool prefilter_visibility) -> Tensor?[]");
}

TORCH_LIBRARY_IMPL(raydn, CUDA, m) {
    m.impl("split_scene_vertex_grad", TORCH_FN(split_scene_vertex_grad));
    m.impl("pack_scene_vertex_tangents", TORCH_FN(pack_scene_vertex_tangents_dispatch));
    m.impl("intersect_forward", TORCH_FN(intersect_forward_dispatch));
    m.impl("intersect_forward_flags", TORCH_FN(intersect_forward_flags_dispatch));
    m.impl("intersect_forward_t", TORCH_FN(intersect_forward_t_dispatch));
    m.impl("intersect_forward_ad_flags", TORCH_FN(intersect_forward_ad_flags_dispatch));
    m.impl("intersect_ad_t", TORCH_FN(intersect_ad_t_dispatch));
    m.impl("intersect_ad_flags", TORCH_FN(intersect_ad_flags_dispatch));
    m.impl("intersection_empty_fields", TORCH_FN(intersection_empty_fields_dispatch));
    m.impl("intersect_backward_optional", TORCH_FN(intersect_backward_optional_dispatch));
    m.impl("intersect_backward_t", TORCH_FN(intersect_backward_t_dispatch));
    m.impl("intersect_jvp_optional", TORCH_FN(intersect_jvp_optional_dispatch));
    m.impl("nearest_edge_forward", TORCH_FN(nearest_edge_forward_dispatch));
    m.impl("nearest_edge_forward_noad", TORCH_FN(nearest_edge_forward_noad_dispatch));
    m.impl("nearest_edge_ray_forward", TORCH_FN(nearest_edge_ray_forward_dispatch));
    m.impl("nearest_edge_backward_optional", TORCH_FN(nearest_edge_backward_optional_dispatch));
    m.impl("nearest_edge_jvp_optional", TORCH_FN(nearest_edge_jvp_optional_dispatch));
    m.impl("visibility_forward", TORCH_FN(visibility_forward_dispatch));
    m.impl("trace_reflections_forward", TORCH_FN(trace_reflections_forward_dispatch));
    m.impl("trace_reflections_forward_noad", TORCH_FN(trace_reflections_forward_noad_dispatch));
    m.impl("trace_reflections_forward_reduced", TORCH_FN(trace_reflections_forward_reduced_dispatch));
    m.impl("trace_reflections_backward_optional", TORCH_FN(trace_reflections_backward_optional_dispatch));
    m.impl("trace_reflections_jvp_optional", TORCH_FN(trace_reflections_jvp_optional_dispatch));
    m.impl("trace_refl_epc_field_forward", TORCH_FN(trace_refl_epc_field_forward_dispatch));
    m.impl("trace_refl_epc_field_backward", TORCH_FN(trace_refl_epc_field_backward_dispatch));
    m.impl("trace_refl_epc_field_jvp", TORCH_FN(trace_refl_epc_field_jvp_dispatch));
    m.impl("reflection_dedup_forward", TORCH_FN(reflection_dedup_forward_dispatch));
    m.impl("reflection_accumulation_forward", TORCH_FN(reflection_accumulation_forward_dispatch));
    m.impl("reflection_trace_stats", TORCH_FN(reflection_trace_stats_dispatch));
    m.impl("diffraction_path_stats", TORCH_FN(diffraction_path_stats_dispatch));
    m.impl("default_dfr_material", TORCH_FN(default_dfr_material_dispatch));
    m.impl("intersection_valid", TORCH_FN(intersection_valid_op));
    m.impl("camera_sample_to_world", TORCH_FN(camera_sample_to_world_op));
    m.impl("camera_sample_to_world_backward", TORCH_FN(camera_sample_to_world_backward_op));
    m.impl("camera_world_to_sample", TORCH_FN(camera_world_to_sample_op));
    m.impl("camera_world_to_sample_backward", TORCH_FN(camera_world_to_sample_backward_op));
    m.impl("camera_sample_ray", TORCH_FN(camera_sample_ray_dispatch));
    m.impl("camera_sample_ray_backward", TORCH_FN(camera_sample_ray_backward_op));
    m.impl("diffraction_paths_order1_forward", TORCH_FN(diffraction_paths_order1_forward_dispatch));
    m.impl("diffraction_accumulation_forward", TORCH_FN(diffraction_accumulation_forward_dispatch));
    m.impl("diffraction_accumulation_direct_backward", TORCH_FN(diffraction_accumulation_direct_backward_dispatch));
    m.impl("diffraction_accumulation_direct_jvp", TORCH_FN(diffraction_accumulation_direct_jvp_dispatch));
    m.impl("diffraction_accumulation_chain_backward", TORCH_FN(diffraction_accumulation_chain_backward_dispatch));
    m.impl("diffraction_accumulation_chain_jvp", TORCH_FN(diffraction_accumulation_chain_jvp_dispatch));
    m.impl("diffraction_coherent_accumulation_forward", TORCH_FN(diffraction_coherent_accumulation_forward_dispatch));
}

} // namespace raydn
