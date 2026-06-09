#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <raydtorch/diffraction/accum_params.h>
#include <raydtorch/diffraction/accum_ad.h>
#include <raydtorch/diffraction/accum_reduce.h>
#include <raydtorch/diffraction/paths_params.h>
#include <raydtorch/diffraction/pipeline.h>
#include <raydtorch/scene/geometry_kernels.h>
#include <raydtorch/common/optix_pipeline.h>
#include <raydtorch/reflection/kernels.h>
#include <raydtorch/reflection/pipeline.h>
#include <raydtorch/common/optix_context.h>
#include <raydtorch/reflection/accum_params.h>
#include <raydtorch/reflection/dedup.h>
#include <raydtorch/reflection/epc_field.h>
#include <raydtorch/reflection/epc_params.h>
#include <raydtorch/reflection/trace_params.h>
#include <raydtorch/reflection/visibility_params.h>
#include <raydtorch/scene/cache.h>
#include <raydtorch/common/tensor_check.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace raydtorch {

namespace {

constexpr int64_t kStagedDfrAccumMinSamples = 2048;
constexpr int64_t kStagedDfrAccumMinSamplesPerCell = 4;

void require_same_batch(const at::Tensor &a, const at::Tensor &b, const char *name) {
    if (a.size(0) != b.size(0))
        throw std::runtime_error(std::string(name) + " tensors must have the same batch size.");
}

void require_flat_i32(const at::Tensor &tensor, const char *name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kInt, name);
    require_rank(tensor, 1, name);
}

void require_flat_f32(const at::Tensor &tensor, const char *name) {
    require_cuda(tensor, name);
    require_contiguous(tensor, name);
    require_dtype(tensor, at::kFloat, name);
    require_rank(tensor, 1, name);
}

void require_state_width(const at::Tensor &tensor, int64_t state_count, const char *name) {
    if (tensor.size(0) < state_count)
        throw std::runtime_error(std::string(name) + " must cover state_count.");
}

int32_t checked_i32(int64_t value, const char *name) {
    if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error(std::string(name) + " does not fit in int32.");
    return static_cast<int32_t>(value);
}

at::Tensor active_mask_for_states(const at::Tensor &active, int64_t state_count, const char *name) {
    if (active.size(0) == state_count)
        return active.contiguous();
    if (active.size(0) == 1)
        return active.expand({state_count}).contiguous();
    throw std::runtime_error(std::string(name) + " active width must be 1 or match state_count.");
}

at::Tensor first_bounce_column(const at::Tensor &value, int64_t ray_count) {
    if (value.dim() == 1)
        return value.reshape({ray_count}).contiguous();
    return value.slice(1, 0, 1).reshape({ray_count}).contiguous();
}

struct Vec3SoA {
    at::Tensor x;
    at::Tensor y;
    at::Tensor z;
};

Vec3SoA split_vec3(const at::Tensor &value) {
    return {
        value.select(1, 0).contiguous(),
        value.select(1, 1).contiguous(),
        value.select(1, 2).contiguous(),
    };
}

struct TriangleSoA {
    at::Tensor p0_x;
    at::Tensor p0_y;
    at::Tensor p0_z;
    at::Tensor e1_x;
    at::Tensor e1_y;
    at::Tensor e1_z;
    at::Tensor e2_x;
    at::Tensor e2_y;
    at::Tensor e2_z;
    at::Tensor fn_x;
    at::Tensor fn_y;
    at::Tensor fn_z;
    at::Tensor face_offsets;
    int32_t n_triangles = 0;
};

TriangleSoA make_triangle_soa(const MeshRecord &mesh) {
    at::Tensor faces_i64 = mesh.faces.to(at::kLong);
    at::Tensor v0 = mesh.vertices.index_select(0, faces_i64.select(1, 0)).contiguous();
    at::Tensor v1 = mesh.vertices.index_select(0, faces_i64.select(1, 1)).contiguous();
    at::Tensor v2 = mesh.vertices.index_select(0, faces_i64.select(1, 2)).contiguous();
    at::Tensor e1 = (v1 - v0).contiguous();
    at::Tensor e2 = (v2 - v0).contiguous();
    at::Tensor fn = at::cross(e1, e2, 1).contiguous();
    return {
        v0.select(1, 0).contiguous(),
        v0.select(1, 1).contiguous(),
        v0.select(1, 2).contiguous(),
        e1.select(1, 0).contiguous(),
        e1.select(1, 1).contiguous(),
        e1.select(1, 2).contiguous(),
        e2.select(1, 0).contiguous(),
        e2.select(1, 1).contiguous(),
        e2.select(1, 2).contiguous(),
        fn.select(1, 0).contiguous(),
        fn.select(1, 1).contiguous(),
        fn.select(1, 2).contiguous(),
        at::zeros({1}, mesh.faces.options()),
        static_cast<int32_t>(mesh.faces.size(0)),
    };
}

TriangleSoA make_scene_triangle_soa(const SceneCache &scene) {
    return {
        scene.tri_p0_x,
        scene.tri_p0_y,
        scene.tri_p0_z,
        scene.tri_e1_x,
        scene.tri_e1_y,
        scene.tri_e1_z,
        scene.tri_e2_x,
        scene.tri_e2_y,
        scene.tri_e2_z,
        scene.tri_fn_x,
        scene.tri_fn_y,
        scene.tri_fn_z,
        scene.face_offsets.contiguous(),
        static_cast<int32_t>(scene.global_faces.size(0)),
    };
}

const uint8_t *mask_ptr(const at::Tensor &mask) {
    return reinterpret_cast<const uint8_t *>(mask.data_ptr<bool>());
}

uint8_t *mutable_mask_ptr(const at::Tensor &mask) {
    return reinterpret_cast<uint8_t *>(mask.data_ptr<bool>());
}

at::Tensor stack_vec3(const at::Tensor &x, const at::Tensor &y, const at::Tensor &z) {
    return at::stack({x, y, z}, 1).contiguous();
}

std::shared_ptr<OptixLaunchPipeline> optix_pipeline_for_scene(
    const SceneCache &scene,
    const OptixPipelineConfig &config) {
    OptixDeviceContextEntry &optix_entry = get_optix_context(static_cast<int>(scene.device_index));
    return shared_optix_launch_pipeline(
        optix_entry.optix_context,
        static_cast<int>(scene.device_index),
        1,
        config);
}

} // namespace

py::tuple diffraction_paths_order1_forward_op(
    int64_t scene_handle,
    at::Tensor tx_pos,
    at::Tensor rx_pos,
    at::Tensor active,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_n0,
    at::Tensor state_n1,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t capacity,
    double wavelength) {
    require_vec3f(tx_pos, "tx_pos");
    require_vec3f(rx_pos, "rx_pos");
    require_mask(active, "active");
    require_flat_i32(state_edge_index, "state_edge_index");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_vec3f(state_n0, "state_n0");
    require_vec3f(state_n1, "state_n1");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    if (capacity < 0)
        throw std::runtime_error("capacity must be non-negative.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");

    SceneCache &scene = get_scene(scene_handle);

    const int64_t tx_count = tx_pos.size(0);
    const int64_t rx_count = rx_pos.size(0);
    const int64_t state_count = state_edge_index.size(0);
    require_state_width(state_edge_pos, state_count, "state_edge_pos");
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_n0, state_count, "state_n0");
    require_state_width(state_n1, state_count, "state_n1");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    const int64_t material_count = material_gain.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_valid.size(0) != material_count)
        throw std::runtime_error("material_gain and material_valid must have matching widths.");

    const int64_t n_rays64 = tx_count * rx_count * state_count;
    if (n_rays64 > capacity)
        throw std::runtime_error("capacity must be at least tx_count * rx_count * state_count.");
    const int32_t n_rays = checked_i32(n_rays64, "n_rays");
    const int32_t capacity_i32 = checked_i32(capacity, "capacity");

    auto fopts = tx_pos.options();
    auto iopts = state_edge_index.options();
    at::Tensor out_count = at::zeros({1}, iopts);
    at::Tensor out_valid = at::zeros({capacity}, active.options());
    at::Tensor out_tx_id = at::full({capacity}, -1, iopts);
    at::Tensor out_rx_id = at::full({capacity}, -1, iopts);
    at::Tensor out_order = at::zeros({capacity}, iopts);
    at::Tensor out_edge0 = at::full({capacity}, -1, iopts);
    at::Tensor out_edge1 = at::full({capacity}, -1, iopts);
    at::Tensor out_edge2 = at::full({capacity}, -1, iopts);
    at::Tensor out_delay = at::zeros({capacity}, fopts);
    at::Tensor out_field_x_re = at::zeros({capacity}, fopts);
    at::Tensor out_field_x_im = at::zeros({capacity}, fopts);
    at::Tensor out_field_y_re = at::zeros({capacity}, fopts);
    at::Tensor out_field_y_im = at::zeros({capacity}, fopts);
    at::Tensor out_field_z_re = at::zeros({capacity}, fopts);
    at::Tensor out_field_z_im = at::zeros({capacity}, fopts);
    at::Tensor out_p0_x = at::zeros({capacity}, fopts);
    at::Tensor out_p0_y = at::zeros({capacity}, fopts);
    at::Tensor out_p0_z = at::zeros({capacity}, fopts);
    at::Tensor out_p1 = at::zeros({capacity, 3}, fopts);
    at::Tensor out_p2 = at::zeros({capacity, 3}, fopts);
    if (n_rays == 0 || capacity_i32 == 0) {
        return py::make_tuple(
            out_count,
            out_valid,
            out_tx_id,
            out_rx_id,
            out_order,
            out_edge0,
            out_edge1,
            out_edge2,
            out_delay,
            out_field_x_re,
            out_field_x_im,
            out_field_y_re,
            out_field_y_im,
            out_field_z_re,
            out_field_z_im,
            at::stack({out_p0_x, out_p0_y, out_p0_z}, 1).contiguous(),
            out_p1,
            out_p2);
    }

    Vec3SoA tx_soa = split_vec3(tx_pos);
    Vec3SoA rx_soa = split_vec3(rx_pos);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_n0_soa = split_vec3(state_n0);
    Vec3SoA state_n1_soa = split_vec3(state_n1);
    Vec3SoA state_src_soa = split_vec3(state_src);
    at::Tensor active_contig = active_mask_for_states(active, state_count, "diffraction_paths_order1_forward");

    DfrPathParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = n_rays;
    params.capacity = capacity_i32;
    params.tx_pos_x = tx_soa.x.data_ptr<float>();
    params.tx_pos_y = tx_soa.y.data_ptr<float>();
    params.tx_pos_z = tx_soa.z.data_ptr<float>();
    params.tx_count = checked_i32(tx_count, "tx_count");
    params.rx_pos_x = rx_soa.x.data_ptr<float>();
    params.rx_pos_y = rx_soa.y.data_ptr<float>();
    params.rx_pos_z = rx_soa.z.data_ptr<float>();
    params.rx_count = checked_i32(rx_count, "rx_count");
    params.active_mask = mask_ptr(active_contig);
    params.active_width = checked_i32(state_count, "active_width");
    params.state_count = checked_i32(state_count, "state_count");
    params.state_limit = checked_i32(state_count, "state_limit");
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_n0_x = state_n0_soa.x.data_ptr<float>();
    params.state_n0_y = state_n0_soa.y.data_ptr<float>();
    params.state_n0_z = state_n0_soa.z.data_ptr<float>();
    params.state_n1_x = state_n1_soa.x.data_ptr<float>();
    params.state_n1_y = state_n1_soa.y.data_ptr<float>();
    params.state_n1_z = state_n1_soa.z.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.seed = 0;
    params.max_order = 1;
    params.strategy_mask = RAYDTORCH_DFR_DIRECT;
    params.sample_count = 1;
    params.return_geom = 1;
    params.receiver_model = RAYDTORCH_DFR_MATCHED_ISO;
    params.temp_visibility = nullptr;
    params.out_count = out_count.data_ptr<int>();
    params.out_valid = mutable_mask_ptr(out_valid);
    params.out_tx_id = out_tx_id.data_ptr<int>();
    params.out_rx_id = out_rx_id.data_ptr<int>();
    params.out_order = out_order.data_ptr<int>();
    params.out_edge0 = out_edge0.data_ptr<int>();
    params.out_edge1 = out_edge1.data_ptr<int>();
    params.out_edge2 = out_edge2.data_ptr<int>();
    params.out_delay = out_delay.data_ptr<float>();
    params.out_field_x_re = out_field_x_re.data_ptr<float>();
    params.out_field_x_im = out_field_x_im.data_ptr<float>();
    params.out_field_y_re = out_field_y_re.data_ptr<float>();
    params.out_field_y_im = out_field_y_im.data_ptr<float>();
    params.out_field_z_re = out_field_z_re.data_ptr<float>();
    params.out_field_z_im = out_field_z_im.data_ptr<float>();
    params.out_p0_x = out_p0_x.data_ptr<float>();
    params.out_p0_y = out_p0_y.data_ptr<float>();
    params.out_p0_z = out_p0_z.data_ptr<float>();
    params.out_p1_x = nullptr;
    params.out_p1_y = nullptr;
    params.out_p1_z = nullptr;
    params.out_p2_x = nullptr;
    params.out_p2_y = nullptr;
    params.out_p2_z = nullptr;

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    auto pipeline = optix_pipeline_for_scene(scene, diffraction_paths_pipeline_config());
    pipeline->launch(0, params, static_cast<unsigned int>(n_rays), torch_ctx.stream);

    return py::make_tuple(
        out_count,
        out_valid,
        out_tx_id,
        out_rx_id,
        out_order,
        out_edge0,
        out_edge1,
        out_edge2,
        out_delay,
        out_field_x_re,
        out_field_x_im,
        out_field_y_re,
        out_field_y_im,
        out_field_z_re,
        out_field_z_im,
        at::stack({out_p0_x, out_p0_y, out_p0_z}, 1).contiguous(),
        out_p1,
        out_p2);
}

py::tuple diffraction_accumulation_forward_op(
    int64_t scene_handle,
    at::Tensor active,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_n0,
    at::Tensor state_n1,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor state_wi,
    at::Tensor state_d0,
    at::Tensor material_eta_r,
    at::Tensor material_sigma,
    at::Tensor material_mu_r,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    int64_t max_order,
    at::Tensor recursive_active,
    at::Tensor recursive_state_edge_index,
    at::Tensor recursive_state_edge_pos,
    at::Tensor recursive_state_edge_dir,
    at::Tensor recursive_state_edge_t_min,
    at::Tensor recursive_state_edge_t_max,
    at::Tensor recursive_state_n0,
    at::Tensor recursive_state_n1,
    at::Tensor recursive_state_prim0,
    at::Tensor recursive_state_prim1,
    at::Tensor recursive_state_exterior_angle,
    int64_t export_tape) {
    require_mask(active, "active");
    require_flat_i32(state_edge_index, "state_edge_index");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_vec3f(state_n0, "state_n0");
    require_vec3f(state_n1, "state_n1");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_vec3f(state_wi, "state_wi");
    require_vec3f(state_d0, "state_d0");
    require_flat_f32(material_eta_r, "material_eta_r");
    require_flat_f32(material_sigma, "material_sigma");
    require_flat_f32(material_mu_r, "material_mu_r");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    if (grid_axis < 0 || grid_axis > 2)
        throw std::runtime_error("grid_axis must be 0, 1, or 2.");
    if (!(grid_coord0_min < grid_coord0_max) || !(grid_coord1_min < grid_coord1_max))
        throw std::runtime_error("grid bounds must be ordered.");
    if (grid_resolution0 <= 0 || grid_resolution1 <= 0)
        throw std::runtime_error("grid resolutions must be positive.");
    if (!(grid_cell_area > 0.0))
        throw std::runtime_error("grid_cell_area must be positive.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");
    if (direct_samples < 0)
        throw std::runtime_error("direct_samples must be non-negative.");
    if (keller_samples < 0)
        throw std::runtime_error("keller_samples must be non-negative.");
    if (suffix_samples < 0)
        throw std::runtime_error("suffix_samples must be non-negative.");
    if (max_order < 1 || max_order > 3)
        throw std::runtime_error("max_order must be 1, 2, or 3.");

    SceneCache &scene = get_scene(scene_handle);
    const int64_t state_count = state_edge_index.size(0);
    require_state_width(state_edge_pos, state_count, "state_edge_pos");
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_n0, state_count, "state_n0");
    require_state_width(state_n1, state_count, "state_n1");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    require_state_width(state_wi, state_count, "state_wi");
    require_state_width(state_d0, state_count, "state_d0");
    const bool use_recursive = max_order > 1;
    int64_t recursive_state_count = 0;
    if (use_recursive) {
        require_mask(recursive_active, "recursive_active");
        require_flat_i32(recursive_state_edge_index, "recursive_state_edge_index");
        require_vec3f(recursive_state_edge_pos, "recursive_state_edge_pos");
        require_vec3f(recursive_state_edge_dir, "recursive_state_edge_dir");
        require_scalar_f(recursive_state_edge_t_min, "recursive_state_edge_t_min");
        require_scalar_f(recursive_state_edge_t_max, "recursive_state_edge_t_max");
        require_vec3f(recursive_state_n0, "recursive_state_n0");
        require_vec3f(recursive_state_n1, "recursive_state_n1");
        require_flat_i32(recursive_state_prim0, "recursive_state_prim0");
        require_flat_i32(recursive_state_prim1, "recursive_state_prim1");
        require_scalar_f(recursive_state_exterior_angle, "recursive_state_exterior_angle");
        recursive_state_count = recursive_state_edge_index.size(0);
        require_state_width(recursive_active, recursive_state_count, "recursive_active");
        require_state_width(recursive_state_edge_pos, recursive_state_count, "recursive_state_edge_pos");
        require_state_width(recursive_state_edge_dir, recursive_state_count, "recursive_state_edge_dir");
        require_state_width(recursive_state_edge_t_min, recursive_state_count, "recursive_state_edge_t_min");
        require_state_width(recursive_state_edge_t_max, recursive_state_count, "recursive_state_edge_t_max");
        require_state_width(recursive_state_n0, recursive_state_count, "recursive_state_n0");
        require_state_width(recursive_state_n1, recursive_state_count, "recursive_state_n1");
        require_state_width(recursive_state_prim0, recursive_state_count, "recursive_state_prim0");
        require_state_width(recursive_state_prim1, recursive_state_count, "recursive_state_prim1");
        require_state_width(recursive_state_exterior_angle, recursive_state_count, "recursive_state_exterior_angle");
    }
    const int64_t material_count = material_eta_r.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_sigma.size(0) != material_count ||
        material_mu_r.size(0) != material_count ||
        material_gain.size(0) != material_count ||
        material_valid.size(0) != material_count) {
        throw std::runtime_error("material payload fields must have matching widths.");
    }

    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    const int32_t direct_launch_count = checked_i32(direct_samples, "direct_samples");
    const int32_t keller_launch_count = checked_i32(keller_samples, "keller_samples");
    const int32_t suffix_launch_count = checked_i32(suffix_samples, "suffix_samples");
    const int32_t launch_count = checked_i32(direct_samples + keller_samples + suffix_samples, "launch_count");
    auto fopts = state_src.options();
    auto iopts = state_edge_index.options();
    at::Tensor power = at::zeros({cell_count}, fopts);
    at::Tensor field_x_re = at::zeros({cell_count}, fopts);
    at::Tensor field_x_im = at::zeros({cell_count}, fopts);
    at::Tensor field_y_re = at::zeros({cell_count}, fopts);
    at::Tensor field_y_im = at::zeros({cell_count}, fopts);
    at::Tensor field_z_re = at::zeros({cell_count}, fopts);
    at::Tensor field_z_im = at::zeros({cell_count}, fopts);
    at::Tensor direct_count = at::zeros({1}, iopts);
    at::Tensor keller_count = at::zeros({1}, iopts);
    at::Tensor suffix_count = at::zeros({1}, iopts);
    at::Tensor vis_rejects = at::zeros({1}, iopts);
    at::Tensor edge_vis_rejects = at::zeros({1}, iopts);
    at::Tensor utd_rejects = at::zeros({1}, iopts);
    at::Tensor edge_uses = at::zeros({1}, iopts);
    const bool write_tape = export_tape != 0;
    at::Tensor tape_active = write_tape ? at::zeros({launch_count}, active.options()) : at::empty({0}, active.options());
    at::Tensor tape_state_idx = write_tape ? at::full({launch_count}, -1, iopts) : at::empty({0}, iopts);
    at::Tensor tape_cell = write_tape ? at::full({launch_count}, -1, iopts) : at::empty({0}, iopts);
    at::Tensor tape_material_idx = write_tape ? at::full({launch_count}, -1, iopts) : at::empty({0}, iopts);
    at::Tensor tape_edge_u = write_tape ? at::zeros({launch_count}, fopts) : at::empty({0}, fopts);
    const bool staged_no_suffix_accum =
        !write_tape &&
        !use_recursive &&
        suffix_launch_count == 0 &&
        (direct_launch_count + keller_launch_count) > 0 &&
        static_cast<int64_t>(launch_count) >= kStagedDfrAccumMinSamples &&
        static_cast<int64_t>(launch_count) >= cell_count * kStagedDfrAccumMinSamplesPerCell;
    at::Tensor stage_cell = staged_no_suffix_accum
        ? at::full({launch_count}, -1, iopts)
        : at::empty({0}, iopts);
    at::Tensor stage_value = staged_no_suffix_accum
        ? at::zeros({launch_count, 4}, fopts)
        : at::empty({0, 4}, fopts);
    if (state_count == 0 || launch_count == 0) {
        return py::make_tuple(
            power.reshape({grid_resolution1, grid_resolution0}),
            field_x_re.reshape({grid_resolution1, grid_resolution0}),
            field_x_im.reshape({grid_resolution1, grid_resolution0}),
            field_y_re.reshape({grid_resolution1, grid_resolution0}),
            field_y_im.reshape({grid_resolution1, grid_resolution0}),
            field_z_re.reshape({grid_resolution1, grid_resolution0}),
            field_z_im.reshape({grid_resolution1, grid_resolution0}),
            direct_count,
            keller_count,
            suffix_count,
            vis_rejects,
            edge_vis_rejects,
            utd_rejects,
            edge_uses,
            tape_active,
            tape_state_idx,
            tape_cell,
            tape_material_idx,
            tape_edge_u);
    }

    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_n0_soa = split_vec3(state_n0);
    Vec3SoA state_n1_soa = split_vec3(state_n1);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA state_wi_soa = split_vec3(state_wi);
    Vec3SoA state_d0_soa = split_vec3(state_d0);
    at::Tensor active_contig = active_mask_for_states(active, state_count, "diffraction_accumulation_forward");
    at::Tensor state_prefix_depth = at::zeros({state_count}, iopts);
    at::Tensor temp_visibility = at::zeros({launch_count}, active.options());
    at::Tensor recursive_active_contig;
    at::Tensor recursive_prefix_depth;
    Vec3SoA recursive_edge_pos_soa;
    Vec3SoA recursive_edge_dir_soa;
    Vec3SoA recursive_n0_soa;
    Vec3SoA recursive_n1_soa;
    if (use_recursive) {
        recursive_active_contig = active_mask_for_states(
            recursive_active,
            recursive_state_count,
            "diffraction_accumulation_forward recursive_active");
        recursive_prefix_depth = at::zeros({recursive_state_count}, iopts);
        recursive_edge_pos_soa = split_vec3(recursive_state_edge_pos);
        recursive_edge_dir_soa = split_vec3(recursive_state_edge_dir);
        recursive_n0_soa = split_vec3(recursive_state_n0);
        recursive_n1_soa = split_vec3(recursive_state_n1);
    }

    DfrAccumParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = launch_count;
    params.active_mask = mask_ptr(active_contig);
    params.state_count = checked_i32(state_count, "state_count");
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_n0_x = state_n0_soa.x.data_ptr<float>();
    params.state_n0_y = state_n0_soa.y.data_ptr<float>();
    params.state_n0_z = state_n0_soa.z.data_ptr<float>();
    params.state_n1_x = state_n1_soa.x.data_ptr<float>();
    params.state_n1_y = state_n1_soa.y.data_ptr<float>();
    params.state_n1_z = state_n1_soa.z.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_wi_x = state_wi_soa.x.data_ptr<float>();
    params.state_wi_y = state_wi_soa.y.data_ptr<float>();
    params.state_wi_z = state_wi_soa.z.data_ptr<float>();
    params.state_d0_x = state_d0_soa.x.data_ptr<float>();
    params.state_d0_y = state_d0_soa.y.data_ptr<float>();
    params.state_d0_z = state_d0_soa.z.data_ptr<float>();
    params.state_prefix_depth = state_prefix_depth.data_ptr<int>();
    params.recursive_state_count = checked_i32(recursive_state_count, "recursive_state_count");
    if (use_recursive) {
        params.recursive_active_mask = mask_ptr(recursive_active_contig);
        params.recursive_state_edge_index = recursive_state_edge_index.data_ptr<int>();
        params.recursive_state_edge_pos_x = recursive_edge_pos_soa.x.data_ptr<float>();
        params.recursive_state_edge_pos_y = recursive_edge_pos_soa.y.data_ptr<float>();
        params.recursive_state_edge_pos_z = recursive_edge_pos_soa.z.data_ptr<float>();
        params.recursive_state_edge_dir_x = recursive_edge_dir_soa.x.data_ptr<float>();
        params.recursive_state_edge_dir_y = recursive_edge_dir_soa.y.data_ptr<float>();
        params.recursive_state_edge_dir_z = recursive_edge_dir_soa.z.data_ptr<float>();
        params.recursive_state_edge_t_min = recursive_state_edge_t_min.data_ptr<float>();
        params.recursive_state_edge_t_max = recursive_state_edge_t_max.data_ptr<float>();
        params.recursive_state_n0_x = recursive_n0_soa.x.data_ptr<float>();
        params.recursive_state_n0_y = recursive_n0_soa.y.data_ptr<float>();
        params.recursive_state_n0_z = recursive_n0_soa.z.data_ptr<float>();
        params.recursive_state_n1_x = recursive_n1_soa.x.data_ptr<float>();
        params.recursive_state_n1_y = recursive_n1_soa.y.data_ptr<float>();
        params.recursive_state_n1_z = recursive_n1_soa.z.data_ptr<float>();
        params.recursive_state_prim0 = recursive_state_prim0.data_ptr<int>();
        params.recursive_state_prim1 = recursive_state_prim1.data_ptr<int>();
        params.recursive_state_exterior_angle = recursive_state_exterior_angle.data_ptr<float>();
    }
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.n_triangles = tri.n_triangles;
    params.suffix_candidate_prim_id = nullptr;
    params.suffix_candidate_count = 0;
    params.material_eta_r = material_eta_r.data_ptr<float>();
    params.material_sigma = material_sigma.data_ptr<float>();
    params.material_mu_r = material_mu_r.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.seed = checked_i32(seed, "seed");
    params.samples = launch_count;
    params.max_order = checked_i32(max_order, "max_order");
    params.direct_samples = direct_launch_count;
    params.keller_samples = keller_launch_count;
    params.suffix_samples = suffix_launch_count;
    params.strategy_mask =
        (direct_launch_count > 0 ? RAYDTORCH_DFR_DIRECT : 0) |
        (keller_launch_count > 0 ? RAYDTORCH_DFR_KELLER : 0) |
        (suffix_launch_count > 0 ? RAYDTORCH_DFR_SUFFIX_REFL : 0);
    params.sample_sequence = RAYDTORCH_DFR_HASH;
    params.receiver_model = RAYDTORCH_DFR_MATCHED_ISO;
    params.select_diffraction_point = 0;
    params.prefilter_visibility = 0;
    params.collect_edge_use = 1;
    params.collect_debug_counts = 1;
    params.omega = 2.0f * 3.14159265358979323846f * 299792458.0f;
    params.tx_pol_x = 1.0f;
    params.tx_pol_y = 0.0f;
    params.tx_pol_z = 0.0f;
    params.out_power = power.data_ptr<float>();
    params.out_field_x_re = field_x_re.data_ptr<float>();
    params.out_field_x_im = field_x_im.data_ptr<float>();
    params.out_field_y_re = field_y_re.data_ptr<float>();
    params.out_field_y_im = field_y_im.data_ptr<float>();
    params.out_field_z_re = field_z_re.data_ptr<float>();
    params.out_field_z_im = field_z_im.data_ptr<float>();
    params.out_direct_count = direct_count.data_ptr<int>();
    params.out_keller_count = keller_count.data_ptr<int>();
    params.out_suffix_count = suffix_count.data_ptr<int>();
    params.out_vis_rejects = vis_rejects.data_ptr<int>();
    params.out_edge_vis_rejects = edge_vis_rejects.data_ptr<int>();
    params.out_utd_rejects = utd_rejects.data_ptr<int>();
    params.out_edge_uses = edge_uses.data_ptr<int>();
    params.temp_visibility = mutable_mask_ptr(temp_visibility);
    params.tape_active = write_tape ? mutable_mask_ptr(tape_active) : nullptr;
    params.tape_state_idx = write_tape ? tape_state_idx.data_ptr<int>() : nullptr;
    params.tape_cell = write_tape ? tape_cell.data_ptr<int>() : nullptr;
    params.tape_material_idx = write_tape ? tape_material_idx.data_ptr<int>() : nullptr;
    params.tape_edge_u = write_tape ? tape_edge_u.data_ptr<float>() : nullptr;
    params.stage_cell = staged_no_suffix_accum ? stage_cell.data_ptr<int>() : nullptr;
    params.stage_value = staged_no_suffix_accum
        ? reinterpret_cast<float4 *>(stage_value.data_ptr<float>())
        : nullptr;

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    auto pipeline = optix_pipeline_for_scene(scene, diffraction_accumulation_pipeline_config());
    if (use_recursive) {
        pipeline->launch(13, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
    } else {
        pipeline->launch(6, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
        if (direct_launch_count + keller_launch_count > 0)
            pipeline->launch(7, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
        if (staged_no_suffix_accum) {
            reduce_dfr_accum_staged_cuda(
                launch_count,
                stage_cell,
                stage_value,
                power,
                field_x_re,
                direct_count,
                keller_count,
                edge_uses);
        }
        if (suffix_launch_count > 0) {
            pipeline->launch(8, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
            pipeline->launch(9, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
        }
    }

    return py::make_tuple(
        power.reshape({grid_resolution1, grid_resolution0}),
        field_x_re.reshape({grid_resolution1, grid_resolution0}),
        field_x_im.reshape({grid_resolution1, grid_resolution0}),
        field_y_re.reshape({grid_resolution1, grid_resolution0}),
        field_y_im.reshape({grid_resolution1, grid_resolution0}),
        field_z_re.reshape({grid_resolution1, grid_resolution0}),
        field_z_im.reshape({grid_resolution1, grid_resolution0}),
        direct_count,
        keller_count,
        suffix_count,
        vis_rejects,
        edge_vis_rejects,
        utd_rejects,
        edge_uses,
        tape_active,
        tape_state_idx,
        tape_cell,
        tape_material_idx,
        tape_edge_u);
}

py::tuple diffraction_accumulation_direct_backward_op(
    int64_t scene_handle,
    at::Tensor tape_active,
    at::Tensor tape_state_idx,
    at::Tensor tape_cell,
    at::Tensor tape_material_idx,
    at::Tensor tape_edge_u,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor state_wi,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    at::Tensor grad_power,
    at::Tensor grad_field_x_re) {
    require_mask(tape_active, "tape_active");
    require_flat_i32(tape_state_idx, "tape_state_idx");
    require_flat_i32(tape_cell, "tape_cell");
    require_flat_i32(tape_material_idx, "tape_material_idx");
    require_flat_f32(tape_edge_u, "tape_edge_u");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_vec3f(state_wi, "state_wi");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    const int64_t launch_count = tape_active.size(0);
    require_state_width(tape_state_idx, launch_count, "tape_state_idx");
    require_state_width(tape_cell, launch_count, "tape_cell");
    require_state_width(tape_material_idx, launch_count, "tape_material_idx");
    require_state_width(tape_edge_u, launch_count, "tape_edge_u");
    const int64_t state_count = state_edge_pos.size(0);
    const int64_t material_count = material_gain.size(0);
    if (material_valid.size(0) != material_count)
        throw std::runtime_error("material_valid must match material_gain width.");

    SceneCache &scene = get_scene(scene_handle);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA state_wi_soa = split_vec3(state_wi);
    at::Tensor grad_power_flat = grad_power.reshape({-1}).contiguous();
    at::Tensor grad_field_x_re_flat = grad_field_x_re.reshape({-1}).contiguous();
    require_flat_f32(grad_power_flat, "grad_power");
    require_flat_f32(grad_field_x_re_flat, "grad_field_x_re");

    at::Tensor grad_edge_pos_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_pos_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_pos_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_t_min = at::zeros_like(state_edge_t_min);
    at::Tensor grad_edge_t_max = at::zeros_like(state_edge_t_max);
    at::Tensor grad_src_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_wi_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_wi_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_wi_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_power = at::zeros_like(state_src_power);
    at::Tensor grad_exterior_angle = at::zeros_like(state_exterior_angle);
    at::Tensor grad_material_gain = at::zeros_like(material_gain);
    at::Tensor grad_tri_p0_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_z = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_z = at::zeros({tri.n_triangles}, state_edge_pos.options());

    DfrDirectAccumADParams params = {};
    params.n_rays = checked_i32(launch_count, "n_rays");
    params.state_count = checked_i32(state_count, "state_count");
    params.material_count = checked_i32(material_count, "material_count");
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.direct_samples = checked_i32(direct_samples, "direct_samples");
    params.keller_samples = checked_i32(keller_samples, "keller_samples");
    params.suffix_samples = checked_i32(suffix_samples, "suffix_samples");
    params.wavelength = static_cast<float>(wavelength);
    params.seed = checked_i32(seed, "seed");
    params.n_triangles = tri.n_triangles;
    params.tape_active = mask_ptr(tape_active);
    params.tape_state_idx = tape_state_idx.data_ptr<int>();
    params.tape_cell = tape_cell.data_ptr<int>();
    params.tape_material_idx = tape_material_idx.data_ptr<int>();
    params.tape_edge_u = tape_edge_u.data_ptr<float>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_wi_x = state_wi_soa.x.data_ptr<float>();
    params.state_wi_y = state_wi_soa.y.data_ptr<float>();
    params.state_wi_z = state_wi_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.grad_out_power = grad_power_flat.data_ptr<float>();
    params.grad_out_field_x_re = grad_field_x_re_flat.data_ptr<float>();
    params.grad_state_edge_pos_x = grad_edge_pos_x.data_ptr<float>();
    params.grad_state_edge_pos_y = grad_edge_pos_y.data_ptr<float>();
    params.grad_state_edge_pos_z = grad_edge_pos_z.data_ptr<float>();
    params.grad_state_edge_dir_x = grad_edge_dir_x.data_ptr<float>();
    params.grad_state_edge_dir_y = grad_edge_dir_y.data_ptr<float>();
    params.grad_state_edge_dir_z = grad_edge_dir_z.data_ptr<float>();
    params.grad_state_edge_t_min = grad_edge_t_min.data_ptr<float>();
    params.grad_state_edge_t_max = grad_edge_t_max.data_ptr<float>();
    params.grad_state_src_x = grad_src_x.data_ptr<float>();
    params.grad_state_src_y = grad_src_y.data_ptr<float>();
    params.grad_state_src_z = grad_src_z.data_ptr<float>();
    params.grad_state_wi_x = grad_wi_x.data_ptr<float>();
    params.grad_state_wi_y = grad_wi_y.data_ptr<float>();
    params.grad_state_wi_z = grad_wi_z.data_ptr<float>();
    params.grad_state_src_power = grad_src_power.data_ptr<float>();
    params.grad_state_exterior_angle = grad_exterior_angle.data_ptr<float>();
    params.grad_material_gain = grad_material_gain.data_ptr<float>();
    params.grad_tri_p0_x = grad_tri_p0_x.data_ptr<float>();
    params.grad_tri_p0_y = grad_tri_p0_y.data_ptr<float>();
    params.grad_tri_p0_z = grad_tri_p0_z.data_ptr<float>();
    params.grad_tri_fn_x = grad_tri_fn_x.data_ptr<float>();
    params.grad_tri_fn_y = grad_tri_fn_y.data_ptr<float>();
    params.grad_tri_fn_z = grad_tri_fn_z.data_ptr<float>();
    dfr_direct_accum_vjp_gpu(params);
    return py::make_tuple(
        stack_vec3(grad_edge_pos_x, grad_edge_pos_y, grad_edge_pos_z),
        stack_vec3(grad_edge_dir_x, grad_edge_dir_y, grad_edge_dir_z),
        grad_edge_t_min,
        grad_edge_t_max,
        stack_vec3(grad_src_x, grad_src_y, grad_src_z),
        stack_vec3(grad_wi_x, grad_wi_y, grad_wi_z),
        grad_src_power,
        grad_exterior_angle,
        grad_material_gain);
}

py::tuple diffraction_accumulation_direct_jvp_op(
    int64_t scene_handle,
    at::Tensor tape_active,
    at::Tensor tape_state_idx,
    at::Tensor tape_cell,
    at::Tensor tape_material_idx,
    at::Tensor tape_edge_u,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor state_wi,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    at::Tensor dot_state_edge_pos,
    at::Tensor dot_state_edge_dir,
    at::Tensor dot_state_edge_t_min,
    at::Tensor dot_state_edge_t_max,
    at::Tensor dot_state_exterior_angle,
    at::Tensor dot_state_src,
    at::Tensor dot_state_src_power,
    at::Tensor dot_state_wi,
    at::Tensor dot_material_gain) {
    SceneCache &scene = get_scene(scene_handle);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA state_wi_soa = split_vec3(state_wi);
    Vec3SoA dot_edge_pos_soa = split_vec3(dot_state_edge_pos);
    Vec3SoA dot_edge_dir_soa = split_vec3(dot_state_edge_dir);
    Vec3SoA dot_src_soa = split_vec3(dot_state_src);
    Vec3SoA dot_wi_soa = split_vec3(dot_state_wi);
    const int64_t launch_count = tape_active.size(0);
    const int64_t state_count = state_edge_pos.size(0);
    const int64_t material_count = material_gain.size(0);
    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    at::Tensor dot_power = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor dot_field_x_re = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor zero_tri = at::zeros({tri.n_triangles}, state_edge_pos.options());

    DfrDirectAccumADParams params = {};
    params.n_rays = checked_i32(launch_count, "n_rays");
    params.state_count = checked_i32(state_count, "state_count");
    params.material_count = checked_i32(material_count, "material_count");
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.direct_samples = checked_i32(direct_samples, "direct_samples");
    params.keller_samples = checked_i32(keller_samples, "keller_samples");
    params.suffix_samples = checked_i32(suffix_samples, "suffix_samples");
    params.wavelength = static_cast<float>(wavelength);
    params.seed = checked_i32(seed, "seed");
    params.n_triangles = tri.n_triangles;
    params.tape_active = mask_ptr(tape_active);
    params.tape_state_idx = tape_state_idx.data_ptr<int>();
    params.tape_cell = tape_cell.data_ptr<int>();
    params.tape_material_idx = tape_material_idx.data_ptr<int>();
    params.tape_edge_u = tape_edge_u.data_ptr<float>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_wi_x = state_wi_soa.x.data_ptr<float>();
    params.state_wi_y = state_wi_soa.y.data_ptr<float>();
    params.state_wi_z = state_wi_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.dot_state_edge_pos_x = dot_edge_pos_soa.x.data_ptr<float>();
    params.dot_state_edge_pos_y = dot_edge_pos_soa.y.data_ptr<float>();
    params.dot_state_edge_pos_z = dot_edge_pos_soa.z.data_ptr<float>();
    params.dot_state_edge_dir_x = dot_edge_dir_soa.x.data_ptr<float>();
    params.dot_state_edge_dir_y = dot_edge_dir_soa.y.data_ptr<float>();
    params.dot_state_edge_dir_z = dot_edge_dir_soa.z.data_ptr<float>();
    params.dot_state_edge_t_min = dot_state_edge_t_min.data_ptr<float>();
    params.dot_state_edge_t_max = dot_state_edge_t_max.data_ptr<float>();
    params.dot_state_src_x = dot_src_soa.x.data_ptr<float>();
    params.dot_state_src_y = dot_src_soa.y.data_ptr<float>();
    params.dot_state_src_z = dot_src_soa.z.data_ptr<float>();
    params.dot_state_wi_x = dot_wi_soa.x.data_ptr<float>();
    params.dot_state_wi_y = dot_wi_soa.y.data_ptr<float>();
    params.dot_state_wi_z = dot_wi_soa.z.data_ptr<float>();
    params.dot_state_src_power = dot_state_src_power.data_ptr<float>();
    params.dot_state_exterior_angle = dot_state_exterior_angle.data_ptr<float>();
    params.dot_material_gain = dot_material_gain.data_ptr<float>();
    params.dot_tri_p0_x = zero_tri.data_ptr<float>();
    params.dot_tri_p0_y = zero_tri.data_ptr<float>();
    params.dot_tri_p0_z = zero_tri.data_ptr<float>();
    params.dot_tri_fn_x = zero_tri.data_ptr<float>();
    params.dot_tri_fn_y = zero_tri.data_ptr<float>();
    params.dot_tri_fn_z = zero_tri.data_ptr<float>();
    params.dot_out_power = dot_power.data_ptr<float>();
    params.dot_out_field_x_re = dot_field_x_re.data_ptr<float>();
    dfr_direct_accum_jvp_gpu(params);
    return py::make_tuple(
        dot_power.reshape({grid_resolution1, grid_resolution0}),
        dot_field_x_re.reshape({grid_resolution1, grid_resolution0}));
}

py::tuple diffraction_accumulation_chain_backward_op(
    int64_t scene_handle,
    at::Tensor tape_active,
    at::Tensor tape_cell,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor recursive_state_edge_index,
    at::Tensor recursive_state_edge_pos,
    at::Tensor recursive_state_edge_dir,
    at::Tensor recursive_state_edge_t_min,
    at::Tensor recursive_state_edge_t_max,
    at::Tensor recursive_state_prim0,
    at::Tensor recursive_state_prim1,
    at::Tensor recursive_state_exterior_angle,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    int64_t max_order,
    at::Tensor grad_power,
    at::Tensor grad_field_x_re) {
    require_mask(tape_active, "tape_active");
    require_flat_i32(tape_cell, "tape_cell");
    require_flat_i32(state_edge_index, "state_edge_index");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_flat_i32(recursive_state_edge_index, "recursive_state_edge_index");
    require_vec3f(recursive_state_edge_pos, "recursive_state_edge_pos");
    require_vec3f(recursive_state_edge_dir, "recursive_state_edge_dir");
    require_scalar_f(recursive_state_edge_t_min, "recursive_state_edge_t_min");
    require_scalar_f(recursive_state_edge_t_max, "recursive_state_edge_t_max");
    require_flat_i32(recursive_state_prim0, "recursive_state_prim0");
    require_flat_i32(recursive_state_prim1, "recursive_state_prim1");
    require_scalar_f(recursive_state_exterior_angle, "recursive_state_exterior_angle");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    const int64_t launch_count = tape_active.size(0);
    require_state_width(tape_cell, launch_count, "tape_cell");
    const int64_t state_count = state_edge_pos.size(0);
    const int64_t recursive_state_count = recursive_state_edge_pos.size(0);
    const int64_t material_count = material_gain.size(0);
    if (material_valid.size(0) != material_count)
        throw std::runtime_error("material_valid must match material_gain width.");

    SceneCache &scene = get_scene(scene_handle);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA recursive_edge_pos_soa = split_vec3(recursive_state_edge_pos);
    Vec3SoA recursive_edge_dir_soa = split_vec3(recursive_state_edge_dir);
    at::Tensor grad_power_flat = grad_power.reshape({-1}).contiguous();
    at::Tensor grad_field_x_re_flat = grad_field_x_re.reshape({-1}).contiguous();
    require_flat_f32(grad_power_flat, "grad_power");
    require_flat_f32(grad_field_x_re_flat, "grad_field_x_re");

    at::Tensor grad_edge_pos_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_pos_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_pos_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_dir_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_edge_t_min = at::zeros_like(state_edge_t_min);
    at::Tensor grad_edge_t_max = at::zeros_like(state_edge_t_max);
    at::Tensor grad_src_x = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_y = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_z = at::zeros({state_count}, state_edge_pos.options());
    at::Tensor grad_src_power = at::zeros_like(state_src_power);
    at::Tensor grad_exterior_angle = at::zeros_like(state_exterior_angle);
    at::Tensor grad_recursive_edge_pos_x = at::zeros({recursive_state_count}, recursive_state_edge_pos.options());
    at::Tensor grad_recursive_edge_pos_y = at::zeros({recursive_state_count}, recursive_state_edge_pos.options());
    at::Tensor grad_recursive_edge_pos_z = at::zeros({recursive_state_count}, recursive_state_edge_pos.options());
    at::Tensor grad_recursive_edge_dir_x = at::zeros({recursive_state_count}, recursive_state_edge_pos.options());
    at::Tensor grad_recursive_edge_dir_y = at::zeros({recursive_state_count}, recursive_state_edge_pos.options());
    at::Tensor grad_recursive_edge_dir_z = at::zeros({recursive_state_count}, recursive_state_edge_pos.options());
    at::Tensor grad_recursive_edge_t_min = at::zeros_like(recursive_state_edge_t_min);
    at::Tensor grad_recursive_edge_t_max = at::zeros_like(recursive_state_edge_t_max);
    at::Tensor grad_recursive_exterior_angle = at::zeros_like(recursive_state_exterior_angle);
    at::Tensor grad_material_gain = at::zeros_like(material_gain);
    at::Tensor grad_tri_p0_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_p0_z = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_x = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_y = at::zeros({tri.n_triangles}, state_edge_pos.options());
    at::Tensor grad_tri_fn_z = at::zeros({tri.n_triangles}, state_edge_pos.options());

    DfrChainAccumADParams params = {};
    params.n_rays = checked_i32(launch_count, "n_rays");
    params.state_count = checked_i32(state_count, "state_count");
    params.recursive_state_count = checked_i32(recursive_state_count, "recursive_state_count");
    params.material_count = checked_i32(material_count, "material_count");
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.direct_samples = checked_i32(direct_samples, "direct_samples");
    params.keller_samples = checked_i32(keller_samples, "keller_samples");
    params.suffix_samples = checked_i32(suffix_samples, "suffix_samples");
    params.max_order = checked_i32(max_order, "max_order");
    params.wavelength = static_cast<float>(wavelength);
    params.seed = checked_i32(seed, "seed");
    params.n_triangles = tri.n_triangles;
    params.tape_active = mask_ptr(tape_active);
    params.tape_cell = tape_cell.data_ptr<int>();
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.recursive_state_edge_index = recursive_state_edge_index.data_ptr<int>();
    params.recursive_state_edge_pos_x = recursive_edge_pos_soa.x.data_ptr<float>();
    params.recursive_state_edge_pos_y = recursive_edge_pos_soa.y.data_ptr<float>();
    params.recursive_state_edge_pos_z = recursive_edge_pos_soa.z.data_ptr<float>();
    params.recursive_state_edge_dir_x = recursive_edge_dir_soa.x.data_ptr<float>();
    params.recursive_state_edge_dir_y = recursive_edge_dir_soa.y.data_ptr<float>();
    params.recursive_state_edge_dir_z = recursive_edge_dir_soa.z.data_ptr<float>();
    params.recursive_state_edge_t_min = recursive_state_edge_t_min.data_ptr<float>();
    params.recursive_state_edge_t_max = recursive_state_edge_t_max.data_ptr<float>();
    params.recursive_state_exterior_angle = recursive_state_exterior_angle.data_ptr<float>();
    params.recursive_state_prim0 = recursive_state_prim0.data_ptr<int>();
    params.recursive_state_prim1 = recursive_state_prim1.data_ptr<int>();
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.grad_out_power = grad_power_flat.data_ptr<float>();
    params.grad_out_field_x_re = grad_field_x_re_flat.data_ptr<float>();
    params.grad_state_edge_pos_x = grad_edge_pos_x.data_ptr<float>();
    params.grad_state_edge_pos_y = grad_edge_pos_y.data_ptr<float>();
    params.grad_state_edge_pos_z = grad_edge_pos_z.data_ptr<float>();
    params.grad_state_edge_dir_x = grad_edge_dir_x.data_ptr<float>();
    params.grad_state_edge_dir_y = grad_edge_dir_y.data_ptr<float>();
    params.grad_state_edge_dir_z = grad_edge_dir_z.data_ptr<float>();
    params.grad_state_edge_t_min = grad_edge_t_min.data_ptr<float>();
    params.grad_state_edge_t_max = grad_edge_t_max.data_ptr<float>();
    params.grad_state_src_x = grad_src_x.data_ptr<float>();
    params.grad_state_src_y = grad_src_y.data_ptr<float>();
    params.grad_state_src_z = grad_src_z.data_ptr<float>();
    params.grad_state_src_power = grad_src_power.data_ptr<float>();
    params.grad_state_exterior_angle = grad_exterior_angle.data_ptr<float>();
    params.grad_recursive_state_edge_pos_x = grad_recursive_edge_pos_x.data_ptr<float>();
    params.grad_recursive_state_edge_pos_y = grad_recursive_edge_pos_y.data_ptr<float>();
    params.grad_recursive_state_edge_pos_z = grad_recursive_edge_pos_z.data_ptr<float>();
    params.grad_recursive_state_edge_dir_x = grad_recursive_edge_dir_x.data_ptr<float>();
    params.grad_recursive_state_edge_dir_y = grad_recursive_edge_dir_y.data_ptr<float>();
    params.grad_recursive_state_edge_dir_z = grad_recursive_edge_dir_z.data_ptr<float>();
    params.grad_recursive_state_edge_t_min = grad_recursive_edge_t_min.data_ptr<float>();
    params.grad_recursive_state_edge_t_max = grad_recursive_edge_t_max.data_ptr<float>();
    params.grad_recursive_state_exterior_angle = grad_recursive_exterior_angle.data_ptr<float>();
    params.grad_material_gain = grad_material_gain.data_ptr<float>();
    params.grad_tri_p0_x = grad_tri_p0_x.data_ptr<float>();
    params.grad_tri_p0_y = grad_tri_p0_y.data_ptr<float>();
    params.grad_tri_p0_z = grad_tri_p0_z.data_ptr<float>();
    params.grad_tri_fn_x = grad_tri_fn_x.data_ptr<float>();
    params.grad_tri_fn_y = grad_tri_fn_y.data_ptr<float>();
    params.grad_tri_fn_z = grad_tri_fn_z.data_ptr<float>();
    dfr_chain_accum_vjp_gpu(params);
    return py::make_tuple(
        stack_vec3(grad_edge_pos_x, grad_edge_pos_y, grad_edge_pos_z),
        stack_vec3(grad_edge_dir_x, grad_edge_dir_y, grad_edge_dir_z),
        grad_edge_t_min,
        grad_edge_t_max,
        stack_vec3(grad_src_x, grad_src_y, grad_src_z),
        grad_src_power,
        grad_exterior_angle,
        stack_vec3(grad_recursive_edge_pos_x, grad_recursive_edge_pos_y, grad_recursive_edge_pos_z),
        stack_vec3(grad_recursive_edge_dir_x, grad_recursive_edge_dir_y, grad_recursive_edge_dir_z),
        grad_recursive_edge_t_min,
        grad_recursive_edge_t_max,
        grad_recursive_exterior_angle,
        grad_material_gain);
}

py::tuple diffraction_accumulation_chain_jvp_op(
    int64_t scene_handle,
    at::Tensor tape_active,
    at::Tensor tape_cell,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor recursive_state_edge_index,
    at::Tensor recursive_state_edge_pos,
    at::Tensor recursive_state_edge_dir,
    at::Tensor recursive_state_edge_t_min,
    at::Tensor recursive_state_edge_t_max,
    at::Tensor recursive_state_prim0,
    at::Tensor recursive_state_prim1,
    at::Tensor recursive_state_exterior_angle,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    int64_t direct_samples,
    int64_t keller_samples,
    int64_t suffix_samples,
    int64_t seed,
    int64_t max_order,
    at::Tensor dot_state_edge_pos,
    at::Tensor dot_state_edge_dir,
    at::Tensor dot_state_edge_t_min,
    at::Tensor dot_state_edge_t_max,
    at::Tensor dot_state_exterior_angle,
    at::Tensor dot_state_src,
    at::Tensor dot_state_src_power,
    at::Tensor dot_recursive_state_edge_pos,
    at::Tensor dot_recursive_state_edge_dir,
    at::Tensor dot_recursive_state_edge_t_min,
    at::Tensor dot_recursive_state_edge_t_max,
    at::Tensor dot_recursive_state_exterior_angle,
    at::Tensor dot_material_gain) {
    SceneCache &scene = get_scene(scene_handle);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA recursive_edge_pos_soa = split_vec3(recursive_state_edge_pos);
    Vec3SoA recursive_edge_dir_soa = split_vec3(recursive_state_edge_dir);
    Vec3SoA dot_edge_pos_soa = split_vec3(dot_state_edge_pos);
    Vec3SoA dot_edge_dir_soa = split_vec3(dot_state_edge_dir);
    Vec3SoA dot_src_soa = split_vec3(dot_state_src);
    Vec3SoA dot_recursive_edge_pos_soa = split_vec3(dot_recursive_state_edge_pos);
    Vec3SoA dot_recursive_edge_dir_soa = split_vec3(dot_recursive_state_edge_dir);
    const int64_t launch_count = tape_active.size(0);
    const int64_t state_count = state_edge_pos.size(0);
    const int64_t recursive_state_count = recursive_state_edge_pos.size(0);
    const int64_t material_count = material_gain.size(0);
    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    at::Tensor dot_power = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor dot_field_x_re = at::zeros({cell_count}, state_edge_pos.options());
    at::Tensor zero_tri = at::zeros({tri.n_triangles}, state_edge_pos.options());

    DfrChainAccumADParams params = {};
    params.n_rays = checked_i32(launch_count, "n_rays");
    params.state_count = checked_i32(state_count, "state_count");
    params.recursive_state_count = checked_i32(recursive_state_count, "recursive_state_count");
    params.material_count = checked_i32(material_count, "material_count");
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.direct_samples = checked_i32(direct_samples, "direct_samples");
    params.keller_samples = checked_i32(keller_samples, "keller_samples");
    params.suffix_samples = checked_i32(suffix_samples, "suffix_samples");
    params.max_order = checked_i32(max_order, "max_order");
    params.wavelength = static_cast<float>(wavelength);
    params.seed = checked_i32(seed, "seed");
    params.n_triangles = tri.n_triangles;
    params.tape_active = mask_ptr(tape_active);
    params.tape_cell = tape_cell.data_ptr<int>();
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.recursive_state_edge_index = recursive_state_edge_index.data_ptr<int>();
    params.recursive_state_edge_pos_x = recursive_edge_pos_soa.x.data_ptr<float>();
    params.recursive_state_edge_pos_y = recursive_edge_pos_soa.y.data_ptr<float>();
    params.recursive_state_edge_pos_z = recursive_edge_pos_soa.z.data_ptr<float>();
    params.recursive_state_edge_dir_x = recursive_edge_dir_soa.x.data_ptr<float>();
    params.recursive_state_edge_dir_y = recursive_edge_dir_soa.y.data_ptr<float>();
    params.recursive_state_edge_dir_z = recursive_edge_dir_soa.z.data_ptr<float>();
    params.recursive_state_edge_t_min = recursive_state_edge_t_min.data_ptr<float>();
    params.recursive_state_edge_t_max = recursive_state_edge_t_max.data_ptr<float>();
    params.recursive_state_exterior_angle = recursive_state_exterior_angle.data_ptr<float>();
    params.recursive_state_prim0 = recursive_state_prim0.data_ptr<int>();
    params.recursive_state_prim1 = recursive_state_prim1.data_ptr<int>();
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.dot_state_edge_pos_x = dot_edge_pos_soa.x.data_ptr<float>();
    params.dot_state_edge_pos_y = dot_edge_pos_soa.y.data_ptr<float>();
    params.dot_state_edge_pos_z = dot_edge_pos_soa.z.data_ptr<float>();
    params.dot_state_edge_dir_x = dot_edge_dir_soa.x.data_ptr<float>();
    params.dot_state_edge_dir_y = dot_edge_dir_soa.y.data_ptr<float>();
    params.dot_state_edge_dir_z = dot_edge_dir_soa.z.data_ptr<float>();
    params.dot_state_edge_t_min = dot_state_edge_t_min.data_ptr<float>();
    params.dot_state_edge_t_max = dot_state_edge_t_max.data_ptr<float>();
    params.dot_state_src_x = dot_src_soa.x.data_ptr<float>();
    params.dot_state_src_y = dot_src_soa.y.data_ptr<float>();
    params.dot_state_src_z = dot_src_soa.z.data_ptr<float>();
    params.dot_state_src_power = dot_state_src_power.data_ptr<float>();
    params.dot_state_exterior_angle = dot_state_exterior_angle.data_ptr<float>();
    params.dot_recursive_state_edge_pos_x = dot_recursive_edge_pos_soa.x.data_ptr<float>();
    params.dot_recursive_state_edge_pos_y = dot_recursive_edge_pos_soa.y.data_ptr<float>();
    params.dot_recursive_state_edge_pos_z = dot_recursive_edge_pos_soa.z.data_ptr<float>();
    params.dot_recursive_state_edge_dir_x = dot_recursive_edge_dir_soa.x.data_ptr<float>();
    params.dot_recursive_state_edge_dir_y = dot_recursive_edge_dir_soa.y.data_ptr<float>();
    params.dot_recursive_state_edge_dir_z = dot_recursive_edge_dir_soa.z.data_ptr<float>();
    params.dot_recursive_state_edge_t_min = dot_recursive_state_edge_t_min.data_ptr<float>();
    params.dot_recursive_state_edge_t_max = dot_recursive_state_edge_t_max.data_ptr<float>();
    params.dot_recursive_state_exterior_angle = dot_recursive_state_exterior_angle.data_ptr<float>();
    params.dot_material_gain = dot_material_gain.data_ptr<float>();
    params.dot_tri_p0_x = zero_tri.data_ptr<float>();
    params.dot_tri_p0_y = zero_tri.data_ptr<float>();
    params.dot_tri_p0_z = zero_tri.data_ptr<float>();
    params.dot_tri_fn_x = zero_tri.data_ptr<float>();
    params.dot_tri_fn_y = zero_tri.data_ptr<float>();
    params.dot_tri_fn_z = zero_tri.data_ptr<float>();
    params.dot_out_power = dot_power.data_ptr<float>();
    params.dot_out_field_x_re = dot_field_x_re.data_ptr<float>();
    dfr_chain_accum_jvp_gpu(params);
    return py::make_tuple(
        dot_power.reshape({grid_resolution1, grid_resolution0}),
        dot_field_x_re.reshape({grid_resolution1, grid_resolution0}));
}

py::tuple diffraction_coherent_accumulation_forward_op(
    int64_t scene_handle,
    at::Tensor active,
    at::Tensor state_edge_index,
    at::Tensor state_edge_pos,
    at::Tensor state_edge_dir,
    at::Tensor state_edge_t_min,
    at::Tensor state_edge_t_max,
    at::Tensor state_n0,
    at::Tensor state_n1,
    at::Tensor state_prim0,
    at::Tensor state_prim1,
    at::Tensor state_exterior_angle,
    at::Tensor state_src,
    at::Tensor state_src_power,
    at::Tensor state_wi,
    at::Tensor state_d0,
    at::Tensor material_eta_r,
    at::Tensor material_sigma,
    at::Tensor material_mu_r,
    at::Tensor material_gain,
    at::Tensor material_valid,
    int64_t grid_axis,
    double grid_position,
    double grid_coord0_min,
    double grid_coord0_max,
    double grid_coord1_min,
    double grid_coord1_max,
    int64_t grid_resolution0,
    int64_t grid_resolution1,
    double grid_cell_area,
    double wavelength,
    bool select_diffraction_point,
    bool prefilter_visibility) {
    require_mask(active, "active");
    require_flat_i32(state_edge_index, "state_edge_index");
    require_vec3f(state_edge_pos, "state_edge_pos");
    require_vec3f(state_edge_dir, "state_edge_dir");
    require_scalar_f(state_edge_t_min, "state_edge_t_min");
    require_scalar_f(state_edge_t_max, "state_edge_t_max");
    require_vec3f(state_n0, "state_n0");
    require_vec3f(state_n1, "state_n1");
    require_flat_i32(state_prim0, "state_prim0");
    require_flat_i32(state_prim1, "state_prim1");
    require_scalar_f(state_exterior_angle, "state_exterior_angle");
    require_vec3f(state_src, "state_src");
    require_scalar_f(state_src_power, "state_src_power");
    require_vec3f(state_wi, "state_wi");
    require_vec3f(state_d0, "state_d0");
    require_flat_f32(material_eta_r, "material_eta_r");
    require_flat_f32(material_sigma, "material_sigma");
    require_flat_f32(material_mu_r, "material_mu_r");
    require_flat_f32(material_gain, "material_gain");
    require_mask(material_valid, "material_valid");
    if (grid_axis < 0 || grid_axis > 2)
        throw std::runtime_error("grid_axis must be 0, 1, or 2.");
    if (!(grid_coord0_min < grid_coord0_max) || !(grid_coord1_min < grid_coord1_max))
        throw std::runtime_error("grid bounds must be ordered.");
    if (grid_resolution0 <= 0 || grid_resolution1 <= 0)
        throw std::runtime_error("grid resolutions must be positive.");
    if (!(grid_cell_area > 0.0))
        throw std::runtime_error("grid_cell_area must be positive.");
    if (!(wavelength > 0.0))
        throw std::runtime_error("wavelength must be positive.");

    SceneCache &scene = get_scene(scene_handle);
    const int64_t state_count = state_edge_index.size(0);
    require_state_width(state_edge_pos, state_count, "state_edge_pos");
    require_state_width(state_edge_dir, state_count, "state_edge_dir");
    require_state_width(state_edge_t_min, state_count, "state_edge_t_min");
    require_state_width(state_edge_t_max, state_count, "state_edge_t_max");
    require_state_width(state_n0, state_count, "state_n0");
    require_state_width(state_n1, state_count, "state_n1");
    require_state_width(state_prim0, state_count, "state_prim0");
    require_state_width(state_prim1, state_count, "state_prim1");
    require_state_width(state_exterior_angle, state_count, "state_exterior_angle");
    require_state_width(state_src, state_count, "state_src");
    require_state_width(state_src_power, state_count, "state_src_power");
    require_state_width(state_wi, state_count, "state_wi");
    require_state_width(state_d0, state_count, "state_d0");
    const int64_t material_count = material_eta_r.size(0);
    if (material_count <= 0)
        throw std::runtime_error("material payload must not be empty.");
    if (material_sigma.size(0) != material_count ||
        material_mu_r.size(0) != material_count ||
        material_gain.size(0) != material_count ||
        material_valid.size(0) != material_count) {
        throw std::runtime_error("material payload fields must have matching widths.");
    }

    const int64_t cell_count = grid_resolution0 * grid_resolution1;
    const int64_t launch_count64 = state_count * cell_count;
    const int32_t launch_count = checked_i32(launch_count64, "launch_count");
    auto fopts = state_src.options();
    auto iopts = state_edge_index.options();
    at::Tensor direct_x_re = at::zeros({cell_count}, fopts);
    at::Tensor direct_x_im = at::zeros({cell_count}, fopts);
    at::Tensor direct_y_re = at::zeros({cell_count}, fopts);
    at::Tensor direct_y_im = at::zeros({cell_count}, fopts);
    at::Tensor direct_z_re = at::zeros({cell_count}, fopts);
    at::Tensor direct_z_im = at::zeros({cell_count}, fopts);
    at::Tensor multi_x_re = at::zeros({cell_count}, fopts);
    at::Tensor multi_x_im = at::zeros({cell_count}, fopts);
    at::Tensor multi_y_re = at::zeros({cell_count}, fopts);
    at::Tensor multi_y_im = at::zeros({cell_count}, fopts);
    at::Tensor multi_z_re = at::zeros({cell_count}, fopts);
    at::Tensor multi_z_im = at::zeros({cell_count}, fopts);
    at::Tensor direct_count = at::zeros({cell_count}, iopts);
    at::Tensor multi_count = at::zeros({cell_count}, iopts);
    at::Tensor visibility_reject_count = at::zeros({cell_count}, iopts);
    at::Tensor utd_reject_count = at::zeros({cell_count}, iopts);
    if (state_count == 0 || launch_count == 0) {
        return py::make_tuple(
            direct_x_re, direct_x_im, direct_y_re, direct_y_im, direct_z_re, direct_z_im,
            multi_x_re, multi_x_im, multi_y_re, multi_y_im, multi_z_re, multi_z_im,
            direct_count, multi_count, visibility_reject_count, utd_reject_count);
    }
    const bool staged_coherent_accum =
        launch_count64 >= kStagedDfrAccumMinSamples &&
        launch_count64 >= cell_count * kStagedDfrAccumMinSamplesPerCell;
    at::Tensor coherent_stage_key = staged_coherent_accum
        ? at::full({launch_count64}, -1, iopts)
        : at::Tensor();
    at::Tensor coherent_stage_value = staged_coherent_accum
        ? at::zeros({launch_count64, 8}, fopts)
        : at::Tensor();

    Vec3SoA state_edge_pos_soa = split_vec3(state_edge_pos);
    Vec3SoA state_edge_dir_soa = split_vec3(state_edge_dir);
    Vec3SoA state_n0_soa = split_vec3(state_n0);
    Vec3SoA state_n1_soa = split_vec3(state_n1);
    Vec3SoA state_src_soa = split_vec3(state_src);
    Vec3SoA state_wi_soa = split_vec3(state_wi);
    Vec3SoA state_d0_soa = split_vec3(state_d0);
    TriangleSoA tri = make_scene_triangle_soa(scene);
    at::Tensor active_contig = active_mask_for_states(active, state_count, "diffraction_coherent_accumulation_forward");
    at::Tensor state_prefix_depth = at::zeros({state_count}, iopts);

    DfrAccumParams params = {};
    params.primary_handle = scene.triangle_ias.traversable;
    params.secondary_handle = 0;
    params.split_mode = 0;
    params.n_rays = launch_count;
    params.active_mask = mask_ptr(active_contig);
    params.state_count = checked_i32(state_count, "state_count");
    params.state_edge_index = state_edge_index.data_ptr<int>();
    params.state_edge_pos_x = state_edge_pos_soa.x.data_ptr<float>();
    params.state_edge_pos_y = state_edge_pos_soa.y.data_ptr<float>();
    params.state_edge_pos_z = state_edge_pos_soa.z.data_ptr<float>();
    params.state_edge_dir_x = state_edge_dir_soa.x.data_ptr<float>();
    params.state_edge_dir_y = state_edge_dir_soa.y.data_ptr<float>();
    params.state_edge_dir_z = state_edge_dir_soa.z.data_ptr<float>();
    params.state_edge_t_min = state_edge_t_min.data_ptr<float>();
    params.state_edge_t_max = state_edge_t_max.data_ptr<float>();
    params.state_n0_x = state_n0_soa.x.data_ptr<float>();
    params.state_n0_y = state_n0_soa.y.data_ptr<float>();
    params.state_n0_z = state_n0_soa.z.data_ptr<float>();
    params.state_n1_x = state_n1_soa.x.data_ptr<float>();
    params.state_n1_y = state_n1_soa.y.data_ptr<float>();
    params.state_n1_z = state_n1_soa.z.data_ptr<float>();
    params.state_prim0 = state_prim0.data_ptr<int>();
    params.state_prim1 = state_prim1.data_ptr<int>();
    params.state_exterior_angle = state_exterior_angle.data_ptr<float>();
    params.state_src_x = state_src_soa.x.data_ptr<float>();
    params.state_src_y = state_src_soa.y.data_ptr<float>();
    params.state_src_z = state_src_soa.z.data_ptr<float>();
    params.state_src_power = state_src_power.data_ptr<float>();
    params.state_wi_x = state_wi_soa.x.data_ptr<float>();
    params.state_wi_y = state_wi_soa.y.data_ptr<float>();
    params.state_wi_z = state_wi_soa.z.data_ptr<float>();
    params.state_d0_x = state_d0_soa.x.data_ptr<float>();
    params.state_d0_y = state_d0_soa.y.data_ptr<float>();
    params.state_d0_z = state_d0_soa.z.data_ptr<float>();
    params.state_prefix_depth = state_prefix_depth.data_ptr<int>();
    params.grid_axis = checked_i32(grid_axis, "grid_axis");
    params.grid_position = static_cast<float>(grid_position);
    params.grid_coord0_min = static_cast<float>(grid_coord0_min);
    params.grid_coord0_max = static_cast<float>(grid_coord0_max);
    params.grid_coord1_min = static_cast<float>(grid_coord1_min);
    params.grid_coord1_max = static_cast<float>(grid_coord1_max);
    params.grid_resolution0 = checked_i32(grid_resolution0, "grid_resolution0");
    params.grid_resolution1 = checked_i32(grid_resolution1, "grid_resolution1");
    params.grid_cell_area = static_cast<float>(grid_cell_area);
    params.tri_p0_x = tri.p0_x.data_ptr<float>();
    params.tri_p0_y = tri.p0_y.data_ptr<float>();
    params.tri_p0_z = tri.p0_z.data_ptr<float>();
    params.tri_e1_x = tri.e1_x.data_ptr<float>();
    params.tri_e1_y = tri.e1_y.data_ptr<float>();
    params.tri_e1_z = tri.e1_z.data_ptr<float>();
    params.tri_e2_x = tri.e2_x.data_ptr<float>();
    params.tri_e2_y = tri.e2_y.data_ptr<float>();
    params.tri_e2_z = tri.e2_z.data_ptr<float>();
    params.tri_fn_x = tri.fn_x.data_ptr<float>();
    params.tri_fn_y = tri.fn_y.data_ptr<float>();
    params.tri_fn_z = tri.fn_z.data_ptr<float>();
    params.face_offsets = tri.face_offsets.data_ptr<int>();
    params.n_meshes = checked_i32(scene.meshes.size(), "n_meshes");
    params.n_triangles = tri.n_triangles;
    params.material_eta_r = material_eta_r.data_ptr<float>();
    params.material_sigma = material_sigma.data_ptr<float>();
    params.material_mu_r = material_mu_r.data_ptr<float>();
    params.material_gain = material_gain.data_ptr<float>();
    params.material_valid = mask_ptr(material_valid);
    params.material_count = checked_i32(material_count, "material_count");
    params.wavelength = static_cast<float>(wavelength);
    params.k = static_cast<float>(2.0 * 3.14159265358979323846 / wavelength);
    params.max_order = 1;
    params.receiver_model = RAYDTORCH_DFR_MATCHED_ISO;
    params.select_diffraction_point = select_diffraction_point ? 1 : 0;
    params.prefilter_visibility = prefilter_visibility ? 1 : 0;
    params.collect_debug_counts = 1;
    params.out_direct_count = direct_count.data_ptr<int>();
    params.out_direct_field_x_re = direct_x_re.data_ptr<float>();
    params.out_direct_field_x_im = direct_x_im.data_ptr<float>();
    params.out_direct_field_y_re = direct_y_re.data_ptr<float>();
    params.out_direct_field_y_im = direct_y_im.data_ptr<float>();
    params.out_direct_field_z_re = direct_z_re.data_ptr<float>();
    params.out_direct_field_z_im = direct_z_im.data_ptr<float>();
    params.out_multi_field_x_re = multi_x_re.data_ptr<float>();
    params.out_multi_field_x_im = multi_x_im.data_ptr<float>();
    params.out_multi_field_y_re = multi_y_re.data_ptr<float>();
    params.out_multi_field_y_im = multi_y_im.data_ptr<float>();
    params.out_multi_field_z_re = multi_z_re.data_ptr<float>();
    params.out_multi_field_z_im = multi_z_im.data_ptr<float>();
    params.out_multi_count = multi_count.data_ptr<int>();
    params.out_visibility_reject_count = visibility_reject_count.data_ptr<int>();
    params.out_utd_reject_count = utd_reject_count.data_ptr<int>();
    params.coherent_stage_key =
        staged_coherent_accum ? coherent_stage_key.data_ptr<int>() : nullptr;
    params.coherent_stage_value = staged_coherent_accum
        ? reinterpret_cast<DfrCoherentStagedValue *>(coherent_stage_value.data_ptr<float>())
        : nullptr;

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    auto pipeline = optix_pipeline_for_scene(scene, diffraction_accumulation_pipeline_config());
    pipeline->launch(11, params, static_cast<unsigned int>(launch_count), torch_ctx.stream);
    if (staged_coherent_accum) {
        reduce_dfr_coherent_accum_staged_cuda(
            launch_count64,
            cell_count,
            coherent_stage_key,
            coherent_stage_value,
            direct_x_re,
            direct_x_im,
            direct_y_re,
            direct_y_im,
            direct_z_re,
            direct_z_im,
            multi_x_re,
            multi_x_im,
            multi_y_re,
            multi_y_im,
            multi_z_re,
            multi_z_im,
            direct_count,
            multi_count);
    }

    return py::make_tuple(
        direct_x_re.reshape({grid_resolution1, grid_resolution0}),
        direct_x_im.reshape({grid_resolution1, grid_resolution0}),
        direct_y_re.reshape({grid_resolution1, grid_resolution0}),
        direct_y_im.reshape({grid_resolution1, grid_resolution0}),
        direct_z_re.reshape({grid_resolution1, grid_resolution0}),
        direct_z_im.reshape({grid_resolution1, grid_resolution0}),
        multi_x_re.reshape({grid_resolution1, grid_resolution0}),
        multi_x_im.reshape({grid_resolution1, grid_resolution0}),
        multi_y_re.reshape({grid_resolution1, grid_resolution0}),
        multi_y_im.reshape({grid_resolution1, grid_resolution0}),
        multi_z_re.reshape({grid_resolution1, grid_resolution0}),
        multi_z_im.reshape({grid_resolution1, grid_resolution0}),
        direct_count.reshape({grid_resolution1, grid_resolution0}),
        multi_count.reshape({grid_resolution1, grid_resolution0}),
        visibility_reject_count.reshape({grid_resolution1, grid_resolution0}),
        utd_reject_count.reshape({grid_resolution1, grid_resolution0}));
}


void bind_diffraction_ops(py::module_ &m) {
    m.def("diffraction_paths_order1_forward", &diffraction_paths_order1_forward_op);
    m.def("diffraction_accumulation_forward", &diffraction_accumulation_forward_op);
    m.def("diffraction_accumulation_direct_backward", &diffraction_accumulation_direct_backward_op);
    m.def("diffraction_accumulation_direct_jvp", &diffraction_accumulation_direct_jvp_op);
    m.def("diffraction_accumulation_chain_backward", &diffraction_accumulation_chain_backward_op);
    m.def("diffraction_accumulation_chain_jvp", &diffraction_accumulation_chain_jvp_op);
    m.def("diffraction_coherent_accumulation_forward", &diffraction_coherent_accumulation_forward_op);
}

} // namespace raydtorch
