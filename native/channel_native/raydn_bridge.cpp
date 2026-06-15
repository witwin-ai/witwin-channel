#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using VisibilityForwardFn = void (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *,
    at::Tensor *,
    at::Tensor *);

using SceneCreateFn = int64_t (*)(
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const int64_t *,
    int64_t);

using SceneDestroyFn = void (*)(int64_t);

using SceneEdgeRecordsFn = int64_t (*)(
    int64_t,
    at::Tensor *,
    int64_t);

using IntersectForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    at::Tensor *,
    int64_t);

using ReflectionEpcPathsForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    int64_t,
    at::Tensor *,
    int64_t);

using ReflectionAccumulationForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    int64_t,
    double,
    double,
    double,
    double,
    double,
    int64_t,
    int64_t,
    double,
    double,
    bool,
    bool,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    bool,
    at::Tensor *,
    int64_t);

using DiffractionDiscoverEdgesFn = void (*)(
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *);

using DiffractionDiscoverEdgesCountedFn = void (*)(
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *);

using DiffractionPathsOrder1ForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    int64_t,
    double,
    at::Tensor *,
    int64_t);

using DiffractionAccumulationForwardFn = int64_t (*)(
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    int64_t,
    double,
    double,
    double,
    double,
    double,
    int64_t,
    int64_t,
    double,
    double,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    const at::Tensor *,
    int64_t,
    const at::Tensor *,
    const at::Tensor *,
    at::Tensor *,
    int64_t);

std::mutex g_raydn_api_mutex;
std::unordered_map<std::uintptr_t, VisibilityForwardFn> g_visibility_forward_cache;
std::unordered_map<std::uintptr_t, SceneCreateFn> g_scene_create_cache;
std::unordered_map<std::uintptr_t, SceneDestroyFn> g_scene_destroy_cache;
std::unordered_map<std::uintptr_t, SceneEdgeRecordsFn> g_scene_edge_records_cache;
std::unordered_map<std::uintptr_t, IntersectForwardFn> g_intersect_forward_cache;
std::unordered_map<std::uintptr_t, ReflectionEpcPathsForwardFn> g_reflection_epc_paths_forward_cache;
std::unordered_map<std::uintptr_t, ReflectionAccumulationForwardFn> g_reflection_accumulation_forward_cache;
std::unordered_map<std::uintptr_t, DiffractionDiscoverEdgesFn> g_diffraction_discover_edges_cache;
std::unordered_map<std::uintptr_t, DiffractionDiscoverEdgesCountedFn> g_diffraction_discover_edges_counted_cache;
std::unordered_map<std::uintptr_t, DiffractionPathsOrder1ForwardFn> g_diffraction_paths_order1_forward_cache;
std::unordered_map<std::uintptr_t, DiffractionAccumulationForwardFn> g_diffraction_accumulation_forward_cache;

#if defined(_WIN32)
void *load_symbol(std::uintptr_t module_handle, const char *symbol) {
    if (module_handle == 0)
        throw std::runtime_error("loaded RayDN native module handle is required");
    HMODULE module = reinterpret_cast<HMODULE>(module_handle);
    FARPROC proc = GetProcAddress(module, symbol);
    if (proc == nullptr) {
        throw std::runtime_error(
            std::string("RayDN native module does not export ") + symbol + ": Windows error " +
            std::to_string(GetLastError()));
    }
    return reinterpret_cast<void *>(proc);
}
#else
void *load_symbol(std::uintptr_t module_handle, const char *symbol) {
    if (module_handle == 0)
        throw std::runtime_error("loaded RayDN native module handle is required");
    void *module = reinterpret_cast<void *>(module_handle);
    if (module == nullptr)
        throw std::runtime_error("loaded RayDN native module handle is required");
    dlerror();
    void *proc = dlsym(module, symbol);
    const char *error = dlerror();
    if (error != nullptr)
        throw std::runtime_error(std::string("RayDN native module does not export ") + symbol + ": " + error);
    return proc;
}
#endif

VisibilityForwardFn raydn_visibility_forward_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_visibility_forward_cache.find(module_handle);
    if (cached != g_visibility_forward_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_visibility_forward");
    auto fn = reinterpret_cast<VisibilityForwardFn>(symbol);
    g_visibility_forward_cache.emplace(module_handle, fn);
    return fn;
}

SceneCreateFn raydn_scene_create_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_scene_create_cache.find(module_handle);
    if (cached != g_scene_create_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_scene_create");
    auto fn = reinterpret_cast<SceneCreateFn>(symbol);
    g_scene_create_cache.emplace(module_handle, fn);
    return fn;
}

SceneDestroyFn raydn_scene_destroy_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_scene_destroy_cache.find(module_handle);
    if (cached != g_scene_destroy_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_scene_destroy");
    auto fn = reinterpret_cast<SceneDestroyFn>(symbol);
    g_scene_destroy_cache.emplace(module_handle, fn);
    return fn;
}

SceneEdgeRecordsFn raydn_scene_edge_records_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_scene_edge_records_cache.find(module_handle);
    if (cached != g_scene_edge_records_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_scene_edge_records");
    auto fn = reinterpret_cast<SceneEdgeRecordsFn>(symbol);
    g_scene_edge_records_cache.emplace(module_handle, fn);
    return fn;
}

IntersectForwardFn raydn_intersect_forward_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_intersect_forward_cache.find(module_handle);
    if (cached != g_intersect_forward_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_intersect_forward");
    auto fn = reinterpret_cast<IntersectForwardFn>(symbol);
    g_intersect_forward_cache.emplace(module_handle, fn);
    return fn;
}

ReflectionEpcPathsForwardFn raydn_reflection_epc_paths_forward_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_reflection_epc_paths_forward_cache.find(module_handle);
    if (cached != g_reflection_epc_paths_forward_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_reflection_epc_paths_forward");
    auto fn = reinterpret_cast<ReflectionEpcPathsForwardFn>(symbol);
    g_reflection_epc_paths_forward_cache.emplace(module_handle, fn);
    return fn;
}

ReflectionAccumulationForwardFn raydn_reflection_accumulation_forward_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_reflection_accumulation_forward_cache.find(module_handle);
    if (cached != g_reflection_accumulation_forward_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_reflection_accumulation_forward");
    auto fn = reinterpret_cast<ReflectionAccumulationForwardFn>(symbol);
    g_reflection_accumulation_forward_cache.emplace(module_handle, fn);
    return fn;
}

DiffractionDiscoverEdgesFn raydn_diffraction_discover_edges_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_diffraction_discover_edges_cache.find(module_handle);
    if (cached != g_diffraction_discover_edges_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_diffraction_discover_edges");
    auto fn = reinterpret_cast<DiffractionDiscoverEdgesFn>(symbol);
    g_diffraction_discover_edges_cache.emplace(module_handle, fn);
    return fn;
}

DiffractionDiscoverEdgesCountedFn raydn_diffraction_discover_edges_counted_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_diffraction_discover_edges_counted_cache.find(module_handle);
    if (cached != g_diffraction_discover_edges_counted_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_diffraction_discover_edges_counted");
    auto fn = reinterpret_cast<DiffractionDiscoverEdgesCountedFn>(symbol);
    g_diffraction_discover_edges_counted_cache.emplace(module_handle, fn);
    return fn;
}

DiffractionPathsOrder1ForwardFn raydn_diffraction_paths_order1_forward_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_diffraction_paths_order1_forward_cache.find(module_handle);
    if (cached != g_diffraction_paths_order1_forward_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_diffraction_paths_order1_forward");
    auto fn = reinterpret_cast<DiffractionPathsOrder1ForwardFn>(symbol);
    g_diffraction_paths_order1_forward_cache.emplace(module_handle, fn);
    return fn;
}

DiffractionAccumulationForwardFn raydn_diffraction_accumulation_forward_fn(std::uintptr_t module_handle) {
    if (module_handle == 0)
        throw std::runtime_error("RayDN native module handle is required");
    std::lock_guard<std::mutex> lock(g_raydn_api_mutex);
    auto cached = g_diffraction_accumulation_forward_cache.find(module_handle);
    if (cached != g_diffraction_accumulation_forward_cache.end())
        return cached->second;
    void *symbol = load_symbol(module_handle, "raydn_native_diffraction_accumulation_forward");
    auto fn = reinterpret_cast<DiffractionAccumulationForwardFn>(symbol);
    g_diffraction_accumulation_forward_cache.emplace(module_handle, fn);
    return fn;
}

const at::Tensor *optional_tensor(pybind11::object value, at::Tensor &storage) {
    if (value.is_none())
        return nullptr;
    storage = value.cast<at::Tensor>();
    if (!storage.defined())
        return nullptr;
    return &storage;
}

struct RaydnSceneOwner {
    int64_t scene_handle = 0;
    std::uintptr_t module_handle = 0;
};

void destroy_raydn_scene_capsule(PyObject *capsule) {
    void *raw = PyCapsule_GetPointer(capsule, "channel_native.raydn_scene");
    auto *owner = reinterpret_cast<RaydnSceneOwner *>(raw);
    if (owner == nullptr)
        return;
    try {
        if (owner->scene_handle != 0)
            raydn_scene_destroy_fn(owner->module_handle)(owner->scene_handle);
    } catch (...) {
    }
    delete owner;
}

} // namespace

using PathBlockTuple = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

std::vector<at::Tensor> cn_deterministic_diffraction_state_pack_selected_cuda(
    at::Tensor selected,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor line_min,
    at::Tensor line_max,
    at::Tensor n0,
    at::Tensor n1,
    at::Tensor face0,
    at::Tensor face1,
    at::Tensor exterior_angle,
    at::Tensor tx,
    at::Tensor tx_power,
    int64_t tx_power_index);
PathBlockTuple cn_path_diffraction_block_cuda(
    at::Tensor valid,
    at::Tensor rx_id,
    at::Tensor depth,
    at::Tensor edge_id,
    at::Tensor delay,
    at::Tensor field_x_re,
    at::Tensor field_x_im,
    at::Tensor field_y_re,
    at::Tensor field_y_im,
    at::Tensor field_z_re,
    at::Tensor field_z_im,
    int64_t tx_index);
PathBlockTuple cn_path_finalize_blocks_cuda(
    std::vector<at::Tensor> valid_blocks,
    std::vector<at::Tensor> tx_id_blocks,
    std::vector<at::Tensor> rx_id_blocks,
    std::vector<at::Tensor> depth_blocks,
    std::vector<at::Tensor> component_id_blocks,
    std::vector<at::Tensor> primitive_id_blocks,
    std::vector<at::Tensor> edge_id_blocks,
    std::vector<at::Tensor> path_length_blocks,
    std::vector<at::Tensor> delay_blocks,
    std::vector<at::Tensor> path_gain_blocks,
    int64_t max_paths,
    int64_t tx_count,
    int64_t max_depth);

namespace {

pybind11::dict path_block_dict(const PathBlockTuple& block) {
    pybind11::dict out;
    out["valid"] = std::get<0>(block);
    out["tx_id"] = std::get<1>(block);
    out["rx_id"] = std::get<2>(block);
    out["depth"] = std::get<3>(block);
    out["component_id"] = std::get<4>(block);
    out["primitive_id"] = std::get<5>(block);
    out["edge_id"] = std::get<6>(block);
    out["path_length_m"] = std::get<7>(block);
    out["delay_s"] = std::get<8>(block);
    out["path_gain"] = std::get<9>(block);
    return out;
}

void append_path_block(
    const PathBlockTuple& block,
    std::vector<at::Tensor>& valid_blocks,
    std::vector<at::Tensor>& tx_id_blocks,
    std::vector<at::Tensor>& rx_id_blocks,
    std::vector<at::Tensor>& depth_blocks,
    std::vector<at::Tensor>& component_id_blocks,
    std::vector<at::Tensor>& primitive_id_blocks,
    std::vector<at::Tensor>& edge_id_blocks,
    std::vector<at::Tensor>& path_length_blocks,
    std::vector<at::Tensor>& delay_blocks,
    std::vector<at::Tensor>& path_gain_blocks) {
    valid_blocks.push_back(std::get<0>(block));
    tx_id_blocks.push_back(std::get<1>(block));
    rx_id_blocks.push_back(std::get<2>(block));
    depth_blocks.push_back(std::get<3>(block));
    component_id_blocks.push_back(std::get<4>(block));
    primitive_id_blocks.push_back(std::get<5>(block));
    edge_id_blocks.push_back(std::get<6>(block));
    path_length_blocks.push_back(std::get<7>(block));
    delay_blocks.push_back(std::get<8>(block));
    path_gain_blocks.push_back(std::get<9>(block));
}

void check_vec3_table(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.dim() == 2, name, " must have rank 2");
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_flat_tensor(const at::Tensor& tensor, const char* name, c10::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == 1, name, " must have rank 1");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

pybind11::tuple cn_raydn_scene_create(
    std::vector<torch::Tensor> vertices,
    std::vector<torch::Tensor> faces,
    std::vector<torch::Tensor> uv,
    std::vector<torch::Tensor> face_uv,
    std::vector<torch::Tensor> to_world_left,
    std::vector<torch::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags,
    std::uintptr_t raydn_module_handle) {
    const size_t mesh_count = vertices.size();
    if (mesh_count == 0)
        throw std::runtime_error("raydn_scene_create requires at least one mesh");
    if (faces.size() != mesh_count ||
        uv.size() != mesh_count ||
        face_uv.size() != mesh_count ||
        to_world_left.size() != mesh_count ||
        to_world_right.size() != mesh_count ||
        mesh_flags.size() != mesh_count) {
        throw std::runtime_error("raydn_scene_create input lists must have the same length");
    }
    int64_t scene_handle = raydn_scene_create_fn(raydn_module_handle)(
        vertices.data(),
        faces.data(),
        uv.data(),
        face_uv.data(),
        to_world_left.data(),
        to_world_right.data(),
        mesh_flags.data(),
        static_cast<int64_t>(mesh_count));
    auto *owner = new RaydnSceneOwner{scene_handle, raydn_module_handle};
    pybind11::capsule capsule(owner, "channel_native.raydn_scene", destroy_raydn_scene_capsule);
    return pybind11::make_tuple(scene_handle, capsule);
}

pybind11::tuple cn_raydn_scene_edge_records(
    int64_t scene_handle,
    std::uintptr_t raydn_module_handle) {
    constexpr int64_t kOutputCount = 12;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_scene_edge_records_fn(raydn_module_handle)(
        scene_handle,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN scene edge records returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_bdpt_intersect_forward(
    int64_t scene_handle,
    torch::Tensor ray_o,
    torch::Tensor ray_d,
    torch::Tensor ray_tmax,
    pybind11::object active,
    int64_t flags,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 10;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_intersect_forward_fn(raydn_module_handle)(
        scene_handle,
        &ray_o,
        &ray_d,
        &ray_tmax,
        active_ptr,
        flags,
        outputs.data(),
        kOutputCount);
    if (output_count != kOutputCount)
        throw std::runtime_error("RayDN intersect forward returned an unexpected output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::tuple cn_bdpt_visibility_forward(
    int64_t scene_handle,
    torch::Tensor start,
    torch::Tensor end,
    pybind11::object active,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    at::Tensor visible;
    at::Tensor blocker_prim;
    at::Tensor tape_t;
    raydn_visibility_forward_fn(raydn_module_handle)(
        scene_handle,
        &start,
        &end,
        active_ptr,
        &visible,
        &blocker_prim,
        &tape_t);
    return pybind11::make_tuple(visible, blocker_prim, tape_t);
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
    int64_t visibility_ignore_mode,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 6;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_epc_paths_forward_fn(raydn_module_handle)(
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
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN reflection EPC path export returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
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
    bool streaming_los_enabled,
    std::uintptr_t raydn_module_handle) {
    constexpr int64_t kOutputCount = 18;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_reflection_accumulation_forward_fn(raydn_module_handle)(
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

torch::Tensor cn_bdpt_diffraction_discover_edges(
    torch::Tensor tx_pos,
    torch::Tensor ray_dir,
    torch::Tensor prim_index,
    torch::Tensor hit_p,
    torch::Tensor hit_n,
    torch::Tensor hit_geo_n,
    torch::Tensor triangle_edge_count,
    torch::Tensor triangle_edge_indices,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor edge_n0,
    torch::Tensor edge_nn,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1,
    std::uintptr_t raydn_module_handle) {
    at::Tensor out;
    raydn_diffraction_discover_edges_fn(raydn_module_handle)(
        &tx_pos,
        &ray_dir,
        &prim_index,
        &hit_p,
        &hit_n,
        &hit_geo_n,
        &triangle_edge_count,
        &triangle_edge_indices,
        &edge_pos,
        &edge_dir,
        &edge_n0,
        &edge_nn,
        &edge_line_min,
        &edge_line_max,
        &edge_adjacent_face1,
        &out);
    return out;
}

torch::Tensor cn_bdpt_diffraction_discover_edges_counted(
    torch::Tensor tx_pos,
    torch::Tensor ray_dir,
    torch::Tensor prim_index,
    torch::Tensor hit_p,
    torch::Tensor hit_n,
    torch::Tensor hit_geo_n,
    torch::Tensor hit_count,
    torch::Tensor triangle_edge_count,
    torch::Tensor triangle_edge_indices,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor edge_n0,
    torch::Tensor edge_nn,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    torch::Tensor edge_adjacent_face1,
    std::uintptr_t raydn_module_handle) {
    at::Tensor out;
    raydn_diffraction_discover_edges_counted_fn(raydn_module_handle)(
        &tx_pos,
        &ray_dir,
        &prim_index,
        &hit_p,
        &hit_n,
        &hit_geo_n,
        &hit_count,
        &triangle_edge_count,
        &triangle_edge_indices,
        &edge_pos,
        &edge_dir,
        &edge_n0,
        &edge_nn,
        &edge_line_min,
        &edge_line_max,
        &edge_adjacent_face1,
        &out);
    return out;
}

pybind11::tuple cn_raydn_diffraction_paths_order1_forward(
    int64_t scene_handle,
    torch::Tensor tx_pos,
    torch::Tensor rx_pos,
    pybind11::object active,
    torch::Tensor state_edge_index,
    torch::Tensor state_edge_pos,
    torch::Tensor state_edge_dir,
    torch::Tensor state_edge_t_min,
    torch::Tensor state_edge_t_max,
    torch::Tensor state_n0,
    torch::Tensor state_n1,
    torch::Tensor state_prim0,
    torch::Tensor state_prim1,
    torch::Tensor state_exterior_angle,
    torch::Tensor state_src,
    torch::Tensor state_src_power,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t state_limit,
    int64_t capacity,
    double wavelength,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    constexpr int64_t kOutputCount = 18;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_diffraction_paths_order1_forward_fn(raydn_module_handle)(
        scene_handle,
        &tx_pos,
        &rx_pos,
        active_ptr,
        &state_edge_index,
        &state_edge_pos,
        &state_edge_dir,
        &state_edge_t_min,
        &state_edge_t_max,
        &state_n0,
        &state_n1,
        &state_prim0,
        &state_prim1,
        &state_exterior_angle,
        &state_src,
        &state_src_power,
        &material_gain,
        &material_valid,
        state_limit,
        capacity,
        wavelength,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN diffraction order-1 path export returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}

pybind11::dict cn_path_diffraction_paths_order1(
    int64_t scene_handle,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor rx_positions,
    torch::Tensor selected,
    torch::Tensor edge_pos,
    torch::Tensor edge_dir,
    torch::Tensor line_min,
    torch::Tensor line_max,
    torch::Tensor n0,
    torch::Tensor n1,
    torch::Tensor face0,
    torch::Tensor face1,
    torch::Tensor exterior_angle,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    double wavelength,
    std::uintptr_t raydn_module_handle) {
    check_vec3_table(tx_positions, "tx_positions");
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(rx_positions, "rx_positions");
    check_flat_tensor(selected, "selected", at::kBool);
    check_vec3_table(edge_pos, "edge_pos");
    check_vec3_table(edge_dir, "edge_dir");
    check_flat_tensor(line_min, "line_min", at::kFloat);
    check_flat_tensor(line_max, "line_max", at::kFloat);
    check_vec3_table(n0, "n0");
    check_vec3_table(n1, "n1");
    check_flat_tensor(face0, "face0", at::kInt);
    check_flat_tensor(face1, "face1", at::kInt);
    check_flat_tensor(exterior_angle, "exterior_angle", at::kFloat);
    check_flat_tensor(material_gain, "material_gain", at::kFloat);
    check_flat_tensor(material_valid, "material_valid", at::kBool);
    TORCH_CHECK(tx_power.size(0) == tx_positions.size(0), "tx_power must match tx_positions");
    TORCH_CHECK(edge_dir.sizes() == edge_pos.sizes(), "edge_dir must match edge_pos");
    TORCH_CHECK(n0.sizes() == edge_pos.sizes(), "n0 must match edge_pos");
    TORCH_CHECK(n1.sizes() == edge_pos.sizes(), "n1 must match edge_pos");
    TORCH_CHECK(selected.size(0) == edge_pos.size(0), "selected must match edge_pos");
    TORCH_CHECK(line_min.size(0) == edge_pos.size(0), "line_min must match edge_pos");
    TORCH_CHECK(line_max.size(0) == edge_pos.size(0), "line_max must match edge_pos");
    TORCH_CHECK(face0.size(0) == edge_pos.size(0), "face0 must match edge_pos");
    TORCH_CHECK(face1.size(0) == edge_pos.size(0), "face1 must match edge_pos");
    TORCH_CHECK(exterior_angle.size(0) == edge_pos.size(0), "exterior_angle must match edge_pos");
    TORCH_CHECK(material_valid.size(0) == material_gain.size(0), "material_valid must match material_gain");
    TORCH_CHECK(wavelength > 0.0, "wavelength must be positive");
    const int device = tx_positions.get_device();
    for (const auto& tensor : {
             tx_power,
             rx_positions,
             selected,
             edge_pos,
             edge_dir,
             line_min,
             line_max,
             n0,
             n1,
             face0,
             face1,
             exterior_angle,
             material_gain,
             material_valid,
         }) {
        TORCH_CHECK(tensor.get_device() == device, "diffraction path tensors must share one CUDA device");
    }

    const int64_t tx_count = tx_positions.size(0);
    std::vector<at::Tensor> valid_blocks;
    std::vector<at::Tensor> tx_id_blocks;
    std::vector<at::Tensor> rx_id_blocks;
    std::vector<at::Tensor> depth_blocks;
    std::vector<at::Tensor> component_id_blocks;
    std::vector<at::Tensor> primitive_id_blocks;
    std::vector<at::Tensor> edge_id_blocks;
    std::vector<at::Tensor> path_length_blocks;
    std::vector<at::Tensor> delay_blocks;
    std::vector<at::Tensor> path_gain_blocks;
    valid_blocks.reserve(static_cast<size_t>(tx_count));
    tx_id_blocks.reserve(static_cast<size_t>(tx_count));
    rx_id_blocks.reserve(static_cast<size_t>(tx_count));
    depth_blocks.reserve(static_cast<size_t>(tx_count));
    component_id_blocks.reserve(static_cast<size_t>(tx_count));
    primitive_id_blocks.reserve(static_cast<size_t>(tx_count));
    edge_id_blocks.reserve(static_cast<size_t>(tx_count));
    path_length_blocks.reserve(static_cast<size_t>(tx_count));
    delay_blocks.reserve(static_cast<size_t>(tx_count));
    path_gain_blocks.reserve(static_cast<size_t>(tx_count));

    constexpr int64_t kOutputCount = 18;
    for (int64_t tx_index = 0; tx_index < tx_count; ++tx_index) {
        at::Tensor tx = tx_positions.select(0, tx_index);
        std::vector<at::Tensor> states = cn_deterministic_diffraction_state_pack_selected_cuda(
            selected,
            edge_pos,
            edge_dir,
            line_min,
            line_max,
            n0,
            n1,
            face0,
            face1,
            exterior_angle,
            tx,
            tx_power,
            tx_index);
        TORCH_CHECK(states.size() == 12, "diffraction state pack must return 12 tensors");

        at::Tensor tx_view = tx_positions.narrow(0, tx_index, 1);
        const int64_t state_limit = states[0].size(0);
        const int64_t capacity = rx_positions.size(0) * state_limit;
        std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
        const int64_t output_count = raydn_diffraction_paths_order1_forward_fn(raydn_module_handle)(
            scene_handle,
            &tx_view,
            &rx_positions,
            nullptr,
            &states[0],
            &states[1],
            &states[2],
            &states[3],
            &states[4],
            &states[5],
            &states[6],
            &states[7],
            &states[8],
            &states[9],
            &states[10],
            &states[11],
            &material_gain,
            &material_valid,
            state_limit,
            capacity,
            wavelength,
            outputs.data(),
            kOutputCount);
        TORCH_CHECK(output_count == kOutputCount, "RayDN order-1 diffraction path export returned an unexpected output count");
        PathBlockTuple block = cn_path_diffraction_block_cuda(
            outputs[1],
            outputs[3],
            outputs[4],
            outputs[5],
            outputs[8],
            outputs[9],
            outputs[10],
            outputs[11],
            outputs[12],
            outputs[13],
            outputs[14],
            tx_index);
        append_path_block(
            block,
            valid_blocks,
            tx_id_blocks,
            rx_id_blocks,
            depth_blocks,
            component_id_blocks,
            primitive_id_blocks,
            edge_id_blocks,
            path_length_blocks,
            delay_blocks,
            path_gain_blocks);
    }

    return path_block_dict(cn_path_finalize_blocks_cuda(
        valid_blocks,
        tx_id_blocks,
        rx_id_blocks,
        depth_blocks,
        component_id_blocks,
        primitive_id_blocks,
        edge_id_blocks,
        path_length_blocks,
        delay_blocks,
        path_gain_blocks,
        -1,
        tx_count,
        1));
}

pybind11::tuple cn_bdpt_diffraction_accumulation_forward(
    int64_t scene_handle,
    pybind11::object active,
    torch::Tensor state_edge_index,
    torch::Tensor state_edge_pos,
    torch::Tensor state_edge_dir,
    torch::Tensor state_edge_t_min,
    torch::Tensor state_edge_t_max,
    torch::Tensor state_n0,
    torch::Tensor state_n1,
    torch::Tensor state_prim0,
    torch::Tensor state_prim1,
    torch::Tensor state_exterior_angle,
    torch::Tensor state_src,
    torch::Tensor state_src_power,
    pybind11::object state_wi,
    pybind11::object state_d0,
    torch::Tensor material_eta_r,
    torch::Tensor material_sigma,
    torch::Tensor material_mu_r,
    torch::Tensor material_gain,
    torch::Tensor material_valid,
    int64_t state_limit,
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
    int64_t recursive_state_limit,
    pybind11::object recursive_active,
    pybind11::object recursive_state_edge_index,
    pybind11::object recursive_state_edge_pos,
    pybind11::object recursive_state_edge_dir,
    pybind11::object recursive_state_edge_t_min,
    pybind11::object recursive_state_edge_t_max,
    pybind11::object recursive_state_n0,
    pybind11::object recursive_state_n1,
    pybind11::object recursive_state_prim0,
    pybind11::object recursive_state_prim1,
    pybind11::object recursive_state_exterior_angle,
    int64_t export_tape,
    pybind11::object sample_state_index,
    pybind11::object sample_edge_weight,
    std::uintptr_t raydn_module_handle) {
    at::Tensor active_storage;
    at::Tensor state_wi_storage;
    at::Tensor state_d0_storage;
    at::Tensor recursive_active_storage;
    at::Tensor recursive_state_edge_index_storage;
    at::Tensor recursive_state_edge_pos_storage;
    at::Tensor recursive_state_edge_dir_storage;
    at::Tensor recursive_state_edge_t_min_storage;
    at::Tensor recursive_state_edge_t_max_storage;
    at::Tensor recursive_state_n0_storage;
    at::Tensor recursive_state_n1_storage;
    at::Tensor recursive_state_prim0_storage;
    at::Tensor recursive_state_prim1_storage;
    at::Tensor recursive_state_exterior_angle_storage;
    at::Tensor sample_state_index_storage;
    at::Tensor sample_edge_weight_storage;

    const at::Tensor *active_ptr = optional_tensor(std::move(active), active_storage);
    const at::Tensor *state_wi_ptr = optional_tensor(std::move(state_wi), state_wi_storage);
    const at::Tensor *state_d0_ptr = optional_tensor(std::move(state_d0), state_d0_storage);
    const at::Tensor *recursive_active_ptr = optional_tensor(std::move(recursive_active), recursive_active_storage);
    const at::Tensor *recursive_state_edge_index_ptr =
        optional_tensor(std::move(recursive_state_edge_index), recursive_state_edge_index_storage);
    const at::Tensor *recursive_state_edge_pos_ptr =
        optional_tensor(std::move(recursive_state_edge_pos), recursive_state_edge_pos_storage);
    const at::Tensor *recursive_state_edge_dir_ptr =
        optional_tensor(std::move(recursive_state_edge_dir), recursive_state_edge_dir_storage);
    const at::Tensor *recursive_state_edge_t_min_ptr =
        optional_tensor(std::move(recursive_state_edge_t_min), recursive_state_edge_t_min_storage);
    const at::Tensor *recursive_state_edge_t_max_ptr =
        optional_tensor(std::move(recursive_state_edge_t_max), recursive_state_edge_t_max_storage);
    const at::Tensor *recursive_state_n0_ptr = optional_tensor(std::move(recursive_state_n0), recursive_state_n0_storage);
    const at::Tensor *recursive_state_n1_ptr = optional_tensor(std::move(recursive_state_n1), recursive_state_n1_storage);
    const at::Tensor *recursive_state_prim0_ptr =
        optional_tensor(std::move(recursive_state_prim0), recursive_state_prim0_storage);
    const at::Tensor *recursive_state_prim1_ptr =
        optional_tensor(std::move(recursive_state_prim1), recursive_state_prim1_storage);
    const at::Tensor *recursive_state_exterior_angle_ptr =
        optional_tensor(std::move(recursive_state_exterior_angle), recursive_state_exterior_angle_storage);
    const at::Tensor *sample_state_index_ptr =
        optional_tensor(std::move(sample_state_index), sample_state_index_storage);
    const at::Tensor *sample_edge_weight_ptr =
        optional_tensor(std::move(sample_edge_weight), sample_edge_weight_storage);

    constexpr int64_t kOutputCount = 19;
    std::array<at::Tensor, static_cast<size_t>(kOutputCount)> outputs;
    int64_t output_count = raydn_diffraction_accumulation_forward_fn(raydn_module_handle)(
        scene_handle,
        active_ptr,
        &state_edge_index,
        &state_edge_pos,
        &state_edge_dir,
        &state_edge_t_min,
        &state_edge_t_max,
        &state_n0,
        &state_n1,
        &state_prim0,
        &state_prim1,
        &state_exterior_angle,
        &state_src,
        &state_src_power,
        state_wi_ptr,
        state_d0_ptr,
        &material_eta_r,
        &material_sigma,
        &material_mu_r,
        &material_gain,
        &material_valid,
        state_limit,
        grid_axis,
        grid_position,
        grid_coord0_min,
        grid_coord0_max,
        grid_coord1_min,
        grid_coord1_max,
        grid_resolution0,
        grid_resolution1,
        grid_cell_area,
        wavelength,
        direct_samples,
        keller_samples,
        suffix_samples,
        seed,
        max_order,
        recursive_state_limit,
        recursive_active_ptr,
        recursive_state_edge_index_ptr,
        recursive_state_edge_pos_ptr,
        recursive_state_edge_dir_ptr,
        recursive_state_edge_t_min_ptr,
        recursive_state_edge_t_max_ptr,
        recursive_state_n0_ptr,
        recursive_state_n1_ptr,
        recursive_state_prim0_ptr,
        recursive_state_prim1_ptr,
        recursive_state_exterior_angle_ptr,
        export_tape,
        sample_state_index_ptr,
        sample_edge_weight_ptr,
        outputs.data(),
        kOutputCount);
    if (output_count < 0 || output_count > kOutputCount)
        throw std::runtime_error("RayDN diffraction accumulation returned an invalid output count");
    pybind11::tuple result(static_cast<size_t>(output_count));
    for (int64_t i = 0; i < output_count; ++i)
        result[static_cast<size_t>(i)] = outputs[static_cast<size_t>(i)];
    return result;
}
