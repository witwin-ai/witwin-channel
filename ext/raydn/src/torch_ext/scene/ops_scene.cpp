#include <raydn/scene/cache.h>
#include <raydn/scene/cache_kernels.h>
#include <raydn/common/tensor_check.h>

#include <torch/extension.h>

#include <string>

namespace raydn {

namespace {

constexpr int64_t kMeshUseFaceNormals = 1;
constexpr int64_t kMeshEdgesEnabled = 2;
constexpr int64_t kMeshDynamic = 4;

void require_mesh_vertex_tangent(const at::Tensor &tensor, const MeshRecord &mesh, const char *name) {
    require_vec3f(tensor, name);
    if (tensor.size(0) != mesh.vertices.size(0)) {
        throw std::runtime_error(std::string(name) + " must match its mesh vertex count.");
    }
}

} // namespace

c10::intrusive_ptr<SceneHandle> create_scene_cache_from_flat(
    std::vector<at::Tensor> vertices,
    std::vector<at::Tensor> faces,
    std::vector<at::Tensor> uv,
    std::vector<at::Tensor> face_uv,
    std::vector<at::Tensor> to_world_left,
    std::vector<at::Tensor> to_world_right,
    std::vector<int64_t> mesh_flags) {
    const size_t mesh_count = vertices.size();
    if (faces.size() != mesh_count ||
        uv.size() != mesh_count ||
        face_uv.size() != mesh_count ||
        to_world_left.size() != mesh_count ||
        to_world_right.size() != mesh_count ||
        mesh_flags.size() != mesh_count) {
        throw std::runtime_error("Scene init lists must have the same length.");
    }
    std::vector<MeshRecord> meshes;
    meshes.reserve(mesh_count);
    for (size_t i = 0; i < mesh_count; ++i) {
        MeshRecord record;
        record.vertices = vertices[i];
        record.faces = faces[i];
        record.uv = uv[i];
        record.face_uv = face_uv[i];
        record.to_world_left = to_world_left[i];
        record.to_world_right = to_world_right[i];
        const int64_t flags = mesh_flags[i];
        record.use_face_normals = (flags & kMeshUseFaceNormals) != 0;
        record.edges_enabled = (flags & kMeshEdgesEnabled) != 0;
        record.dynamic = (flags & kMeshDynamic) != 0;
        meshes.push_back(record);
    }
    const int64_t handle = create_scene(std::move(meshes));
    return c10::make_intrusive<SceneHandle>(handle);
}

int64_t create_scene_op(py::list mesh_specs) {
    std::vector<MeshRecord> meshes;
    meshes.reserve(py::len(mesh_specs));
    for (py::handle item : mesh_specs) {
        py::dict spec = py::reinterpret_borrow<py::dict>(item);
        MeshRecord record;
        record.vertices = spec["vertices"].cast<at::Tensor>();
        record.faces = spec["faces"].cast<at::Tensor>();
        record.uv = spec["uv"].cast<at::Tensor>();
        record.face_uv = spec["face_uv"].cast<at::Tensor>();
        record.to_world_left = spec["to_world_left"].cast<at::Tensor>();
        record.to_world_right = spec["to_world_right"].cast<at::Tensor>();
        record.use_face_normals = spec["use_face_normals"].cast<bool>();
        record.edges_enabled = spec["edges_enabled"].cast<bool>();
        record.dynamic = spec["dynamic"].cast<bool>();
        meshes.push_back(record);
    }
    return create_scene(std::move(meshes));
}

py::tuple split_scene_vertex_grad_op(int64_t handle, at::Tensor grad_vertices) {
    std::vector<at::Tensor> parts =
        split_scene_vertex_grad(c10::make_intrusive<SceneHandle>(handle, false), grad_vertices);
    py::tuple result(parts.size());
    for (size_t i = 0; i < parts.size(); ++i)
        result[i] = parts[i];
    return result;
}

std::vector<at::Tensor> split_scene_vertex_grad(
    c10::intrusive_ptr<SceneHandle> scene_handle,
    at::Tensor grad_vertices) {
    SceneCache &scene = get_scene(scene_handle->handle);
    require_vec3f(grad_vertices, "grad_vertices");
    if (grad_vertices.size(0) != scene.global_vertices.size(0)) {
        throw std::runtime_error("grad_vertices must match scene global vertex count.");
    }

    std::vector<at::Tensor> result;
    result.reserve(scene.meshes.size());
    int64_t vertex_offset = 0;
    for (size_t mesh_index = 0; mesh_index < scene.meshes.size(); ++mesh_index) {
        const int64_t vertex_count = scene.meshes[mesh_index].vertices.size(0);
        result.push_back(grad_vertices.narrow(0, vertex_offset, vertex_count));
        vertex_offset += vertex_count;
    }
    return result;
}

py::object pack_scene_vertex_tangents_op(int64_t handle, py::args tangent_args) {
    c10::intrusive_ptr<SceneHandle> scene = c10::make_intrusive<SceneHandle>(handle, false);
    SceneCache &cache = get_scene(handle);
    if (static_cast<size_t>(py::len(tangent_args)) != cache.meshes.size()) {
        throw std::runtime_error("pack_scene_vertex_tangents() expects one tangent per mesh.");
    }
    std::vector<c10::optional<at::Tensor>> tangents;
    tangents.reserve(py::len(tangent_args));
    for (size_t mesh_index = 0; mesh_index < cache.meshes.size(); ++mesh_index) {
        if (tangent_args[mesh_index].is_none())
            tangents.emplace_back(c10::nullopt);
        else
            tangents.emplace_back(tangent_args[mesh_index].cast<at::Tensor>());
    }
    at::Tensor packed = pack_scene_vertex_tangents(scene, std::move(tangents));
    if (!packed.defined())
        return py::none();
    return py::cast(packed);
}

at::Tensor pack_scene_vertex_tangents(
    c10::intrusive_ptr<SceneHandle> scene_handle,
    std::vector<c10::optional<at::Tensor>> tangents) {
    SceneCache &scene = get_scene(scene_handle->handle);
    if (tangents.size() != scene.meshes.size()) {
        throw std::runtime_error("pack_scene_vertex_tangents() expects one tangent per mesh.");
    }
    bool any_tangent = false;
    for (const c10::optional<at::Tensor> &tangent : tangents) {
        if (tangent.has_value() && tangent->defined() && tangent->numel() != 0) {
            any_tangent = true;
            break;
        }
    }
    if (!any_tangent)
        return at::Tensor();

    at::Tensor global_tangent = at::empty_like(scene.global_vertices);
    int64_t vertex_offset = 0;
    for (size_t mesh_index = 0; mesh_index < scene.meshes.size(); ++mesh_index) {
        const MeshRecord &mesh = scene.meshes[mesh_index];
        const int64_t vertex_count = mesh.vertices.size(0);
        const c10::optional<at::Tensor> &tangent_obj = tangents[mesh_index];
        if (!tangent_obj.has_value() || !tangent_obj->defined() || tangent_obj->numel() == 0) {
            zero_global_vertex_tangent_range_cuda(vertex_offset, vertex_count, global_tangent);
        } else {
            at::Tensor tangent = *tangent_obj;
            require_mesh_vertex_tangent(tangent, mesh, "mesh tangent");
            pack_global_vertex_tangent_cuda(tangent, vertex_offset, vertex_count, global_tangent);
        }
        vertex_offset += vertex_count;
    }
    return global_tangent;
}

} // namespace raydn
