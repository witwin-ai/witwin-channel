#include <raydtorch/scene/cache.h>

#include <torch/extension.h>

namespace raydtorch {

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

void bind_scene_ops(py::module_ &m) {
    m.def("create_scene", &create_scene_op);
    m.def("destroy_scene", &destroy_scene);
    m.def("scene_version", &scene_version);
    m.def("scene_num_meshes", &scene_num_meshes);
    m.def("scene_edge_count", &scene_edge_count);
    m.def("update_mesh_vertices", &update_mesh_vertices);
    m.def("sync_scene", &sync_scene);
}

} // namespace raydtorch
