#include <raydn/scene/geometry_kernels.h>
#include <raydn/scene/cache.h>
#include <raydn/common/tensor_check.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace raydn {

namespace {

bool tensor_has_values(const at::Tensor &tensor) {
    return tensor.defined() && tensor.numel() != 0;
}

const at::Tensor *optional_tensor(const at::Tensor &tensor) {
    if (!tensor_has_values(tensor))
        return nullptr;
    return &tensor;
}

const at::Tensor *optional_tensor(py::object obj, at::Tensor &storage) {
    if (obj.is_none())
        return nullptr;
    storage = obj.cast<at::Tensor>();
    return optional_tensor(storage);
}

struct IntersectRequest {
    at::Tensor ray_o;
    at::Tensor ray_d;
    at::Tensor ray_tmax;
    at::Tensor active;
};

thread_local std::vector<IntersectRequest> g_intersect_requests;

struct EmptyIntersectionKey {
    int device_index;
    int float_dtype;
    int int_dtype;

    bool operator==(const EmptyIntersectionKey &other) const {
        return device_index == other.device_index &&
               float_dtype == other.float_dtype &&
               int_dtype == other.int_dtype;
    }
};

struct EmptyIntersectionKeyHash {
    size_t operator()(const EmptyIntersectionKey &key) const {
        size_t value = static_cast<size_t>(key.device_index);
        value = value * 1315423911u + static_cast<size_t>(key.float_dtype);
        value = value * 1315423911u + static_cast<size_t>(key.int_dtype);
        return value;
    }
};

struct EmptyIntersectionTensors {
    at::Tensor active;
    at::Tensor p;
    at::Tensor n;
    at::Tensor geo_n;
    at::Tensor uv;
    at::Tensor barycentric;
    at::Tensor shape_id;
    at::Tensor prim_id;
    at::Tensor local_prim_id;
    at::Tensor global_prim_id;
};

std::mutex g_empty_intersection_mutex;
std::unordered_map<EmptyIntersectionKey, EmptyIntersectionTensors, EmptyIntersectionKeyHash>
    g_empty_intersection_tensors;

EmptyIntersectionTensors empty_intersection_tensors(
    const at::Tensor &float_like,
    const at::Tensor &int_like) {
    EmptyIntersectionKey key{
        float_like.device().index(),
        static_cast<int>(float_like.scalar_type()),
        static_cast<int>(int_like.scalar_type()),
    };
    std::lock_guard<std::mutex> lock(g_empty_intersection_mutex);
    auto it = g_empty_intersection_tensors.find(key);
    if (it == g_empty_intersection_tensors.end()) {
        auto fopts = float_like.options();
        auto iopts = int_like.options();
        EmptyIntersectionTensors tensors{
            at::empty({0}, fopts.dtype(at::kBool)),
            at::empty({0, 3}, fopts),
            at::empty({0, 3}, fopts),
            at::empty({0, 3}, fopts),
            at::empty({0, 2}, fopts),
            at::empty({0, 3}, fopts),
            at::empty({0}, iopts),
            at::empty({0}, iopts),
            at::empty({0}, iopts),
            at::empty({0}, iopts),
        };
        it = g_empty_intersection_tensors.emplace(key, std::move(tensors)).first;
    }
    return it->second;
}

int64_t stash_intersect_request(
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active) {
    g_intersect_requests.push_back(IntersectRequest{
        std::move(ray_o),
        std::move(ray_d),
        std::move(ray_tmax),
        std::move(active),
    });
    return static_cast<int64_t>(g_intersect_requests.size() - 1);
}

IntersectRequest take_intersect_request(int64_t request_handle) {
    if (request_handle < 0 ||
        static_cast<size_t>(request_handle) >= g_intersect_requests.size())
        throw std::runtime_error("invalid internal intersect request handle.");
    if (static_cast<size_t>(request_handle) != g_intersect_requests.size() - 1)
        throw std::runtime_error("internal intersect request stack is out of order.");
    IntersectRequest request = std::move(g_intersect_requests.back());
    g_intersect_requests.pop_back();
    return request;
}

at::Tensor optional_active_from_py(py::object active_obj, int64_t ray_count, const char *name) {
    if (active_obj.is_none())
        return at::Tensor();
    at::Tensor active = active_obj.cast<at::Tensor>();
    require_mask(active, name);
    if (active.numel() == 0)
        return active.contiguous();
    if (active.size(0) != ray_count)
        throw std::runtime_error(std::string(name) + " must match the ray batch size.");
    return active.contiguous();
}

void require_ray_tmax(const at::Tensor &ray_tmax, int64_t ray_count) {
    require_scalar_f(ray_tmax, "ray_tmax");
    if (ray_tmax.numel() != 0 && ray_tmax.size(0) != ray_count)
        throw std::runtime_error("ray_tmax must be empty or match the ray batch size.");
}

class IntersectAdFunction : public torch::autograd::Function<IntersectAdFunction> {
  public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int64_t scene_handle,
        torch::autograd::Variable vertices,
        torch::autograd::Variable ray_o,
        torch::autograd::Variable ray_d,
        torch::autograd::Variable ray_tmax,
        torch::autograd::Variable active,
        int64_t flags) {
        SceneCache &scene = get_scene(scene_handle);
        IntersectForwardOutputs out =
            intersect_forward_ad_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
        ctx->set_materialize_grads(false);
        ctx->saved_data["scene_handle"] = scene_handle;
        ctx->saved_data["flags"] = flags;
        ctx->saved_data["need_grad_vertices"] = vertices.requires_grad();
        ctx->saved_data["need_grad_ray_o"] = ray_o.requires_grad();
        ctx->saved_data["need_grad_ray_d"] = ray_d.requires_grad();
        ctx->saved_data["need_grad_ray_tmax"] = ray_tmax.requires_grad();
        ctx->save_for_backward({ray_o, ray_d, ray_tmax, active, out.tape_prim_id, out.tape_barycentric});
        return {
            out.t,
            out.p,
            out.n,
            out.geo_n,
            out.uv,
            out.barycentric,
            out.shape_id,
            out.prim_id,
            out.local_prim_id,
            out.global_prim_id,
        };
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor &ray_o = saved[0];
        const at::Tensor &ray_d = saved[1];
        const at::Tensor &ray_tmax = saved[2];
        const at::Tensor &active = saved[3];
        const at::Tensor &tape_prim_id = saved[4];
        const at::Tensor &tape_barycentric = saved[5];
        const int64_t scene_handle = ctx->saved_data["scene_handle"].toInt();
        const int64_t flags = ctx->saved_data["flags"].toInt();
        const bool need_grad_vertices = ctx->saved_data["need_grad_vertices"].toBool();
        const bool need_grad_ray_o = ctx->saved_data["need_grad_ray_o"].toBool();
        const bool need_grad_ray_d = ctx->saved_data["need_grad_ray_d"].toBool();
        const bool need_grad_ray_tmax = ctx->saved_data["need_grad_ray_tmax"].toBool();
        const bool need_any_grad =
            need_grad_vertices || need_grad_ray_o || need_grad_ray_d || need_grad_ray_tmax;
        if (!need_any_grad) {
            return {
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
            };
        }

        const at::Tensor &grad_t = grad_outputs[0];
        const bool only_t_grad =
            !tensor_has_values(grad_outputs[1]) &&
            !tensor_has_values(grad_outputs[2]) &&
            !tensor_has_values(grad_outputs[3]) &&
            !tensor_has_values(grad_outputs[4]) &&
            !tensor_has_values(grad_outputs[5]);
        SceneCache &scene = get_scene(scene_handle);
        IntersectBackwardOutputs out;
        if ((flags == 0 || only_t_grad) && tensor_has_values(grad_t)) {
            out = intersect_backward_t_cuda(
                scene.global_vertices,
                scene.global_faces,
                ray_o,
                ray_d,
                active,
                tape_prim_id,
                tape_barycentric,
                grad_t,
                grad_t.stride(0),
                need_grad_vertices,
                need_grad_ray_o,
                need_grad_ray_d,
                need_grad_ray_tmax);
        } else if (flags == 0 || only_t_grad) {
            out.grad_vertices = need_grad_vertices ? at::zeros_like(scene.global_vertices) : at::Tensor();
            out.grad_ray_o = need_grad_ray_o ? at::zeros_like(ray_o) : at::Tensor();
            out.grad_ray_d = need_grad_ray_d ? at::zeros_like(ray_d) : at::Tensor();
            out.grad_ray_tmax = need_grad_ray_tmax ? at::zeros_like(ray_tmax) : at::Tensor();
        } else {
            const at::Tensor *gt = optional_tensor(grad_t);
            const at::Tensor *gp = optional_tensor(grad_outputs[1]);
            const at::Tensor *gn = optional_tensor(grad_outputs[2]);
            const at::Tensor *ggn = optional_tensor(grad_outputs[3]);
            const at::Tensor *guv = optional_tensor(grad_outputs[4]);
            const at::Tensor *gbary = optional_tensor(grad_outputs[5]);
            out = intersect_backward_optional_cuda(
                scene.global_vertices,
                scene.global_faces,
                ray_o,
                ray_d,
                ray_tmax,
                active,
                tape_prim_id,
                tape_barycentric,
                gt,
                gp,
                gn,
                ggn,
                guv,
                gbary,
                need_grad_vertices,
                need_grad_ray_o,
                need_grad_ray_d,
                need_grad_ray_tmax);
        }

        return {
            at::Tensor(),
            need_grad_vertices ? out.grad_vertices : at::Tensor(),
            need_grad_ray_o ? out.grad_ray_o : at::Tensor(),
            need_grad_ray_d ? out.grad_ray_d : at::Tensor(),
            need_grad_ray_tmax ? out.grad_ray_tmax : at::Tensor(),
            at::Tensor(),
            at::Tensor(),
        };
    }
};

class IntersectTAdFunction : public torch::autograd::Function<IntersectTAdFunction> {
  public:
    static torch::autograd::Variable forward(
        torch::autograd::AutogradContext *ctx,
        int64_t scene_handle,
        torch::autograd::Variable vertices,
        torch::autograd::Variable ray_o,
        torch::autograd::Variable ray_d,
        torch::autograd::Variable ray_tmax,
        torch::autograd::Variable active) {
        SceneCache &scene = get_scene(scene_handle);
        IntersectForwardOutputs out =
            intersect_forward_tape_cuda(scene, ray_o, ray_d, ray_tmax, active);
        ctx->set_materialize_grads(false);
        ctx->saved_data["scene_handle"] = scene_handle;
        ctx->saved_data["need_grad_vertices"] = vertices.requires_grad();
        ctx->saved_data["need_grad_ray_o"] = ray_o.requires_grad();
        ctx->saved_data["need_grad_ray_d"] = ray_d.requires_grad();
        ctx->saved_data["need_grad_ray_tmax"] = ray_tmax.requires_grad();
        ctx->save_for_backward({ray_o, ray_d, active, out.tape_prim_id, out.tape_barycentric});
        return out.t;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor &ray_o = saved[0];
        const at::Tensor &ray_d = saved[1];
        const at::Tensor &active = saved[2];
        const at::Tensor &tape_prim_id = saved[3];
        const at::Tensor &tape_barycentric = saved[4];
        const int64_t scene_handle = ctx->saved_data["scene_handle"].toInt();
        const bool need_grad_vertices = ctx->saved_data["need_grad_vertices"].toBool();
        const bool need_grad_ray_o = ctx->saved_data["need_grad_ray_o"].toBool();
        const bool need_grad_ray_d = ctx->saved_data["need_grad_ray_d"].toBool();
        const bool need_grad_ray_tmax = ctx->saved_data["need_grad_ray_tmax"].toBool();
        const bool need_any_grad =
            need_grad_vertices || need_grad_ray_o || need_grad_ray_d || need_grad_ray_tmax;
        if (!need_any_grad) {
            return {
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
            };
        }
        const at::Tensor &grad_t = grad_outputs[0];
        IntersectBackwardOutputs out;
        if (tensor_has_values(grad_t)) {
            SceneCache &scene = get_scene(scene_handle);
            out = intersect_backward_t_cuda(
                scene.global_vertices,
                scene.global_faces,
                ray_o,
                ray_d,
                active,
                tape_prim_id,
                tape_barycentric,
                grad_t,
                grad_t.stride(0),
                need_grad_vertices,
                need_grad_ray_o,
                need_grad_ray_d,
                need_grad_ray_tmax);
        } else {
            SceneCache &scene = get_scene(scene_handle);
            out.grad_vertices = need_grad_vertices ? at::zeros_like(scene.global_vertices) : at::Tensor();
            out.grad_ray_o = need_grad_ray_o ? at::zeros_like(ray_o) : at::Tensor();
            out.grad_ray_d = need_grad_ray_d ? at::zeros_like(ray_d) : at::Tensor();
            out.grad_ray_tmax = need_grad_ray_tmax ? at::zeros({ray_d.size(0)}, ray_d.options()) : at::Tensor();
        }
        return {
            at::Tensor(),
            need_grad_vertices ? out.grad_vertices : at::Tensor(),
            need_grad_ray_o ? out.grad_ray_o : at::Tensor(),
            need_grad_ray_d ? out.grad_ray_d : at::Tensor(),
            need_grad_ray_tmax ? out.grad_ray_tmax : at::Tensor(),
            at::Tensor(),
        };
    }
};

class IntersectTVerticesAdFunction
    : public torch::autograd::Function<IntersectTVerticesAdFunction> {
  public:
    static torch::autograd::Variable forward(
        torch::autograd::AutogradContext *ctx,
        int64_t scene_handle,
        int64_t request_handle,
        torch::autograd::Variable vertices) {
        (void)vertices;
        IntersectRequest request = take_intersect_request(request_handle);
        SceneCache &scene = get_scene(scene_handle);
        IntersectForwardOutputs out =
            intersect_forward_tape_cuda(scene, request.ray_o, request.ray_d, request.ray_tmax, request.active);
        ctx->set_materialize_grads(false);
        ctx->saved_data["scene_handle"] = scene_handle;
        ctx->save_for_backward({request.ray_o, request.ray_d, request.active, out.tape_prim_id, out.tape_barycentric});
        return out.t;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor &ray_o = saved[0];
        const at::Tensor &ray_d = saved[1];
        const at::Tensor &active = saved[2];
        const at::Tensor &tape_prim_id = saved[3];
        const at::Tensor &tape_barycentric = saved[4];
        const int64_t scene_handle = ctx->saved_data["scene_handle"].toInt();
        const at::Tensor &grad_t = grad_outputs[0];
        SceneCache &scene = get_scene(scene_handle);
        IntersectBackwardOutputs out;
        if (tensor_has_values(grad_t)) {
            out = intersect_backward_t_cuda(
                scene.global_vertices,
                scene.global_faces,
                ray_o,
                ray_d,
                active,
                tape_prim_id,
                tape_barycentric,
                grad_t,
                grad_t.stride(0),
                true,
                false,
                false,
                false);
        } else {
            out.grad_vertices = at::zeros_like(scene.global_vertices);
        }
        return {
            at::Tensor(),
            at::Tensor(),
            out.grad_vertices,
        };
    }
};

class IntersectTNoActiveVerticesAdFunction
    : public torch::autograd::Function<IntersectTNoActiveVerticesAdFunction> {
  public:
    static torch::autograd::Variable forward(
        torch::autograd::AutogradContext *ctx,
        int64_t scene_handle,
        int64_t request_handle,
        torch::autograd::Variable vertices) {
        (void)vertices;
        IntersectRequest request = take_intersect_request(request_handle);
        SceneCache &scene = get_scene(scene_handle);
        IntersectForwardOutputs out =
            intersect_forward_tape_cuda(scene, request.ray_o, request.ray_d, request.ray_tmax, at::Tensor());
        ctx->set_materialize_grads(false);
        ctx->saved_data["scene_handle"] = scene_handle;
        ctx->save_for_backward({request.ray_o, request.ray_d, out.tape_prim_id});
        return out.t;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor &ray_o = saved[0];
        const at::Tensor &ray_d = saved[1];
        const at::Tensor &tape_prim_id = saved[2];
        const int64_t scene_handle = ctx->saved_data["scene_handle"].toInt();
        const at::Tensor &grad_t = grad_outputs[0];
        SceneCache &scene = get_scene(scene_handle);
        IntersectBackwardOutputs out;
        if (tensor_has_values(grad_t)) {
            out = intersect_backward_t_cuda(
                scene.global_vertices,
                scene.global_faces,
                ray_o,
                ray_d,
                at::Tensor(),
                tape_prim_id,
                at::Tensor(),
                grad_t,
                grad_t.stride(0),
                true,
                false,
                false,
                false);
        } else {
            out.grad_vertices = at::zeros_like(scene.global_vertices);
        }
        return {
            at::Tensor(),
            at::Tensor(),
            out.grad_vertices,
        };
    }
};

class IntersectVerticesAdFunction
    : public torch::autograd::Function<IntersectVerticesAdFunction> {
  public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int64_t scene_handle,
        int64_t request_handle,
        torch::autograd::Variable vertices,
        int64_t flags) {
        (void)vertices;
        IntersectRequest request = take_intersect_request(request_handle);
        SceneCache &scene = get_scene(scene_handle);
        IntersectForwardOutputs out =
            intersect_forward_ad_flags_cuda(scene, request.ray_o, request.ray_d, request.ray_tmax, request.active, flags);
        ctx->set_materialize_grads(false);
        ctx->saved_data["scene_handle"] = scene_handle;
        ctx->saved_data["flags"] = flags;
        ctx->save_for_backward(
            {request.ray_o, request.ray_d, request.ray_tmax, request.active, out.tape_prim_id, out.tape_barycentric});
        return {
            out.t,
            out.p,
            out.n,
            out.geo_n,
            out.uv,
            out.barycentric,
            out.shape_id,
            out.prim_id,
            out.local_prim_id,
            out.global_prim_id,
        };
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor &ray_o = saved[0];
        const at::Tensor &ray_d = saved[1];
        const at::Tensor &ray_tmax = saved[2];
        const at::Tensor &active = saved[3];
        const at::Tensor &tape_prim_id = saved[4];
        const at::Tensor &tape_barycentric = saved[5];
        const int64_t scene_handle = ctx->saved_data["scene_handle"].toInt();
        const int64_t flags = ctx->saved_data["flags"].toInt();
        const at::Tensor &grad_t = grad_outputs[0];
        const bool only_t_grad =
            !tensor_has_values(grad_outputs[1]) &&
            !tensor_has_values(grad_outputs[2]) &&
            !tensor_has_values(grad_outputs[3]) &&
            !tensor_has_values(grad_outputs[4]) &&
            !tensor_has_values(grad_outputs[5]);
        SceneCache &scene = get_scene(scene_handle);
        IntersectBackwardOutputs out;
        if ((flags == 0 || only_t_grad) && tensor_has_values(grad_t)) {
            out = intersect_backward_t_cuda(
                scene.global_vertices,
                scene.global_faces,
                ray_o,
                ray_d,
                active,
                tape_prim_id,
                tape_barycentric,
                grad_t,
                grad_t.stride(0),
                true,
                false,
                false,
                false);
        } else if (flags == 0 || only_t_grad) {
            out.grad_vertices = at::zeros_like(scene.global_vertices);
        } else {
            const at::Tensor *gt = optional_tensor(grad_t);
            const at::Tensor *gp = optional_tensor(grad_outputs[1]);
            const at::Tensor *gn = optional_tensor(grad_outputs[2]);
            const at::Tensor *ggn = optional_tensor(grad_outputs[3]);
            const at::Tensor *guv = optional_tensor(grad_outputs[4]);
            const at::Tensor *gbary = optional_tensor(grad_outputs[5]);
            out = intersect_backward_optional_cuda(
                scene.global_vertices,
                scene.global_faces,
                ray_o,
                ray_d,
                ray_tmax,
                active,
                tape_prim_id,
                tape_barycentric,
                gt,
                gp,
                gn,
                ggn,
                guv,
                gbary,
                true,
                false,
                false,
                false);
        }

        return {
            at::Tensor(),
            at::Tensor(),
            out.grad_vertices,
            at::Tensor(),
        };
    }
};

} // namespace

py::tuple intersect_forward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active_obj) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0));
    at::Tensor active = optional_active_from_py(active_obj, ray_o.size(0), "active");
    SceneCache &scene = get_scene(scene_handle);
    IntersectForwardOutputs out = intersect_forward_cuda(scene, ray_o, ray_d, ray_tmax, active);
    return py::make_tuple(
        out.t,
        out.p,
        out.n,
        out.geo_n,
        out.uv,
        out.barycentric,
        out.shape_id,
        out.prim_id,
        out.local_prim_id,
        out.global_prim_id,
        out.tape_prim_id,
        out.tape_barycentric,
        out.tape_t);
}

py::tuple intersection_public_tuple(const IntersectForwardOutputs &out) {
    return py::make_tuple(
        out.t,
        out.p,
        out.n,
        out.geo_n,
        out.uv,
        out.barycentric,
        out.shape_id,
        out.prim_id,
        out.local_prim_id,
        out.global_prim_id);
}

py::tuple intersect_forward_flags_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active_obj,
    int64_t flags) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0));
    at::Tensor active = optional_active_from_py(active_obj, ray_o.size(0), "active");
    SceneCache &scene = get_scene(scene_handle);
    IntersectForwardOutputs out =
        intersect_forward_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
    return intersection_public_tuple(out);
}

at::Tensor intersect_forward_t_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active_obj) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0));
    at::Tensor active = optional_active_from_py(active_obj, ray_o.size(0), "active");
    SceneCache &scene = get_scene(scene_handle);
    return intersect_forward_t_only_cuda(scene, ray_o, ray_d, ray_tmax, active);
}

py::tuple intersect_forward_ad_flags_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active_obj,
    int64_t flags) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0));
    at::Tensor active = optional_active_from_py(active_obj, ray_o.size(0), "active");
    SceneCache &scene = get_scene(scene_handle);
    EmptyIntersectionTensors empty_tensors = empty_intersection_tensors(ray_o, scene.global_faces);
    at::Tensor active_ctx = active.defined() ? active : empty_tensors.active;
    IntersectForwardOutputs out =
        intersect_forward_ad_flags_cuda(scene, ray_o, ray_d, ray_tmax, active, flags);
    return py::make_tuple(
        out.t,
        out.p,
        out.n,
        out.geo_n,
        out.uv,
        out.barycentric,
        out.shape_id,
        out.prim_id,
        out.local_prim_id,
        out.global_prim_id,
        out.tape_prim_id,
        out.tape_barycentric,
        out.tape_t,
        active_ctx);
}

py::tuple intersection_empty_fields_op(int64_t scene_handle, at::Tensor like) {
    SceneCache &scene = get_scene(scene_handle);
    EmptyIntersectionTensors empty_tensors = empty_intersection_tensors(like, scene.global_faces);
    return py::make_tuple(
        empty_tensors.p,
        empty_tensors.n,
        empty_tensors.geo_n,
        empty_tensors.uv,
        empty_tensors.barycentric,
        empty_tensors.shape_id,
        empty_tensors.prim_id,
        empty_tensors.local_prim_id,
        empty_tensors.global_prim_id);
}

at::Tensor intersect_ad_t_op(
    int64_t scene_handle,
    at::Tensor vertices,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active_obj) {
    require_vec3f(vertices, "vertices");
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0));
    at::Tensor active = optional_active_from_py(active_obj, ray_o.size(0), "active");
    const bool vertices_only_reverse =
        vertices.requires_grad() &&
        !ray_o.requires_grad() &&
        !ray_d.requires_grad() &&
        !ray_tmax.requires_grad();
    if (!vertices_only_reverse && !active.defined()) {
        SceneCache &scene = get_scene(scene_handle);
        active = empty_intersection_tensors(ray_o, scene.global_faces).active;
    }
    if (vertices_only_reverse) {
        at::Tensor request_ray_tmax = ray_tmax.numel() == 0 ? at::Tensor() : ray_tmax;
        const int64_t request_handle =
            stash_intersect_request(ray_o, ray_d, request_ray_tmax, active);
        if (!active.defined()) {
            return IntersectTNoActiveVerticesAdFunction::apply(
                scene_handle,
                request_handle,
                vertices);
        }
        return IntersectTVerticesAdFunction::apply(
            scene_handle,
            request_handle,
            vertices);
    }
    return IntersectTAdFunction::apply(
        scene_handle,
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        active);
}

py::tuple intersect_ad_flags_op(
    int64_t scene_handle,
    at::Tensor vertices,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    py::object active_obj,
    int64_t flags) {
    require_vec3f(vertices, "vertices");
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_ray_tmax(ray_tmax, ray_o.size(0));
    at::Tensor active = optional_active_from_py(active_obj, ray_o.size(0), "active");
    const bool vertices_only_reverse =
        vertices.requires_grad() &&
        !ray_o.requires_grad() &&
        !ray_d.requires_grad() &&
        !ray_tmax.requires_grad();
    EmptyIntersectionTensors empty_tensors;
    bool have_empty_tensors = false;
    auto ensure_empty_tensors = [&]() -> EmptyIntersectionTensors & {
        if (!have_empty_tensors) {
            SceneCache &scene = get_scene(scene_handle);
            empty_tensors = empty_intersection_tensors(ray_o, scene.global_faces);
            have_empty_tensors = true;
        }
        return empty_tensors;
    };
    if (!vertices_only_reverse && !active.defined())
        active = ensure_empty_tensors().active;
    if (flags == 0) {
        torch::autograd::Variable t;
        if (vertices_only_reverse) {
            at::Tensor request_ray_tmax = ray_tmax.numel() == 0 ? at::Tensor() : ray_tmax;
            const int64_t request_handle =
                stash_intersect_request(ray_o, ray_d, request_ray_tmax, active);
            if (!active.defined()) {
                t = IntersectTNoActiveVerticesAdFunction::apply(
                    scene_handle,
                    request_handle,
                    vertices);
            } else {
                t = IntersectTVerticesAdFunction::apply(
                    scene_handle,
                    request_handle,
                    vertices);
            }
        } else {
            t = IntersectTAdFunction::apply(
                scene_handle,
                vertices,
                ray_o,
                ray_d,
                ray_tmax,
                active);
        }
        EmptyIntersectionTensors &empty_tensors = ensure_empty_tensors();
        return py::make_tuple(
            t,
            empty_tensors.p,
            empty_tensors.n,
            empty_tensors.geo_n,
            empty_tensors.uv,
            empty_tensors.barycentric,
            empty_tensors.shape_id,
            empty_tensors.prim_id,
            empty_tensors.local_prim_id,
            empty_tensors.global_prim_id);
    }
    if (vertices_only_reverse) {
        const int64_t request_handle =
            stash_intersect_request(ray_o, ray_d, ray_tmax, active);
        auto values = IntersectVerticesAdFunction::apply(
            scene_handle,
            request_handle,
            vertices,
            flags);
        return py::make_tuple(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9]);
    }
    auto values = IntersectAdFunction::apply(
        scene_handle,
        vertices,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        flags);
    return py::make_tuple(
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9]);
}

py::tuple intersect_backward_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor grad_t,
    at::Tensor grad_p,
    at::Tensor grad_n,
    at::Tensor grad_geo_n,
    at::Tensor grad_uv,
    at::Tensor grad_barycentric) {
    SceneCache &scene = get_scene(scene_handle);
    IntersectBackwardOutputs out = intersect_backward_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t,
        grad_p,
        grad_n,
        grad_geo_n,
        grad_uv,
        grad_barycentric);
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple intersect_backward_optional_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor ray_tmax,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    py::object grad_t_obj,
    py::object grad_p_obj,
    py::object grad_n_obj,
    py::object grad_geo_n_obj,
    py::object grad_uv_obj,
    py::object grad_barycentric_obj,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax) {
    at::Tensor grad_t_storage;
    at::Tensor grad_p_storage;
    at::Tensor grad_n_storage;
    at::Tensor grad_geo_n_storage;
    at::Tensor grad_uv_storage;
    at::Tensor grad_barycentric_storage;
    const at::Tensor *grad_t = optional_tensor(grad_t_obj, grad_t_storage);
    const at::Tensor *grad_p = optional_tensor(grad_p_obj, grad_p_storage);
    const at::Tensor *grad_n = optional_tensor(grad_n_obj, grad_n_storage);
    const at::Tensor *grad_geo_n = optional_tensor(grad_geo_n_obj, grad_geo_n_storage);
    const at::Tensor *grad_uv = optional_tensor(grad_uv_obj, grad_uv_storage);
    const at::Tensor *grad_barycentric =
        optional_tensor(grad_barycentric_obj, grad_barycentric_storage);
    SceneCache &scene = get_scene(scene_handle);
    IntersectBackwardOutputs out = intersect_backward_optional_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        ray_tmax,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t,
        grad_p,
        grad_n,
        grad_geo_n,
        grad_uv,
        grad_barycentric,
        need_grad_vertices,
        need_grad_ray_o,
        need_grad_ray_d,
        need_grad_ray_tmax);
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple intersect_backward_t_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor grad_t,
    bool need_grad_vertices,
    bool need_grad_ray_o,
    bool need_grad_ray_d,
    bool need_grad_ray_tmax) {
    require_vec3f(ray_o, "ray_o");
    require_vec3f(ray_d, "ray_d");
    require_mask(active, "active");
    require_contiguous(tape_prim_id, "tape_prim_id");
    require_dtype(tape_prim_id, at::kInt, "tape_prim_id");
    require_rank(tape_prim_id, 1, "tape_prim_id");
    require_contiguous(tape_barycentric, "tape_barycentric");
    require_dtype(tape_barycentric, at::kFloat, "tape_barycentric");
    require_rank(tape_barycentric, 2, "tape_barycentric");
    require_cuda(grad_t, "grad_t");
    require_dtype(grad_t, at::kFloat, "grad_t");
    require_rank(grad_t, 1, "grad_t");
    if (grad_t.size(0) != ray_d.size(0)) {
        throw std::runtime_error("grad_t has the wrong length.");
    }
    SceneCache &scene = get_scene(scene_handle);
    IntersectBackwardOutputs out = intersect_backward_t_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        grad_t,
        grad_t.stride(0),
        need_grad_vertices,
        need_grad_ray_o,
        need_grad_ray_d,
        need_grad_ray_tmax);
    return py::make_tuple(out.grad_vertices, out.grad_ray_o, out.grad_ray_d, out.grad_ray_tmax);
}

py::tuple intersect_jvp_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    at::Tensor tangent_vertices,
    at::Tensor tangent_ray_o,
    at::Tensor tangent_ray_d) {
    SceneCache &scene = get_scene(scene_handle);
    IntersectJvpOutputs out = intersect_jvp_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d);
    return py::make_tuple(
        out.tangent_t,
        out.tangent_p,
        out.tangent_n,
        out.tangent_geo_n,
        out.tangent_uv,
        out.tangent_barycentric);
}

py::tuple intersect_jvp_optional_op(
    int64_t scene_handle,
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor active,
    at::Tensor tape_prim_id,
    at::Tensor tape_barycentric,
    py::object tangent_vertices_obj,
    py::object tangent_ray_o_obj,
    py::object tangent_ray_d_obj,
    int64_t flags) {
    at::Tensor tangent_vertices_storage;
    at::Tensor tangent_ray_o_storage;
    at::Tensor tangent_ray_d_storage;
    const at::Tensor *tangent_vertices =
        optional_tensor(tangent_vertices_obj, tangent_vertices_storage);
    const at::Tensor *tangent_ray_o = optional_tensor(tangent_ray_o_obj, tangent_ray_o_storage);
    const at::Tensor *tangent_ray_d = optional_tensor(tangent_ray_d_obj, tangent_ray_d_storage);
    SceneCache &scene = get_scene(scene_handle);
    IntersectJvpOutputs out = intersect_jvp_optional_cuda(
        scene.global_vertices,
        scene.global_faces,
        ray_o,
        ray_d,
        active,
        tape_prim_id,
        tape_barycentric,
        tangent_vertices,
        tangent_ray_o,
        tangent_ray_d,
        flags);
    return py::make_tuple(
        out.tangent_t,
        out.tangent_p,
        out.tangent_n,
        out.tangent_geo_n,
        out.tangent_uv,
        out.tangent_barycentric);
}

} // namespace raydn
