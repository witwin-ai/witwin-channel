// Copyright Xingyu Chen.
// Declares resource native contracts.

#pragma once

#include <rayd/torch/integration.h>
#include <torch/extension.h>

#include <memory>
#include <optional>
#include <string_view>
#include <utility>

static_assert(
    rayd::torch::kIntegrationApiVersion == 6u,
    "Channel requires RayD Torch integration API 6.");
static_assert(
    rayd::torch::kIntegrationHeaderIdentity ==
        std::string_view{"rayd.torch.integration"},
    "Channel requires the stable RayD Torch integration identity.");

class RayDSceneResource final {
public:
    explicit RayDSceneResource(rayd::torch::SceneResource resource)
        : resource_(std::move(resource)) {}

    rayd::torch::SceneResource &resource() noexcept { return resource_; }
    const rayd::torch::SceneResource &resource() const noexcept { return resource_; }
    bool available() const noexcept { return resource_.valid(); }
    int device_index() const { return resource_.device_index(); }
    const rayd::torch::SceneEdgeRecordsResult &edge_records();

private:
    rayd::torch::SceneResource resource_;
    std::optional<rayd::torch::SceneEdgeRecordsResult> edge_records_;
};

inline std::optional<at::Tensor> optional_tensor(pybind11::handle value) {
    if (value.is_none())
        return std::nullopt;
    return pybind11::cast<at::Tensor>(value);
}

inline pybind11::object tensor_or_none(const at::Tensor &tensor) {
    if (!tensor.defined())
        return pybind11::none();
    return pybind11::cast(tensor);
}
