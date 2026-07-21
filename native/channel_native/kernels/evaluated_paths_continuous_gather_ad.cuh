#pragma once

#include <ATen/ATen.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>

namespace channel_native::evaluated_paths_ad {

using cfloat = c10::complex<float>;

struct ContinuousOutputs {
    float *path_length_m;
    float *delay_s;
    float *field_direction;
    float *interaction_position;
    float *interaction_normal;
    float *interaction_positions;
    float *interaction_normals;
    float *path_gain;
    cfloat *path_field;
    cfloat *field_xyz;
    cfloat *coefficient;
};

template <typename View, typename T>
View make_optional_view(
    const std::optional<at::Tensor>& value,
    const char *name,
    c10::ScalarType dtype,
    at::IntArrayRef shape,
    int device,
    bool honor_lazy_conjugation) {
    if (!value.has_value()) {
        return {nullptr, 0, 0, 0, false};
    }
    const at::Tensor& tensor = *value;
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == shape, name, " has the wrong shape");
    TORCH_CHECK(tensor.get_device() == device, name, " must share valid device");
    return {
        tensor.data_ptr<T>(),
        tensor.stride(0),
        tensor.dim() > 1 ? tensor.stride(1) : 0,
        tensor.dim() > 2 ? tensor.stride(2) : 0,
        honor_lazy_conjugation ? tensor.is_conj() : true};
}

template <typename Views, typename FloatView, typename ComplexView>
Views make_continuous_views(
    const std::optional<at::Tensor>& path_length_m,
    const std::optional<at::Tensor>& delay_s,
    const std::optional<at::Tensor>& field_direction,
    const std::optional<at::Tensor>& interaction_position,
    const std::optional<at::Tensor>& interaction_normal,
    const std::optional<at::Tensor>& interaction_positions,
    const std::optional<at::Tensor>& interaction_normals,
    const std::optional<at::Tensor>& path_gain,
    const std::optional<at::Tensor>& path_field,
    const std::optional<at::Tensor>& field_xyz,
    const std::optional<at::Tensor>& coefficient,
    int64_t rows,
    int64_t sequence_width,
    int device,
    bool honor_lazy_conjugation) {
    return {
        make_optional_view<FloatView, float>(path_length_m, "path_length_m", at::kFloat, {rows}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(delay_s, "delay_s", at::kFloat, {rows}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(field_direction, "field_direction", at::kFloat, {rows, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(interaction_position, "interaction_position", at::kFloat, {rows, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(interaction_normal, "interaction_normal", at::kFloat, {rows, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(interaction_positions, "interaction_positions", at::kFloat, {rows, sequence_width, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(interaction_normals, "interaction_normals", at::kFloat, {rows, sequence_width, 3}, device, honor_lazy_conjugation),
        make_optional_view<FloatView, float>(path_gain, "path_gain", at::kFloat, {rows}, device, honor_lazy_conjugation),
        make_optional_view<ComplexView, cfloat>(path_field, "path_field", at::kComplexFloat, {rows}, device, honor_lazy_conjugation),
        make_optional_view<ComplexView, cfloat>(field_xyz, "field_xyz", at::kComplexFloat, {rows, 3}, device, honor_lazy_conjugation),
        make_optional_view<ComplexView, cfloat>(coefficient, "coefficient", at::kComplexFloat, {rows}, device, honor_lazy_conjugation)};
}

struct AllocatedContinuous {
    at::Tensor path_length_m;
    at::Tensor delay_s;
    at::Tensor field_direction;
    at::Tensor interaction_position;
    at::Tensor interaction_normal;
    at::Tensor interaction_positions;
    at::Tensor interaction_normals;
    at::Tensor path_gain;
    at::Tensor path_field;
    at::Tensor field_xyz;
    at::Tensor coefficient;

    ContinuousOutputs view() const {
        return {
            path_length_m.data_ptr<float>(), delay_s.data_ptr<float>(),
            field_direction.data_ptr<float>(), interaction_position.data_ptr<float>(),
            interaction_normal.data_ptr<float>(), interaction_positions.data_ptr<float>(),
            interaction_normals.data_ptr<float>(), path_gain.data_ptr<float>(),
            path_field.data_ptr<cfloat>(), field_xyz.data_ptr<cfloat>(),
            coefficient.data_ptr<cfloat>()};
    }

    pybind11::dict dict() const {
        pybind11::dict result;
        result["path_length_m"] = path_length_m;
        result["delay_s"] = delay_s;
        result["field_direction"] = field_direction;
        result["interaction_position"] = interaction_position;
        result["interaction_normal"] = interaction_normal;
        result["interaction_positions"] = interaction_positions;
        result["interaction_normals"] = interaction_normals;
        result["path_gain"] = path_gain;
        result["path_field"] = path_field;
        result["field_xyz"] = field_xyz;
        result["coefficient"] = coefficient;
        return result;
    }
};

inline AllocatedContinuous allocate_continuous(
    const at::Tensor& reference,
    int64_t rows,
    int64_t sequence_width) {
    auto float_options = reference.options().dtype(at::kFloat);
    auto complex_options = reference.options().dtype(at::kComplexFloat);
    return {
        at::empty({rows}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, 3}, float_options),
        at::empty({rows, sequence_width, 3}, float_options),
        at::empty({rows, sequence_width, 3}, float_options),
        at::empty({rows}, float_options),
        at::empty({rows}, complex_options),
        at::empty({rows, 3}, complex_options),
        at::empty({rows}, complex_options)};
}

}  // namespace channel_native::evaluated_paths_ad
