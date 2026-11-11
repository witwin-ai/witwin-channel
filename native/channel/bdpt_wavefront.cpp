// Copyright Xingyu Chen.
// Drives the fused BDPT mixed reflection+transmission wavefront over RayD queries.

#include <torch/extension.h>

#include "rayd/resource.h"

#include "bdpt_dict_helpers.h"
#include "kernels/bdpt_wavefront.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>

void channel_bdpt_wavefront_launch_cuda(
    int64_t count,
    int64_t first,
    int64_t samples,
    int64_t seed,
    const at::Tensor& tx_positions,
    const at::Tensor& tx_polarization,
    at::Tensor& origin,
    at::Tensor& direction,
    at::Tensor& field_real,
    at::Tensor& field_imag,
    at::Tensor& pdf_forward,
    at::Tensor& path_length,
    at::Tensor& component_mask,
    at::Tensor& valid);
void channel_bdpt_wavefront_step_cuda(
    int64_t count,
    int64_t first,
    int64_t depth,
    int64_t seed,
    const at::Tensor& hit_t,
    const at::Tensor& hit_p,
    const at::Tensor& hit_n,
    const at::Tensor& hit_prim,
    const std::vector<at::Tensor>& materials,
    const channel::bdpt_wavefront::ScatterTables& tables,
    double frequency_hz,
    const channel::bdpt_wavefront::State& state,
    at::Tensor& counters,
    at::Tensor& alive);
void channel_bdpt_wavefront_visibility_inputs_cuda(
    int64_t pairs,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t first,
    int64_t samples,
    int64_t light_depth,
    const at::Tensor& alive,
    const channel::bdpt_wavefront::State& state,
    const channel::bdpt_wavefront::ScatterTables& tables,
    double amplitude_sq,
    const at::Tensor& tx_power,
    const at::Tensor& rx_positions,
    at::Tensor& start,
    at::Tensor& end,
    at::Tensor& active);
void channel_bdpt_wavefront_connect_accumulate_cuda(
    int64_t pairs,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t first,
    int64_t samples,
    int64_t light_depth,
    const at::Tensor& active,
    const at::Tensor& visible,
    const channel::bdpt_wavefront::State& state,
    const channel::bdpt_wavefront::ScatterTables& tables,
    double amplitude_sq,
    const at::Tensor& tx_power,
    const at::Tensor& rx_positions,
    const at::Tensor& rx_polarization,
    double frequency_hz,
    int64_t mode_id,
    double beta,
    at::Tensor& transmission,
    at::Tensor& scattering,
    at::Tensor& rows,
    const std::vector<at::Tensor>& row_out,
    int64_t max_depth);
std::vector<at::Tensor> channel_bdpt_wavefront_pack_rows_cuda(const std::vector<at::Tensor>& row_out);
at::Tensor channel_bdpt_wavefront_finalize_cuda(const at::Tensor& accumulated, const at::Tensor& base);
at::Tensor channel_bdpt_wavefront_zeros_cuda(at::IntArrayRef sizes, const at::TensorOptions& options);

namespace {

void check_wavefront_tensor(
    const at::Tensor& tensor, const char* name, c10::ScalarType dtype, int64_t rank,
    const at::Tensor& reference) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.dim() == rank, name, " has the wrong rank");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.get_device() == reference.get_device(), name, " must share the transmitter device");
}

void check_vec3_rows(const at::Tensor& tensor, const char* name, int64_t rows, const at::Tensor& reference) {
    check_wavefront_tensor(tensor, name, at::kFloat, 2, reference);
    TORCH_CHECK(tensor.size(0) == rows && tensor.size(1) == 3, name, " must have shape (", rows, ", 3)");
}

void check_rows(
    const at::Tensor& tensor, const char* name, c10::ScalarType dtype, int64_t rows,
    const at::Tensor& reference) {
    check_wavefront_tensor(tensor, name, dtype, 1, reference);
    TORCH_CHECK(tensor.size(0) == rows, name, " must have ", rows, " rows");
}

}  // namespace


namespace {

at::Tensor checked_rows(
    const pybind11::dict& values, const char* field, c10::ScalarType dtype, int64_t rows,
    const at::Tensor& reference) {
    at::Tensor tensor = tensor_from_dict(values, field);
    check_rows(tensor, field, dtype, rows, reference);
    return tensor;
}

at::Tensor checked_flat(
    const pybind11::dict& values, const char* field, c10::ScalarType dtype, const at::Tensor& reference) {
    at::Tensor tensor = tensor_from_dict(values, field);
    TORCH_CHECK(tensor.dim() == 1, field, " must be one-dimensional");
    check_rows(tensor, field, dtype, tensor.size(0), reference);
    return tensor;
}

}  // namespace

// Light subpaths of every transmitter in one wavefront: mixed
// reflection+transmission chains connect to the receivers (when emit_mixed) and
// rough-face scatter vertices make next-event connections (when the scatter
// tables are given). Samples run in chunks of chunk_samples; per bounce a chunk
// costs one RayD intersection, one native step and, per receiver tile of at most
// pair_tile pairs, one visibility-input kernel, one RayD visibility query and one
// fused connect-and-accumulate kernel. Returns float [tx, rx] matrices
// base + sampled estimate for transmission and scattering, and the int64
// counters [transmit, reflect, scatter events, mixed rows, scatter rows].
pybind11::dict channel_bdpt_wavefront_transmission(
    RayDSceneResource& scene,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    pybind11::dict materials,
    pybind11::object scatter,
    torch::Tensor base_transmission,
    torch::Tensor base_scattering,
    double frequency_hz,
    int64_t samples,
    int64_t max_depth,
    int64_t seed,
    int64_t mode_id,
    double beta,
    bool emit_mixed,
    int64_t max_scattering_order,
    double scene_diagonal,
    int64_t chunk_samples,
    int64_t pair_tile,
    int64_t rows_capacity) {
    TORCH_CHECK(scene.available(), "BDPT wavefront requires an available RayD scene resource");
    TORCH_CHECK(rows_capacity >= -1, "rows_capacity must be -1 or non-negative");
    check_wavefront_tensor(tx_positions, "tx_positions", at::kFloat, 2, tx_positions);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (T, 3)");
    const int64_t tx_count = tx_positions.size(0);
    const int64_t rx_count = rx_positions.dim() == 2 ? rx_positions.size(0) : -1;
    TORCH_CHECK(rx_count >= 0, "rx_positions must have shape (R, 3)");
    check_rows(tx_power, "tx_power", at::kFloat, tx_count, tx_positions);
    check_vec3_rows(tx_polarization, "tx_polarization", tx_count, tx_positions);
    check_vec3_rows(rx_positions, "rx_positions", rx_count, tx_positions);
    check_vec3_rows(rx_polarization, "rx_polarization", rx_count, tx_positions);
    for (const auto* base : {&base_transmission, &base_scattering}) {
        check_wavefront_tensor(*base, "base matrix", at::kFloat, 2, tx_positions);
        TORCH_CHECK(base->size(0) == tx_count && base->size(1) == rx_count, "base matrices must have shape (T, R)");
    }
    TORCH_CHECK(
        std::isfinite(frequency_hz) && frequency_hz > 0.0 && static_cast<float>(frequency_hz) > 0.0f &&
            frequency_hz <= static_cast<double>(std::numeric_limits<float>::max()),
        "frequency_hz must be a positive finite float32 value");
    TORCH_CHECK(samples > 0, "samples must be positive");
    TORCH_CHECK(max_depth >= 1, "max_depth must be at least 1");
    TORCH_CHECK(seed >= 0, "seed must be non-negative");
    TORCH_CHECK(mode_id >= 0 && mode_id <= 2, "mode_id must be 0, 1 or 2");
    TORCH_CHECK(std::isfinite(beta) && beta > 0.0, "beta must be positive and finite");
    TORCH_CHECK(max_scattering_order >= 1 && max_scattering_order <= 255, "max_scattering_order must be in [1, 255]");
    TORCH_CHECK(std::isfinite(scene_diagonal) && scene_diagonal >= 0.0, "scene_diagonal must be finite");
    TORCH_CHECK(chunk_samples > 0, "chunk_samples must be positive");
    TORCH_CHECK(pair_tile > 0, "pair_tile must be positive");
    TORCH_CHECK(
        scene.device_index() == tx_positions.get_device(),
        "BDPT wavefront tensors must live on the RayD scene device");

    const at::Tensor face_material_id = tensor_from_dict(materials, "material_id");
    const int64_t face_count = face_material_id.dim() == 1 ? face_material_id.size(0) : -1;
    TORCH_CHECK(face_count >= 0, "materials.material_id must have shape (F,)");
    const at::Tensor layer_offset = tensor_from_dict(materials, "layer_offset");
    const int64_t material_count = layer_offset.dim() == 1 ? layer_offset.size(0) : -1;
    TORCH_CHECK(material_count >= 0, "materials.layer_offset must have shape (M,)");
    const at::Tensor layer_eps_r = tensor_from_dict(materials, "layer_eps_r");
    const int64_t layer_total = layer_eps_r.dim() == 1 ? layer_eps_r.size(0) : -1;
    TORCH_CHECK(layer_total >= 0, "materials.layer_eps_r must have shape (L,)");
    const std::vector<at::Tensor> material_tensors = {
        checked_rows(materials, "material_id", at::kInt, face_count, tx_positions),
        checked_rows(materials, "valid", at::kBool, face_count, tx_positions),
        checked_rows(materials, "gain", at::kFloat, face_count, tx_positions),
        checked_rows(materials, "eps_r", at::kFloat, face_count, tx_positions),
        checked_rows(materials, "sigma_e", at::kFloat, face_count, tx_positions),
        checked_rows(materials, "mu_r", at::kFloat, face_count, tx_positions),
        checked_rows(materials, "thickness", at::kFloat, face_count, tx_positions),
        checked_rows(materials, "layer_offset", at::kInt, material_count, tx_positions),
        checked_rows(materials, "layer_count", at::kInt, material_count, tx_positions),
        checked_rows(materials, "layer_thickness_m", at::kFloat, layer_total, tx_positions),
        checked_rows(materials, "layer_eps_r", at::kFloat, layer_total, tx_positions),
        checked_rows(materials, "layer_sigma_e", at::kFloat, layer_total, tx_positions),
        checked_rows(materials, "layer_mu_r", at::kFloat, layer_total, tx_positions),
    };

    channel::bdpt_wavefront::ScatterTables tables{};
    tables.emit_mixed = emit_mixed;
    tables.max_order = static_cast<int>(max_scattering_order);
    tables.epsilon_floor = static_cast<float>(std::max(1.0e-6, scene_diagonal * 1.0e-6));
    std::vector<at::Tensor> scatter_tensors;
    if (!scatter.is_none()) {
        const pybind11::dict values = scatter.cast<pybind11::dict>();
        const at::Tensor slot = checked_rows(values, "material_slot", at::kInt, material_count, tx_positions);
        const at::Tensor scatter_model = checked_rows(values, "scatter_model_id", at::kInt, material_count, tx_positions);
        const at::Tensor sigma_h = checked_rows(values, "rough_sigma_h_m", at::kFloat, material_count, tx_positions);
        const at::Tensor axis = checked_rows(values, "rough_axis_rad", at::kFloat, material_count, tx_positions);
        const at::Tensor f_te = checked_flat(values, "f_te", at::kFloat, tx_positions);
        const at::Tensor f_tm = checked_rows(values, "f_tm", at::kFloat, f_te.size(0), tx_positions);
        const at::Tensor conditional = checked_rows(values, "conditional_cdf", at::kFloat, f_te.size(0), tx_positions);
        const at::Tensor density = checked_rows(values, "sample_density", at::kFloat, f_te.size(0), tx_positions);
        const at::Tensor marginal = checked_flat(values, "marginal_cdf", at::kFloat, tx_positions);
        const at::Tensor offset = checked_flat(values, "table_offset", at::kLong, tx_positions);
        const at::Tensor marginal_offset =
            checked_rows(values, "marginal_offset", at::kLong, offset.size(0), tx_positions);
        const at::Tensor dims = tensor_from_dict(values, "table_dims");
        check_wavefront_tensor(dims, "table_dims", at::kInt, 2, tx_positions);
        TORCH_CHECK(dims.size(0) == offset.size(0) && dims.size(1) == 4, "table_dims must have shape (S, 4)");
        scatter_tensors = {slot, scatter_model, sigma_h, axis, f_te, f_tm, conditional, density, marginal,
                           offset, marginal_offset, dims};
        tables.enabled = true;
        tables.material_slot = slot.data_ptr<int>();
        tables.scatter_model = scatter_model.data_ptr<int>();
        tables.rough_sigma_h = sigma_h.data_ptr<float>();
        tables.rough_axis_rad = axis.data_ptr<float>();
        tables.f_te = f_te.data_ptr<float>();
        tables.f_tm = f_tm.data_ptr<float>();
        tables.marginal = marginal.data_ptr<float>();
        tables.conditional = conditional.data_ptr<float>();
        tables.density = density.data_ptr<float>();
        tables.table_offset = offset.data_ptr<int64_t>();
        tables.marginal_offset = marginal_offset.data_ptr<int64_t>();
        tables.dims = dims.data_ptr<int>();
    }
    const double wavelength = 299792458.0 / frequency_hz;
    const double amplitude = wavelength / (4.0 * 3.14159265358979323846);
    const double amplitude_sq = amplitude * amplitude;

    const auto float_options = tx_positions.options().dtype(at::kFloat);
    const auto long_options = tx_positions.options().dtype(at::kLong);
    const auto double_options = tx_positions.options().dtype(at::kDouble);
    auto transmission = channel_bdpt_wavefront_zeros_cuda({tx_count, rx_count}, double_options);
    auto scattering = channel_bdpt_wavefront_zeros_cuda({tx_count, rx_count}, double_options);
    // [transmit, reflect, scatter events, mixed rows, scatter rows, written rows]
    auto counters = channel_bdpt_wavefront_zeros_cuda({6}, long_options);
    std::vector<at::Tensor> row_out;
    if (rows_capacity >= 0) {
        const auto int_options = tx_positions.options().dtype(at::kInt);
        row_out = {
            counters.narrow(0, 5, 1),
            at::empty({rows_capacity}, long_options),
            at::empty({rows_capacity}, int_options),
            at::empty({rows_capacity}, int_options),
            at::empty({rows_capacity}, int_options),
            at::empty({rows_capacity}, int_options),
            at::empty({rows_capacity}, float_options),
            at::empty({rows_capacity}, float_options),
            at::empty({rows_capacity}, float_options),
            at::empty({rows_capacity}, float_options),
        };
    }
    auto alive = channel_bdpt_wavefront_zeros_cuda({max_depth}, long_options);
    auto event_counters = counters.narrow(0, 0, 3);
    auto row_counters = counters.narrow(0, 3, 2);

    const int64_t total = tx_count * samples;
    const int64_t capacity = std::min<int64_t>(total, chunk_samples);
    const bool connects = (emit_mixed && max_depth >= 2) || tables.enabled;
    if (total > 0 && rx_count > 0 && face_count > 0 && connects) {
        const int64_t nee_capacity = tables.enabled ? capacity : 1;
        auto origin = at::empty({capacity, 3}, float_options);
        auto direction = at::empty({capacity, 3}, float_options);
        auto field_real = at::empty({capacity, 3}, float_options);
        auto field_imag = at::empty({capacity, 3}, float_options);
        auto pdf_forward = at::empty({capacity}, float_options);
        auto path_length = at::empty({capacity}, float_options);
        auto carrier = at::empty({nee_capacity}, float_options);
        auto component_mask = at::empty({capacity}, tx_positions.options().dtype(at::kByte));
        auto scatter_count = at::empty({nee_capacity}, tx_positions.options().dtype(at::kByte));
        auto valid = at::empty({capacity}, tx_positions.options().dtype(at::kBool));
        auto nee_vec = at::empty({nee_capacity, 9}, float_options);
        auto nee_scalar = at::empty({nee_capacity, 4}, float_options);
        auto nee_material = at::empty({nee_capacity}, tx_positions.options().dtype(at::kInt));
        const channel::bdpt_wavefront::State state{
            origin.data_ptr<float>(),
            direction.data_ptr<float>(),
            field_real.data_ptr<float>(),
            field_imag.data_ptr<float>(),
            pdf_forward.data_ptr<float>(),
            path_length.data_ptr<float>(),
            carrier.data_ptr<float>(),
            component_mask.data_ptr<uint8_t>(),
            scatter_count.data_ptr<uint8_t>(),
            valid.data_ptr<bool>(),
            nee_vec.data_ptr<float>(),
            nee_vec.data_ptr<float>() + nee_capacity * 3,
            nee_vec.data_ptr<float>() + nee_capacity * 6,
            nee_scalar.data_ptr<float>(),
            nee_scalar.data_ptr<float>() + nee_capacity,
            nee_scalar.data_ptr<float>() + nee_capacity * 2,
            nee_scalar.data_ptr<float>() + nee_capacity * 3,
            nee_material.data_ptr<int>(),
        };
        const int64_t rx_tile = std::max<int64_t>(1, std::min<int64_t>(rx_count, pair_tile / capacity));
        const int64_t pair_capacity = capacity * rx_tile;
        auto start = at::empty({pair_capacity, 3}, float_options);
        auto end = at::empty({pair_capacity, 3}, float_options);
        auto active = at::empty({pair_capacity}, tx_positions.options().dtype(at::kBool));
        const at::Tensor no_tmax = at::empty({0}, float_options);
        for (int64_t first = 0; first < total; first += capacity) {
            const int64_t count = std::min<int64_t>(capacity, total - first);
            auto chunk_origin = origin.narrow(0, 0, count);
            auto chunk_direction = direction.narrow(0, 0, count);
            auto chunk_valid = valid.narrow(0, 0, count);
            channel_bdpt_wavefront_launch_cuda(
                count, first, samples, seed, tx_positions, tx_polarization, origin, direction,
                field_real, field_imag, pdf_forward, path_length, component_mask, valid);
            for (int64_t bounce = 0; bounce < max_depth; ++bounce) {
                auto alive_bounce = alive.narrow(0, bounce, 1);
                {
                    rayd::torch::IntersectResult hit = rayd::torch::intersect_forward(
                        scene.resource(),
                        rayd::torch::RayBatch{chunk_origin, chunk_direction, no_tmax, chunk_valid},
                        3);
                    channel_bdpt_wavefront_step_cuda(
                        count, first, bounce, seed, hit.t, hit.p, hit.n, hit.global_prim_id, material_tensors,
                        tables, frequency_hz, state, event_counters, alive_bounce);
                }
                // A mixed reflection+transmission vertex needs two events; scatter
                // vertices connect from the first bounce on.
                if (!tables.enabled && bounce == 0) {
                    continue;
                }
                for (int64_t rx_begin = 0; rx_begin < rx_count; rx_begin += rx_tile) {
                    const int64_t tile = std::min<int64_t>(rx_tile, rx_count - rx_begin);
                    const int64_t pairs = count * tile;
                    auto pair_start = start.narrow(0, 0, pairs);
                    auto pair_end = end.narrow(0, 0, pairs);
                    auto pair_active = active.narrow(0, 0, pairs);
                    channel_bdpt_wavefront_visibility_inputs_cuda(
                        pairs, tile, rx_begin, first, samples, bounce + 1, alive_bounce, state, tables,
                        amplitude_sq, tx_power, rx_positions, start, end, active);
                    rayd::torch::VisibilityResult visibility = rayd::torch::visibility_forward(
                        scene.resource(),
                        rayd::torch::VisibilityRequest{pair_start, pair_end, pair_active});
                    channel_bdpt_wavefront_connect_accumulate_cuda(
                        pairs, tile, rx_begin, first, samples, bounce + 1, active, visibility.visible, state,
                        tables, amplitude_sq, tx_power, rx_positions, rx_polarization, frequency_hz, mode_id,
                        beta, transmission, scattering, row_counters, row_out, max_depth);
                }
            }
        }
    }
    pybind11::dict out;
    out["transmission"] = channel_bdpt_wavefront_finalize_cuda(transmission, base_transmission);
    out["scattering"] = channel_bdpt_wavefront_finalize_cuda(scattering, base_scattering);
    out["counts"] = counters;
    if (rows_capacity >= 0) {
        static constexpr const char* kRowFields[] = {
            "topology", "contribution", "pdf", "mis_weight", "component_id", "valid",
            "tx_id", "rx_id", "grid_linear_id", "light_depth", "sensor_depth", "path_length_m"};
        const std::vector<at::Tensor> packed = channel_bdpt_wavefront_pack_rows_cuda(row_out);
        pybind11::dict rows;
        for (size_t index = 0; index < packed.size(); ++index) {
            rows[kRowFields[index]] = packed[index];
        }
        out["rows"] = rows;
    }
    return out;
}
