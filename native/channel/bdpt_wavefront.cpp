// Copyright Xingyu Chen.
// Drives the fused BDPT mixed reflection+transmission wavefront over RayD queries.

#include <torch/extension.h>

#include "rayd/resource.h"

#include "bdpt_dict_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    const at::Tensor& face_material_id,
    const at::Tensor& face_valid,
    const at::Tensor& face_gain,
    const at::Tensor& face_eps_r,
    const at::Tensor& face_sigma_e,
    const at::Tensor& face_mu_r,
    const at::Tensor& face_thickness,
    const at::Tensor& layer_offset,
    const at::Tensor& layer_count,
    const at::Tensor& layer_thickness_m,
    const at::Tensor& layer_eps_r,
    const at::Tensor& layer_sigma_e,
    const at::Tensor& layer_mu_r,
    double frequency_hz,
    at::Tensor& origin,
    at::Tensor& direction,
    at::Tensor& field_real,
    at::Tensor& field_imag,
    at::Tensor& pdf_forward,
    at::Tensor& path_length,
    at::Tensor& component_mask,
    at::Tensor& valid,
    at::Tensor& counters,
    at::Tensor& alive);
void channel_bdpt_wavefront_visibility_inputs_cuda(
    int64_t pairs,
    int64_t rx_tile,
    int64_t rx_begin,
    int64_t light_depth,
    const at::Tensor& alive,
    const at::Tensor& origin,
    const at::Tensor& direction,
    const at::Tensor& component_mask,
    const at::Tensor& valid,
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
    const at::Tensor& origin,
    const at::Tensor& direction,
    const at::Tensor& field_real,
    const at::Tensor& field_imag,
    const at::Tensor& pdf_forward,
    const at::Tensor& path_length,
    const at::Tensor& tx_power,
    const at::Tensor& rx_positions,
    const at::Tensor& rx_polarization,
    double frequency_hz,
    int64_t samples_per_tx,
    int64_t mode_id,
    double beta,
    at::Tensor& transmission,
    at::Tensor& valid_rows);
at::Tensor channel_bdpt_wavefront_finalize_cuda(const at::Tensor& accumulated, const at::Tensor& base);

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

// Mixed reflection+transmission light subpaths of every transmitter in one
// wavefront. Samples run in chunks of chunk_samples; per bounce a chunk costs
// one RayD intersection, one native step, and per receiver tile of at most
// pair_tile pairs one visibility-input kernel, one RayD visibility query and
// one fused connect-accumulate kernel. Returns the float [tx, rx]
// transmission matrix base + sampled estimate and the int64 counters
// [transmit events, reflect events, accumulated rows].
pybind11::dict channel_bdpt_wavefront_transmission(
    RayDSceneResource& scene,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    pybind11::dict materials,
    torch::Tensor base_transmission,
    double frequency_hz,
    int64_t samples,
    int64_t max_depth,
    int64_t seed,
    int64_t mode_id,
    double beta,
    int64_t chunk_samples,
    int64_t pair_tile) {
    TORCH_CHECK(scene.available(), "BDPT wavefront requires an available RayD scene resource");
    check_wavefront_tensor(tx_positions, "tx_positions", at::kFloat, 2, tx_positions);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (T, 3)");
    const int64_t tx_count = tx_positions.size(0);
    const int64_t rx_count = rx_positions.dim() == 2 ? rx_positions.size(0) : -1;
    TORCH_CHECK(rx_count >= 0, "rx_positions must have shape (R, 3)");
    check_rows(tx_power, "tx_power", at::kFloat, tx_count, tx_positions);
    check_vec3_rows(tx_polarization, "tx_polarization", tx_count, tx_positions);
    check_vec3_rows(rx_positions, "rx_positions", rx_count, tx_positions);
    check_vec3_rows(rx_polarization, "rx_polarization", rx_count, tx_positions);
    check_wavefront_tensor(base_transmission, "base_transmission", at::kFloat, 2, tx_positions);
    TORCH_CHECK(
        base_transmission.size(0) == tx_count && base_transmission.size(1) == rx_count,
        "base_transmission must have shape (T, R)");
    TORCH_CHECK(
        std::isfinite(frequency_hz) && frequency_hz > 0.0 &&
            static_cast<float>(frequency_hz) > 0.0f &&
            frequency_hz <= static_cast<double>(std::numeric_limits<float>::max()),
        "frequency_hz must be a positive finite float32 value");
    TORCH_CHECK(samples > 0, "samples must be positive");
    TORCH_CHECK(max_depth >= 1, "max_depth must be at least 1");
    TORCH_CHECK(seed >= 0, "seed must be non-negative");
    TORCH_CHECK(mode_id >= 0 && mode_id <= 2, "mode_id must be 0, 1 or 2");
    TORCH_CHECK(std::isfinite(beta) && beta > 0.0, "beta must be positive and finite");
    TORCH_CHECK(chunk_samples > 0, "chunk_samples must be positive");
    TORCH_CHECK(pair_tile > 0, "pair_tile must be positive");
    TORCH_CHECK(
        scene.device_index() == tx_positions.get_device(),
        "BDPT wavefront tensors must live on the RayD scene device");

    const at::Tensor face_material_id = tensor_from_dict(materials, "material_id");
    const int64_t face_count = face_material_id.dim() == 1 ? face_material_id.size(0) : -1;
    TORCH_CHECK(face_count >= 0, "materials.material_id must have shape (F,)");
    check_rows(face_material_id, "materials.material_id", at::kInt, face_count, tx_positions);
    const at::Tensor face_valid = tensor_from_dict(materials, "valid");
    check_rows(face_valid, "materials.valid", at::kBool, face_count, tx_positions);
    const at::Tensor face_gain = tensor_from_dict(materials, "gain");
    const at::Tensor face_eps_r = tensor_from_dict(materials, "eps_r");
    const at::Tensor face_sigma_e = tensor_from_dict(materials, "sigma_e");
    const at::Tensor face_mu_r = tensor_from_dict(materials, "mu_r");
    const at::Tensor face_thickness = tensor_from_dict(materials, "thickness");
    check_rows(face_gain, "materials.gain", at::kFloat, face_count, tx_positions);
    check_rows(face_eps_r, "materials.eps_r", at::kFloat, face_count, tx_positions);
    check_rows(face_sigma_e, "materials.sigma_e", at::kFloat, face_count, tx_positions);
    check_rows(face_mu_r, "materials.mu_r", at::kFloat, face_count, tx_positions);
    check_rows(face_thickness, "materials.thickness", at::kFloat, face_count, tx_positions);
    const at::Tensor layer_offset = tensor_from_dict(materials, "layer_offset");
    const at::Tensor layer_count = tensor_from_dict(materials, "layer_count");
    const int64_t material_count = layer_offset.dim() == 1 ? layer_offset.size(0) : -1;
    TORCH_CHECK(material_count >= 0, "materials.layer_offset must have shape (M,)");
    check_rows(layer_offset, "materials.layer_offset", at::kInt, material_count, tx_positions);
    check_rows(layer_count, "materials.layer_count", at::kInt, material_count, tx_positions);
    const at::Tensor layer_thickness_m = tensor_from_dict(materials, "layer_thickness_m");
    const at::Tensor layer_eps_r = tensor_from_dict(materials, "layer_eps_r");
    const at::Tensor layer_sigma_e = tensor_from_dict(materials, "layer_sigma_e");
    const at::Tensor layer_mu_r = tensor_from_dict(materials, "layer_mu_r");
    const int64_t layer_total = layer_eps_r.dim() == 1 ? layer_eps_r.size(0) : -1;
    TORCH_CHECK(layer_total >= 0, "materials.layer_eps_r must have shape (L,)");
    check_rows(layer_thickness_m, "materials.layer_thickness_m", at::kFloat, layer_total, tx_positions);
    check_rows(layer_eps_r, "materials.layer_eps_r", at::kFloat, layer_total, tx_positions);
    check_rows(layer_sigma_e, "materials.layer_sigma_e", at::kFloat, layer_total, tx_positions);
    check_rows(layer_mu_r, "materials.layer_mu_r", at::kFloat, layer_total, tx_positions);

    const auto float_options = tx_positions.options().dtype(at::kFloat);
    const auto long_options = tx_positions.options().dtype(at::kLong);
    auto accumulated = at::zeros({tx_count, rx_count}, tx_positions.options().dtype(at::kDouble));
    // [transmit events, reflect events, accumulated rows]
    auto counters = at::zeros({3}, long_options);
    auto alive = at::zeros({max_depth}, long_options);
    auto event_counters = counters.narrow(0, 0, 2);
    auto valid_rows = counters.narrow(0, 2, 1);

    const int64_t total = tx_count * samples;
    const int64_t capacity = std::min<int64_t>(total, chunk_samples);
    if (total > 0 && rx_count > 0 && face_count > 0) {
        auto origin = at::empty({capacity, 3}, float_options);
        auto direction = at::empty({capacity, 3}, float_options);
        auto field_real = at::empty({capacity, 3}, float_options);
        auto field_imag = at::empty({capacity, 3}, float_options);
        auto pdf_forward = at::empty({capacity}, float_options);
        auto path_length = at::empty({capacity}, float_options);
        auto component_mask = at::empty({capacity}, tx_positions.options().dtype(at::kByte));
        auto valid = at::empty({capacity}, tx_positions.options().dtype(at::kBool));
        // A mixed reflection+transmission vertex needs two events, so a chunk
        // connects from the second bounce on.
        const bool connects = max_depth >= 2;
        const int64_t rx_tile = std::max<int64_t>(1, std::min<int64_t>(rx_count, pair_tile / capacity));
        const int64_t pair_capacity = connects ? capacity * rx_tile : 0;
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
                        count, first, bounce, seed, hit.t, hit.p, hit.n, hit.global_prim_id,
                        face_material_id, face_valid, face_gain, face_eps_r, face_sigma_e,
                        face_mu_r, face_thickness, layer_offset, layer_count, layer_thickness_m,
                        layer_eps_r, layer_sigma_e, layer_mu_r, frequency_hz, origin, direction,
                        field_real, field_imag, pdf_forward, path_length, component_mask, valid,
                        event_counters, alive_bounce);
                }
                if (!connects || bounce == 0) {
                    continue;
                }
                for (int64_t rx_begin = 0; rx_begin < rx_count; rx_begin += rx_tile) {
                    const int64_t tile = std::min<int64_t>(rx_tile, rx_count - rx_begin);
                    const int64_t pairs = count * tile;
                    auto pair_start = start.narrow(0, 0, pairs);
                    auto pair_end = end.narrow(0, 0, pairs);
                    auto pair_active = active.narrow(0, 0, pairs);
                    channel_bdpt_wavefront_visibility_inputs_cuda(
                        pairs, tile, rx_begin, bounce + 1, alive_bounce, origin, direction,
                        component_mask, valid, rx_positions, start, end, active);
                    rayd::torch::VisibilityResult visibility = rayd::torch::visibility_forward(
                        scene.resource(),
                        rayd::torch::VisibilityRequest{pair_start, pair_end, pair_active});
                    channel_bdpt_wavefront_connect_accumulate_cuda(
                        pairs, tile, rx_begin, first, samples, bounce + 1, active,
                        visibility.visible, origin, direction, field_real, field_imag,
                        pdf_forward, path_length, tx_power, rx_positions, rx_polarization,
                        frequency_hz, samples, mode_id, beta, accumulated, valid_rows);
                }
            }
        }
    }
    pybind11::dict out;
    out["transmission"] = channel_bdpt_wavefront_finalize_cuda(accumulated, base_transmission);
    out["counts"] = counters;
    return out;
}
