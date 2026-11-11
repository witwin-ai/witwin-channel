// Copyright Xingyu Chen.
// Drives the fused BDPT mixed reflection+transmission wavefront over RayD queries.

#include <torch/extension.h>

#include "rayd/resource.h"

#include "bdpt_dict_helpers.h"
#include "kernels/bdpt_wavefront.cuh"
#include "kernels/constants.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
#include <string>
#include <utility>

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
    at::Tensor& alive,
    const channel::bdpt_wavefront::Tangent* tangent,
    const channel::bdpt_wavefront::Tape* tape);
void channel_bdpt_wavefront_reverse_cuda(
    int64_t count,
    int64_t bounces,
    const std::vector<at::Tensor>& materials,
    const channel::bdpt_wavefront::ScatterTables& tables,
    double frequency_hz,
    const channel::bdpt_wavefront::Tape& tape,
    const channel::bdpt_wavefront::Gradients& grads,
    const at::Tensor& reference);
void channel_bdpt_wavefront_connect_backward_cuda(
    int64_t count,
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
    const channel::bdpt_wavefront::Tape& tape,
    const channel::bdpt_wavefront::Gradients& grads);
at::Tensor channel_bdpt_wavefront_cast_cuda(const at::Tensor& values);
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
    int64_t max_depth,
    const channel::bdpt_wavefront::Tangent* tangent,
    at::Tensor* t_transmission,
    at::Tensor* t_scattering);
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

namespace {

namespace wf = channel::bdpt_wavefront;

// Validated inputs of one wavefront solve shared by the primal, tangent and
// reverse passes.
struct WavefrontSetup {
    RayDSceneResource* scene;
    at::Tensor tx_positions;
    at::Tensor tx_power;
    at::Tensor tx_polarization;
    at::Tensor rx_positions;
    at::Tensor rx_polarization;
    std::vector<at::Tensor> material_tensors;
    std::vector<at::Tensor> scatter_tensors;
    wf::ScatterTables tables;
    int64_t tx_count;
    int64_t rx_count;
    int64_t face_count;
    int64_t material_count;
    int64_t layer_total;
    double frequency_hz;
    double amplitude_sq;
    int64_t samples;
    int64_t max_depth;
    int64_t seed;
    int64_t mode_id;
    double beta;
    int64_t chunk_samples;
    int64_t pair_tile;
};

WavefrontSetup make_setup(
    RayDSceneResource& scene, const torch::Tensor& tx_positions, const torch::Tensor& tx_power,
    const torch::Tensor& tx_polarization, const torch::Tensor& rx_positions,
    const torch::Tensor& rx_polarization, const pybind11::dict& materials, const pybind11::object& scatter,
    double frequency_hz, int64_t samples, int64_t max_depth, int64_t seed, int64_t mode_id, double beta,
    bool emit_mixed, int64_t max_scattering_order, double scene_diagonal, int64_t chunk_samples,
    int64_t pair_tile) {
    TORCH_CHECK(scene.available(), "BDPT wavefront requires an available RayD scene resource");
    check_wavefront_tensor(tx_positions, "tx_positions", at::kFloat, 2, tx_positions);
    TORCH_CHECK(tx_positions.size(1) == 3, "tx_positions must have shape (T, 3)");
    WavefrontSetup setup{};
    setup.scene = &scene;
    setup.tx_count = tx_positions.size(0);
    setup.rx_count = rx_positions.dim() == 2 ? rx_positions.size(0) : -1;
    TORCH_CHECK(setup.rx_count >= 0, "rx_positions must have shape (R, 3)");
    check_rows(tx_power, "tx_power", at::kFloat, setup.tx_count, tx_positions);
    check_vec3_rows(tx_polarization, "tx_polarization", setup.tx_count, tx_positions);
    check_vec3_rows(rx_positions, "rx_positions", setup.rx_count, tx_positions);
    check_vec3_rows(rx_polarization, "rx_polarization", setup.rx_count, tx_positions);
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
    setup.tx_positions = tx_positions;
    setup.tx_power = tx_power;
    setup.tx_polarization = tx_polarization;
    setup.rx_positions = rx_positions;
    setup.rx_polarization = rx_polarization;

    const at::Tensor face_material_id = tensor_from_dict(materials, "material_id");
    setup.face_count = face_material_id.dim() == 1 ? face_material_id.size(0) : -1;
    TORCH_CHECK(setup.face_count >= 0, "materials.material_id must have shape (F,)");
    const at::Tensor layer_offset = tensor_from_dict(materials, "layer_offset");
    setup.material_count = layer_offset.dim() == 1 ? layer_offset.size(0) : -1;
    TORCH_CHECK(setup.material_count >= 0, "materials.layer_offset must have shape (M,)");
    const at::Tensor layer_eps_r = tensor_from_dict(materials, "layer_eps_r");
    setup.layer_total = layer_eps_r.dim() == 1 ? layer_eps_r.size(0) : -1;
    TORCH_CHECK(setup.layer_total >= 0, "materials.layer_eps_r must have shape (L,)");
    const int64_t face_count = setup.face_count;
    const int64_t material_count = setup.material_count;
    const int64_t layer_total = setup.layer_total;
    setup.material_tensors = {
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

    wf::ScatterTables& tables = setup.tables;
    tables.emit_mixed = emit_mixed;
    tables.max_order = static_cast<int>(max_scattering_order);
    tables.epsilon_floor = static_cast<float>(std::max(1.0e-6, scene_diagonal * 1.0e-6));
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
        setup.scatter_tensors = {slot, scatter_model, sigma_h, axis, f_te, f_tm, conditional, density, marginal,
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
    const double wavelength = channel::kSpeedOfLight / frequency_hz;
    const double amplitude = wavelength / (4.0 * channel::kPi);
    setup.amplitude_sq = amplitude * amplitude;
    setup.frequency_hz = frequency_hz;
    setup.samples = samples;
    setup.max_depth = max_depth;
    setup.seed = seed;
    setup.mode_id = mode_id;
    setup.beta = beta;
    setup.chunk_samples = chunk_samples;
    setup.pair_tile = pair_tile;
    return setup;
}

void check_matrix(const at::Tensor& matrix, const char* name, const WavefrontSetup& setup) {
    check_wavefront_tensor(matrix, name, at::kFloat, 2, setup.tx_positions);
    TORCH_CHECK(
        matrix.size(0) == setup.tx_count && matrix.size(1) == setup.rx_count, name, " must have shape (T, R)");
}

// Optional per-entry float tensor of a dictionary (absent or None means zero).
at::Tensor optional_rows(const pybind11::dict& values, const char* field, int64_t rows, const WavefrontSetup& setup) {
    if (!values.contains(field) || values[pybind11::str(field)].is_none()) {
        return at::Tensor();
    }
    at::Tensor tensor = tensor_from_dict(values, field);
    check_rows(tensor, field, at::kFloat, rows, setup.tx_positions);
    return tensor;
}

const float* optional_ptr(const at::Tensor& tensor) {
    return tensor.defined() ? tensor.data_ptr<float>() : nullptr;
}

// Extra work of a derivative pass: the forward-mode tangent of every chunk, or
// the replay tape plus the reverse kernels.
struct WavefrontPass {
    const wf::Tangent* tangent = nullptr;
    at::Tensor* t_transmission = nullptr;
    at::Tensor* t_scattering = nullptr;
    const wf::Gradients* grads = nullptr;
};

// The chunk loop shared by every pass. Per chunk sample the wavefront holds its
// state; per bounce it costs one RayD intersection, one native step and, per
// receiver tile, one visibility-input kernel, one RayD visibility query and one
// connect kernel (accumulate, or cotangent under the reverse pass, which then
// adds one reverse kernel per chunk).
void run_wavefront(
    const WavefrontSetup& setup, const WavefrontPass& pass, at::Tensor& transmission, at::Tensor& scattering,
    at::Tensor& counters, const std::vector<at::Tensor>& row_out) {
    const wf::ScatterTables& tables = setup.tables;
    const at::Tensor& reference = setup.tx_positions;
    const auto float_options = reference.options().dtype(at::kFloat);
    const auto long_options = reference.options().dtype(at::kLong);
    const int64_t max_depth = setup.max_depth;
    const int64_t rx_count = setup.rx_count;
    auto alive = channel_bdpt_wavefront_zeros_cuda({max_depth}, long_options);
    auto event_counters = counters.narrow(0, 0, 3);
    auto row_counters = counters.narrow(0, 3, 2);

    const int64_t total = setup.tx_count * setup.samples;
    const int64_t capacity = std::min<int64_t>(total, setup.chunk_samples);
    const bool connects = (tables.emit_mixed && max_depth >= 2) || tables.enabled;
    if (!(total > 0 && rx_count > 0 && setup.face_count > 0 && connects)) {
        return;
    }
    const int64_t nee_capacity = tables.enabled ? capacity : 1;
    auto origin = at::empty({capacity, 3}, float_options);
    auto direction = at::empty({capacity, 3}, float_options);
    auto field_real = at::empty({capacity, 3}, float_options);
    auto field_imag = at::empty({capacity, 3}, float_options);
    auto pdf_forward = at::empty({capacity}, float_options);
    auto path_length = at::empty({capacity}, float_options);
    auto carrier = at::empty({nee_capacity}, float_options);
    auto component_mask = at::empty({capacity}, reference.options().dtype(at::kByte));
    auto scatter_count = at::empty({nee_capacity}, reference.options().dtype(at::kByte));
    auto valid = at::empty({capacity}, reference.options().dtype(at::kBool));
    auto nee_vec = at::empty({nee_capacity, 9}, float_options);
    auto nee_scalar = at::empty({nee_capacity, 4}, float_options);
    auto nee_material = at::empty({nee_capacity}, reference.options().dtype(at::kInt));
    const wf::State state{
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
    // Tangent state of one chunk.
    wf::Tangent tangent{};
    std::vector<at::Tensor> tangent_storage;
    if (pass.tangent != nullptr) {
        tangent = *pass.tangent;
        tangent_storage = {
            at::empty({capacity, 3}, float_options), at::empty({capacity, 3}, float_options),
            at::empty({nee_capacity, 3}, float_options)};
        tangent.field_real = tangent_storage[0].data_ptr<float>();
        tangent.field_imag = tangent_storage[1].data_ptr<float>();
        tangent.carrier = tangent_storage[2].data_ptr<float>();
        tangent.nee_p_te = tangent.carrier + nee_capacity;
        tangent.nee_p_tm = tangent.carrier + nee_capacity * 2;
    }
    // Replay tape of one chunk (every bounce).
    wf::Tape tape{};
    std::vector<at::Tensor> tape_storage;
    if (pass.grads != nullptr) {
        const int64_t slots = max_depth * capacity;
        tape_storage = {
            at::empty({slots}, reference.options().dtype(at::kByte)),
            at::empty({slots}, reference.options().dtype(at::kInt)),
            at::empty({slots, 18}, float_options),
            at::empty({tables.enabled ? slots : 1, wf::kTapeScatterLanes}, float_options),
        };
        float* lanes = tape_storage[2].data_ptr<float>();
        tape.capacity = capacity;
        tape.code = tape_storage[0].data_ptr<uint8_t>();
        tape.prim = tape_storage[1].data_ptr<int>();
        tape.direction = lanes;
        tape.normal = lanes + slots * 3;
        tape.field_real = lanes + slots * 6;
        tape.field_imag = lanes + slots * 9;
        tape.g_field_real = lanes + slots * 12;
        tape.g_field_imag = lanes + slots * 15;
        tape_storage.push_back(at::empty({slots, 4}, float_options));
        float* scalars = tape_storage[4].data_ptr<float>();
        tape.carrier = scalars;
        tape.scale = scalars + slots;
        tape.g_p = scalars + slots * 2;
        tape.scatter = tape_storage[3].data_ptr<float>();
    }
    const int64_t rx_tile = std::max<int64_t>(1, std::min<int64_t>(rx_count, setup.pair_tile / capacity));
    const int64_t pair_capacity = capacity * rx_tile;
    auto start = at::empty({pair_capacity, 3}, float_options);
    auto end = at::empty({pair_capacity, 3}, float_options);
    auto active = at::empty({pair_capacity}, reference.options().dtype(at::kBool));
    const at::Tensor no_tmax = at::empty({0}, float_options);
    for (int64_t first = 0; first < total; first += capacity) {
        const int64_t count = std::min<int64_t>(capacity, total - first);
        auto chunk_origin = origin.narrow(0, 0, count);
        auto chunk_direction = direction.narrow(0, 0, count);
        auto chunk_valid = valid.narrow(0, 0, count);
        channel_bdpt_wavefront_launch_cuda(
            count, first, setup.samples, setup.seed, setup.tx_positions, setup.tx_polarization, origin, direction,
            field_real, field_imag, pdf_forward, path_length, component_mask, valid);
        for (int64_t bounce = 0; bounce < max_depth; ++bounce) {
            auto alive_bounce = alive.narrow(0, bounce, 1);
            {
                rayd::torch::IntersectResult hit = rayd::torch::intersect_forward(
                    setup.scene->resource(),
                    rayd::torch::RayBatch{chunk_origin, chunk_direction, no_tmax, chunk_valid},
                    3);
                channel_bdpt_wavefront_step_cuda(
                    count, first, bounce, setup.seed, hit.t, hit.p, hit.n, hit.global_prim_id,
                    setup.material_tensors, tables, setup.frequency_hz, state, event_counters, alive_bounce,
                    pass.tangent != nullptr ? &tangent : nullptr, pass.grads != nullptr ? &tape : nullptr);
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
                    pairs, tile, rx_begin, first, setup.samples, bounce + 1, alive_bounce, state, tables,
                    setup.amplitude_sq, setup.tx_power, setup.rx_positions, start, end, active);
                rayd::torch::VisibilityResult visibility = rayd::torch::visibility_forward(
                    setup.scene->resource(), rayd::torch::VisibilityRequest{pair_start, pair_end, pair_active});
                if (pass.grads != nullptr) {
                    channel_bdpt_wavefront_connect_backward_cuda(
                        count, tile, rx_begin, first, setup.samples, bounce + 1, active, visibility.visible, state,
                        tables, setup.amplitude_sq, setup.tx_power, setup.rx_positions, setup.rx_polarization,
                        setup.frequency_hz, setup.mode_id, setup.beta, tape, *pass.grads);
                } else {
                    channel_bdpt_wavefront_connect_accumulate_cuda(
                        pairs, tile, rx_begin, first, setup.samples, bounce + 1, active, visibility.visible, state,
                        tables, setup.amplitude_sq, setup.tx_power, setup.rx_positions, setup.rx_polarization,
                        setup.frequency_hz, setup.mode_id, setup.beta, transmission, scattering, row_counters,
                        row_out, max_depth, pass.tangent != nullptr ? &tangent : nullptr, pass.t_transmission,
                        pass.t_scattering);
                }
            }
        }
        if (pass.grads != nullptr) {
            channel_bdpt_wavefront_reverse_cuda(
                count, max_depth, setup.material_tensors, tables, setup.frequency_hz, tape, *pass.grads, reference);
        }
    }
}

}  // namespace

// Light subpaths of every transmitter in one wavefront: mixed
// reflection+transmission chains connect to the receivers (when emit_mixed) and
// rough-face scatter vertices make next-event connections (when the scatter
// tables are given). Samples run in chunks of chunk_samples; see run_wavefront
// for the launches per chunk. Returns float [tx, rx] matrices
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
    TORCH_CHECK(rows_capacity >= -1, "rows_capacity must be -1 or non-negative");
    const WavefrontSetup setup = make_setup(
        scene, tx_positions, tx_power, tx_polarization, rx_positions, rx_polarization, materials, scatter,
        frequency_hz, samples, max_depth, seed, mode_id, beta, emit_mixed, max_scattering_order, scene_diagonal,
        chunk_samples, pair_tile);
    check_matrix(base_transmission, "base_transmission", setup);
    check_matrix(base_scattering, "base_scattering", setup);
    const auto float_options = tx_positions.options().dtype(at::kFloat);
    const auto long_options = tx_positions.options().dtype(at::kLong);
    const auto double_options = tx_positions.options().dtype(at::kDouble);
    auto transmission = channel_bdpt_wavefront_zeros_cuda({setup.tx_count, setup.rx_count}, double_options);
    auto scattering = channel_bdpt_wavefront_zeros_cuda({setup.tx_count, setup.rx_count}, double_options);
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
    run_wavefront(setup, WavefrontPass{}, transmission, scattering, counters, row_out);
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

// Forward-mode pass: the primal of channel_bdpt_wavefront_transmission plus, in
// the same wavefront, the tangent of both matrices for the leaf tangents in
// `tangents` (face_eps_r, face_sigma_e, face_gain, face_thickness,
// layer_thickness_m, layer_eps_r, layer_sigma_e, f_te, f_tm, tx_power: absent or
// None is zero; frequency: float; base_transmission, base_scattering: the base
// matrix tangents). Event selection, hits, visibility and MIS weights are fixed.
pybind11::dict channel_bdpt_wavefront_jvp(
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
    pybind11::dict tangents) {
    const WavefrontSetup setup = make_setup(
        scene, tx_positions, tx_power, tx_polarization, rx_positions, rx_polarization, materials, scatter,
        frequency_hz, samples, max_depth, seed, mode_id, beta, emit_mixed, max_scattering_order, scene_diagonal,
        chunk_samples, pair_tile);
    check_matrix(base_transmission, "base_transmission", setup);
    check_matrix(base_scattering, "base_scattering", setup);
    const at::Tensor t_base_transmission = tensor_from_dict(tangents, "base_transmission");
    const at::Tensor t_base_scattering = tensor_from_dict(tangents, "base_scattering");
    check_matrix(t_base_transmission, "tangents.base_transmission", setup);
    check_matrix(t_base_scattering, "tangents.base_scattering", setup);
    TORCH_CHECK(tangents.contains("frequency"), "missing tangent field: frequency");
    const double t_frequency = tangents["frequency"].cast<double>();
    TORCH_CHECK(std::isfinite(t_frequency), "tangents.frequency must be finite");
    const int64_t table_size = setup.tables.enabled ? setup.scatter_tensors[4].size(0) : 0;
    const at::Tensor t_face_eps = optional_rows(tangents, "face_eps_r", setup.face_count, setup);
    const at::Tensor t_face_sigma = optional_rows(tangents, "face_sigma_e", setup.face_count, setup);
    const at::Tensor t_face_gain = optional_rows(tangents, "face_gain", setup.face_count, setup);
    const at::Tensor t_face_thickness = optional_rows(tangents, "face_thickness", setup.face_count, setup);
    const at::Tensor t_layer_thickness = optional_rows(tangents, "layer_thickness_m", setup.layer_total, setup);
    const at::Tensor t_layer_eps = optional_rows(tangents, "layer_eps_r", setup.layer_total, setup);
    const at::Tensor t_layer_sigma = optional_rows(tangents, "layer_sigma_e", setup.layer_total, setup);
    const at::Tensor t_f_te = optional_rows(tangents, "f_te", table_size, setup);
    const at::Tensor t_f_tm = optional_rows(tangents, "f_tm", table_size, setup);
    TORCH_CHECK(t_f_te.defined() == t_f_tm.defined(), "f_te and f_tm tangents must be given together");
    TORCH_CHECK(!t_f_te.defined() || setup.tables.enabled, "table tangents need scatter tables");
    const at::Tensor t_power = optional_rows(tangents, "tx_power", setup.tx_count, setup);
    wf::Tangent tangent{};
    tangent.face_eps_r = optional_ptr(t_face_eps);
    tangent.face_sigma_e = optional_ptr(t_face_sigma);
    tangent.face_gain = optional_ptr(t_face_gain);
    tangent.face_thickness = optional_ptr(t_face_thickness);
    tangent.layer_thickness_m = optional_ptr(t_layer_thickness);
    tangent.layer_eps_r = optional_ptr(t_layer_eps);
    tangent.layer_sigma_e = optional_ptr(t_layer_sigma);
    tangent.f_te = optional_ptr(t_f_te);
    tangent.f_tm = optional_ptr(t_f_tm);
    tangent.tx_power = optional_ptr(t_power);
    tangent.frequency = static_cast<float>(t_frequency);

    const auto long_options = tx_positions.options().dtype(at::kLong);
    const auto double_options = tx_positions.options().dtype(at::kDouble);
    auto transmission = channel_bdpt_wavefront_zeros_cuda({setup.tx_count, setup.rx_count}, double_options);
    auto scattering = channel_bdpt_wavefront_zeros_cuda({setup.tx_count, setup.rx_count}, double_options);
    auto t_transmission = channel_bdpt_wavefront_zeros_cuda({setup.tx_count, setup.rx_count}, double_options);
    auto t_scattering = channel_bdpt_wavefront_zeros_cuda({setup.tx_count, setup.rx_count}, double_options);
    auto counters = channel_bdpt_wavefront_zeros_cuda({6}, long_options);
    WavefrontPass pass;
    pass.tangent = &tangent;
    pass.t_transmission = &t_transmission;
    pass.t_scattering = &t_scattering;
    run_wavefront(setup, pass, transmission, scattering, counters, {});
    pybind11::dict out;
    out["transmission"] = channel_bdpt_wavefront_finalize_cuda(transmission, base_transmission);
    out["scattering"] = channel_bdpt_wavefront_finalize_cuda(scattering, base_scattering);
    out["tangent_transmission"] = channel_bdpt_wavefront_finalize_cuda(t_transmission, t_base_transmission);
    out["tangent_scattering"] = channel_bdpt_wavefront_finalize_cuda(t_scattering, t_base_scattering);
    out["counts"] = counters;
    return out;
}

// Reverse pass: replays the wavefront (same counter-based random numbers, same
// RayD hits and visibility), takes the row cotangents from grad_transmission and
// grad_scattering, walks each sample's events backwards and returns the float
// gradients named in `needs` (face_eps_r, face_sigma_e, face_gain,
// face_thickness, layer_thickness_m, layer_eps_r, layer_sigma_e, f_te, f_tm,
// tx_power, frequency).
pybind11::dict channel_bdpt_wavefront_backward(
    RayDSceneResource& scene,
    torch::Tensor tx_positions,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_positions,
    torch::Tensor rx_polarization,
    pybind11::dict materials,
    pybind11::object scatter,
    torch::Tensor grad_transmission,
    torch::Tensor grad_scattering,
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
    std::vector<std::string> needs) {
    const WavefrontSetup setup = make_setup(
        scene, tx_positions, tx_power, tx_polarization, rx_positions, rx_polarization, materials, scatter,
        frequency_hz, samples, max_depth, seed, mode_id, beta, emit_mixed, max_scattering_order, scene_diagonal,
        chunk_samples, pair_tile);
    check_matrix(grad_transmission, "grad_transmission", setup);
    check_matrix(grad_scattering, "grad_scattering", setup);
    const int64_t table_size = setup.tables.enabled ? setup.scatter_tensors[4].size(0) : 0;
    const std::vector<std::pair<std::string, int64_t>> sizes = {
        {"face_eps_r", setup.face_count},
        {"face_sigma_e", setup.face_count},
        {"face_gain", setup.face_count},
        {"face_thickness", setup.face_count},
        {"layer_thickness_m", setup.layer_total},
        {"layer_eps_r", setup.layer_total},
        {"layer_sigma_e", setup.layer_total},
        {"f_te", table_size},
        {"f_tm", table_size},
        {"tx_power", setup.tx_count},
        {"frequency", 1},
    };
    const auto double_options = tx_positions.options().dtype(at::kDouble);
    std::vector<at::Tensor> buffers(sizes.size());
    for (const std::string& name : needs) {
        bool known = false;
        for (size_t index = 0; index < sizes.size(); ++index) {
            if (sizes[index].first == name) {
                buffers[index] = channel_bdpt_wavefront_zeros_cuda({sizes[index].second}, double_options);
                known = true;
            }
        }
        TORCH_CHECK(known, "unknown BDPT wavefront gradient: ", name);
    }
    auto ptr = [&](size_t index) -> double* {
        return buffers[index].defined() ? buffers[index].data_ptr<double>() : nullptr;
    };
    const wf::Gradients grads{
        ptr(0), ptr(1), ptr(2), ptr(3), ptr(4), ptr(5), ptr(6), ptr(7), ptr(8), ptr(9), ptr(10),
        grad_transmission.data_ptr<float>(), grad_scattering.data_ptr<float>()};
    const auto long_options = tx_positions.options().dtype(at::kLong);
    auto counters = channel_bdpt_wavefront_zeros_cuda({6}, long_options);
    at::Tensor unused;
    WavefrontPass pass;
    pass.grads = &grads;
    run_wavefront(setup, pass, unused, unused, counters, {});
    pybind11::dict out;
    for (size_t index = 0; index < sizes.size(); ++index) {
        if (buffers[index].defined()) {
            out[pybind11::str(sizes[index].first)] = channel_bdpt_wavefront_cast_cuda(buffers[index]);
        }
    }
    return out;
}
