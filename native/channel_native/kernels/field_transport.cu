#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include "../field_transport.cuh"
#include "../tensor_checks.h"

#include <vector>

namespace {

constexpr int kBlockSize = 256;
namespace field = witwin::channel::native_ext;
namespace transport = channel_native::field_transport;

__device__ __forceinline__ field::float3a load3(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ c10::complex<float> to_complex(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

__device__ __forceinline__ field::Complex from_complex(c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

__global__ void free_space_kernel(
    int64_t count,
    const float* source,
    const float* target,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    float frequency_hz,
    c10::complex<float>* field_vector,
    c10::complex<float>* coefficient,
    c10::complex<float>* path_field,
    float* path_gain,
    float* path_length,
    float* delay,
    float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const field::float3a source_value = load3(source, index);
        const field::float3a target_value = load3(target, index);
        const field::float3a offset = field::f3_sub(target_value, source_value);
        const float distance = field::safe_length(offset);
        const field::float3a direction = field::safe_normalize(
            offset, field::make_f3(0.0f, 0.0f, 1.0f));
        const field::float3a tx_pol = load3(tx_polarization, index);
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
        const float source_amplitude = sqrtf(fmaxf(tx_power[index], 0.0f));
        const field::Complex3 value = transport::free_space_complex3(
            source_value, target_value, wave_number, tx_pol);
        const int64_t base = index * 3;
        field_vector[base] = to_complex(value.x);
        field_vector[base + 1] = to_complex(value.y);
        field_vector[base + 2] = to_complex(value.z);
        const field::Complex scalar = transport::project_receiver(
            value, direction, load3(rx_polarization, index));
        coefficient[index] = to_complex(scalar);
        const field::Complex received = field::cplx_mul_real(scalar, source_amplitude);
        path_field[index] = to_complex(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        path_length[index] = distance;
        delay[index] = distance / transport::kSpeedOfLight;
        direction_out[base] = direction.x;
        direction_out[base + 1] = direction.y;
        direction_out[base + 2] = direction.z;
    }
}

__global__ void project_complex3_kernel(
    int64_t count,
    const c10::complex<float>* field_vector,
    const float* direction,
    const float* rx_polarization,
    c10::complex<float>* coefficient,
    float* path_gain) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        const field::Complex3 value = {
            from_complex(field_vector[base]),
            from_complex(field_vector[base + 1]),
            from_complex(field_vector[base + 2]),
        };
        const field::Complex scalar = transport::project_receiver(
            value, load3(direction, index), load3(rx_polarization, index));
        coefficient[index] = to_complex(scalar);
        path_gain[index] = field::cplx_abs_sqr(scalar);
    }
}

__device__ __forceinline__ field::float3a load_sequence3(
    const float* values, int64_t index, int64_t bounce, int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__global__ void reflection_sequence_kernel(
    int64_t count,
    int64_t depth,
    const float* source,
    const float* target,
    const float* interaction_positions,
    const float* interaction_normals,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* eps_r,
    const float* sigma_e,
    const float* mu_r,
    const float* gain,
    const float* thickness,
    float frequency_hz,
    c10::complex<float>* field_vector,
    c10::complex<float>* coefficient,
    c10::complex<float>* path_field,
    float* path_gain,
    float* path_length,
    float* delay,
    float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        field::float3a previous = load3(source, index);
        field::float3a first_hit = load_sequence3(
            interaction_positions, index, 0, depth);
        field::float3a incident = field::safe_normalize(
            field::f3_sub(first_hit, previous), field::make_f3(0.0f, 0.0f, 1.0f));
        field::float3a tx_axis = field::stable_perp_basis(
            incident, load3(tx_polarization, index));
        field::Complex3 value = field::cplx_scale_real(
            tx_axis, field::cplx(1.0f, 0.0f));
        float total_length = 0.0f;
        field::float3a outgoing = incident;
        for (int64_t bounce = 0; bounce < depth; ++bounce) {
            const field::float3a hit = load_sequence3(
                interaction_positions, index, bounce, depth);
            incident = field::safe_normalize(
                field::f3_sub(hit, previous), outgoing);
            total_length += field::safe_length(field::f3_sub(hit, previous));
            const int64_t scalar = index * depth + bounce;
            value = transport::reflect_complex3(
                value,
                incident,
                load_sequence3(interaction_normals, index, bounce, depth),
                eps_r[scalar],
                sigma_e[scalar],
                mu_r[scalar],
                gain[scalar],
                thickness[scalar],
                frequency_hz,
                outgoing);
            previous = hit;
        }
        const field::float3a target_value = load3(target, index);
        const field::float3a final_offset = field::f3_sub(target_value, previous);
        const field::float3a final_direction = field::safe_normalize(
            final_offset, outgoing);
        total_length += field::safe_length(final_offset);
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
        const float amplitude = 1.0f /
                                (2.0f * wave_number *
                                 fmaxf(total_length, field::UTD_EPS));
        const field::Complex propagation = field::cplx_mul_real(
            field::cplx_exp_phase(
                transport::precise_neg_kd(wave_number, total_length)),
            amplitude);
        value = field::c3_scale(value, propagation);
        const int64_t base = index * 3;
        field_vector[base] = to_complex(value.x);
        field_vector[base + 1] = to_complex(value.y);
        field_vector[base + 2] = to_complex(value.z);
        const field::Complex scalar = transport::project_receiver(
            value, final_direction, load3(rx_polarization, index));
        coefficient[index] = to_complex(scalar);
        const field::Complex received = field::cplx_mul_real(
            scalar, sqrtf(fmaxf(tx_power[index], 0.0f)));
        path_field[index] = to_complex(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        path_length[index] = total_length;
        delay[index] = total_length / transport::kSpeedOfLight;
        direction_out[base] = final_direction.x;
        direction_out[base + 1] = final_direction.y;
        direction_out[base + 2] = final_direction.z;
    }
}

__device__ __forceinline__ field::float3a reflect_point(
    field::float3a point,
    field::float3a plane_point,
    field::float3a normal) {
    const field::float3a n = field::safe_normalize(
        normal, field::make_f3(0.0f, 0.0f, 1.0f));
    return field::f3_sub(
        point,
        field::f3_mul(n, 2.0f * field::f3_dot(field::f3_sub(point, plane_point), n)));
}

__global__ void coupled_rd_field_kernel(
    int64_t count,
    const float* source,
    const float* target,
    const float* reflection_position,
    const float* reflection_normal,
    const float* edge_position,
    const float* edge_direction,
    const float* edge_n0,
    const float* edge_n1,
    const float* exterior_angle,
    const float* tx_power,
    const float* tx_polarization,
    const float* rx_polarization,
    const float* reflection_eps_r,
    const float* reflection_sigma_e,
    const float* reflection_mu_r,
    const float* reflection_gain,
    const float* reflection_thickness,
    const float* wedge_eps_r0,
    const float* wedge_sigma_e0,
    const float* wedge_mu_r0,
    const float* wedge_gain0,
    const float* wedge_thickness0,
    const float* wedge_eps_r1,
    const float* wedge_sigma_e1,
    const float* wedge_mu_r1,
    const float* wedge_gain1,
    const float* wedge_thickness1,
    float frequency_hz,
    bool reverse,
    c10::complex<float>* field_vector,
    c10::complex<float>* coefficient,
    c10::complex<float>* path_field,
    float* path_gain,
    float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const field::float3a src = load3(source, index);
        const field::float3a dst = load3(target, index);
        const field::float3a hit = load3(reflection_position, index);
        const field::float3a normal = load3(reflection_normal, index);
        const field::float3a edge = load3(edge_position, index);
        const field::float3a edge_axis = field::safe_normalize(
            load3(edge_direction, index), field::make_f3(0.0f, 0.0f, 1.0f));
        const float wave_number =
            2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;

        const field::float3a diffraction_source =
            reverse ? src : reflect_point(src, hit, normal);
        const field::float3a diffraction_target =
            reverse ? reflect_point(dst, hit, normal) : dst;
        const field::float3a incident_direction = field::safe_normalize(
            field::f3_sub(edge, diffraction_source),
            field::make_f3(1.0f, 0.0f, 0.0f));
        const field::float3a outgoing_direction = field::safe_normalize(
            field::f3_sub(diffraction_target, edge),
            field::make_f3(1.0f, 0.0f, 0.0f));
        const field::Basis3 input_edge_basis = field::diffraction_edge_basis(
            field::f3_sub(edge, diffraction_source), edge_axis, false);
        const field::Basis3 output_edge_basis = field::diffraction_edge_basis(
            field::f3_sub(diffraction_target, edge), edge_axis, true);

        field::Complex3 incident_field;
        if (reverse) {
            incident_field = transport::free_space_complex3(
                src,
                edge,
                wave_number,
                load3(tx_polarization, index));
        } else {
            const field::float3a source_to_hit = field::safe_normalize(
                field::f3_sub(hit, src), incident_direction);
            const field::float3a tx_axis = field::stable_perp_basis(
                source_to_hit, load3(tx_polarization, index));
            field::float3a reflected_direction;
            field::Complex3 reflected = transport::reflect_complex3(
                field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f)),
                source_to_hit,
                normal,
                reflection_eps_r[index],
                reflection_sigma_e[index],
                reflection_mu_r[index],
                reflection_gain[index],
                reflection_thickness[index],
                frequency_hz,
                reflected_direction);
            const float unfolded_distance = field::safe_length(
                field::f3_sub(edge, diffraction_source));
            const float amplitude = 1.0f /
                                    (2.0f * wave_number *
                                     fmaxf(unfolded_distance, field::UTD_EPS));
            const field::Complex propagation = field::cplx_mul_real(
                field::cplx_exp_phase(transport::precise_neg_kd(
                    wave_number, unfolded_distance)),
                amplitude);
            incident_field = field::c3_scale(reflected, propagation);
        }

        field::PairInputs pair{};
        pair.edgePos = edge;
        pair.edgeDir = edge_axis;
        pair.n0 = load3(edge_n0, index);
        pair.nn = load3(edge_n1, index);
        pair.wedgeN = exterior_angle[index] / field::UTD_PI;
        pair.edgeLineMin = -1.0e5f;
        pair.edgeLineMax = 1.0e5f;
        pair.sourcePos = diffraction_source;
        pair.incidentBasis = input_edge_basis;
        pair.incidentJones = field::jones_from_vector(
            incident_field, input_edge_basis);
        pair.incidentDerivativeJones = field::jones_zero();
        pair.face0Material.present = 1.0f;
        pair.face1Material.present = 1.0f;
        pair.face0Operator = transport::slab_face_operator(
            fabsf(field::f3_dot(pair.n0, incident_direction)),
            wedge_eps_r0[index],
            wedge_sigma_e0[index],
            wedge_mu_r0[index],
            wedge_gain0[index],
            wedge_thickness0[index],
            frequency_hz,
            pair.n0,
            incident_direction,
            outgoing_direction,
            input_edge_basis,
            output_edge_basis);
        pair.face1Operator = transport::slab_face_operator(
            fabsf(field::f3_dot(pair.nn, incident_direction)),
            wedge_eps_r1[index],
            wedge_sigma_e1[index],
            wedge_mu_r1[index],
            wedge_gain1[index],
            wedge_thickness1[index],
            frequency_hz,
            pair.nn,
            incident_direction,
            outgoing_direction,
            input_edge_basis,
            output_edge_basis);
        pair.selectStationaryPoint = 0.0f;
        field::MaterialParams material{};
        material.omega = -1.0f;
        field::Complex3 value = field::compute_pair_contribution(
            pair, diffraction_target, wave_number, material).vectorField;

        field::float3a final_direction = outgoing_direction;
        if (reverse) {
            value = transport::reflect_complex3(
                value,
                field::safe_normalize(field::f3_sub(hit, edge), outgoing_direction),
                normal,
                reflection_eps_r[index],
                reflection_sigma_e[index],
                reflection_mu_r[index],
                reflection_gain[index],
                reflection_thickness[index],
                frequency_hz,
                final_direction);
        }
        const int64_t base = index * 3;
        field_vector[base] = to_complex(value.x);
        field_vector[base + 1] = to_complex(value.y);
        field_vector[base + 2] = to_complex(value.z);
        const field::Complex scalar = transport::project_receiver(
            value, final_direction, load3(rx_polarization, index));
        coefficient[index] = to_complex(scalar);
        const field::Complex received = field::cplx_mul_real(
            scalar, sqrtf(fmaxf(tx_power[index], 0.0f)));
        path_field[index] = to_complex(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        direction_out[base] = final_direction.x;
        direction_out[base + 1] = final_direction.y;
        direction_out[base + 2] = final_direction.z;
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

}  // namespace

pybind11::dict cn_field_free_space(
    at::Tensor source,
    at::Tensor target,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    double frequency_hz) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(target, "target");
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = source.size(0);
    TORCH_CHECK(target.size(0) == count, "target must match source rows");
    TORCH_CHECK(tx_power.size(0) == count, "tx_power must match source rows");
    TORCH_CHECK(tx_polarization.size(0) == count, "tx_polarization must match source rows");
    TORCH_CHECK(rx_polarization.size(0) == count, "rx_polarization must match source rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        free_space_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            field_vector.data_ptr<c10::complex<float>>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_field.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>(),
            direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["direction"] = direction;
    return out;
}

pybind11::dict cn_field_project_complex3(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization) {
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_tensor(field_vector, "field_vector", at::kComplexFloat, 2);
    TORCH_CHECK(field_vector.size(1) == 3, "field_vector must have shape (N, 3)");
    check_vec3_table(direction, "direction");
    check_vec3_table(rx_polarization, "rx_polarization");
    const int64_t count = field_vector.size(0);
    TORCH_CHECK(direction.size(0) == count, "direction must match field_vector rows");
    TORCH_CHECK(rx_polarization.size(0) == count,
                "rx_polarization must match field_vector rows");
    auto coefficient = at::empty({count}, field_vector.options());
    auto path_gain = at::empty({count}, direction.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(direction.get_device()).stream();
        project_complex3_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            field_vector.data_ptr<c10::complex<float>>(),
            direction.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["coefficient"] = coefficient;
    out["path_gain"] = path_gain;
    return out;
}

pybind11::dict cn_field_reflection_sequence(
    at::Tensor source,
    at::Tensor target,
    at::Tensor interaction_positions,
    at::Tensor interaction_normals,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor eps_r,
    at::Tensor sigma_e,
    at::Tensor mu_r,
    at::Tensor gain,
    at::Tensor thickness,
    double frequency_hz) {
    using channel_native::check_flat_tensor;
    using channel_native::check_tensor;
    using channel_native::check_vec3_table;
    check_vec3_table(source, "source");
    check_vec3_table(target, "target");
    check_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    check_tensor(eps_r, "eps_r", at::kFloat, 2);
    check_tensor(sigma_e, "sigma_e", at::kFloat, 2);
    check_tensor(mu_r, "mu_r", at::kFloat, 2);
    check_tensor(gain, "gain", at::kFloat, 2);
    check_tensor(thickness, "thickness", at::kFloat, 2);
    const int64_t count = source.size(0);
    const int64_t depth = interaction_positions.size(1);
    TORCH_CHECK(depth > 0 && interaction_positions.size(2) == 3,
                "interaction_positions must have shape (N, D, 3) with D > 0");
    TORCH_CHECK(interaction_positions.size(0) == count,
                "interaction_positions must match source rows");
    TORCH_CHECK(interaction_normals.sizes() == interaction_positions.sizes(),
                "interaction_normals must match interaction_positions");
    for (const auto& tensor : {eps_r, sigma_e, mu_r, gain, thickness})
        TORCH_CHECK(tensor.size(0) == count && tensor.size(1) == depth,
                    "reflection material tensors must have shape (N, D)");
    TORCH_CHECK(target.size(0) == count && tx_power.size(0) == count &&
                    tx_polarization.size(0) == count && rx_polarization.size(0) == count,
                "reflection endpoint tensors must match source rows");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        reflection_sequence_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            depth,
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            interaction_positions.data_ptr<float>(),
            interaction_normals.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            eps_r.data_ptr<float>(),
            sigma_e.data_ptr<float>(),
            mu_r.data_ptr<float>(),
            gain.data_ptr<float>(),
            thickness.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            field_vector.data_ptr<c10::complex<float>>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_field.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>(),
            path_length.data_ptr<float>(),
            delay.data_ptr<float>(),
            direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["path_length_m"] = path_length;
    out["delay_s"] = delay;
    out["direction"] = direction;
    return out;
}

pybind11::dict cn_field_coupled_rd(
    at::Tensor source,
    at::Tensor target,
    at::Tensor reflection_position,
    at::Tensor reflection_normal,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor reflection_eps_r,
    at::Tensor reflection_sigma_e,
    at::Tensor reflection_mu_r,
    at::Tensor reflection_gain,
    at::Tensor reflection_thickness,
    at::Tensor wedge_eps_r0,
    at::Tensor wedge_sigma_e0,
    at::Tensor wedge_mu_r0,
    at::Tensor wedge_gain0,
    at::Tensor wedge_thickness0,
    at::Tensor wedge_eps_r1,
    at::Tensor wedge_sigma_e1,
    at::Tensor wedge_mu_r1,
    at::Tensor wedge_gain1,
    at::Tensor wedge_thickness1,
    double frequency_hz,
    bool reverse) {
    using channel_native::check_flat_tensor;
    using channel_native::check_vec3_table;
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {source, "source"},
             {target, "target"},
             {reflection_position, "reflection_position"},
             {reflection_normal, "reflection_normal"},
             {edge_position, "edge_position"},
             {edge_direction, "edge_direction"},
             {edge_n0, "edge_n0"},
             {edge_n1, "edge_n1"},
             {tx_polarization, "tx_polarization"},
             {rx_polarization, "rx_polarization"}})
        check_vec3_table(named.first, named.second);
    for (const auto& named : std::vector<std::pair<at::Tensor, const char*>>{
             {exterior_angle, "exterior_angle"},
             {tx_power, "tx_power"},
             {reflection_eps_r, "reflection_eps_r"},
             {reflection_sigma_e, "reflection_sigma_e"},
             {reflection_mu_r, "reflection_mu_r"},
             {reflection_gain, "reflection_gain"},
             {reflection_thickness, "reflection_thickness"},
             {wedge_eps_r0, "wedge_eps_r0"},
             {wedge_sigma_e0, "wedge_sigma_e0"},
             {wedge_mu_r0, "wedge_mu_r0"},
             {wedge_gain0, "wedge_gain0"},
             {wedge_thickness0, "wedge_thickness0"},
             {wedge_eps_r1, "wedge_eps_r1"},
             {wedge_sigma_e1, "wedge_sigma_e1"},
             {wedge_mu_r1, "wedge_mu_r1"},
             {wedge_gain1, "wedge_gain1"},
             {wedge_thickness1, "wedge_thickness1"}})
        check_flat_tensor(named.first, named.second, at::kFloat);
    const int64_t count = source.size(0);
    for (const auto& tensor : {target,
                               reflection_position,
                               reflection_normal,
                               edge_position,
                               edge_direction,
                               edge_n0,
                               edge_n1,
                               tx_polarization,
                               rx_polarization})
        TORCH_CHECK(tensor.size(0) == count, "coupled field vector rows must match source");
    for (const auto& tensor : {exterior_angle,
                               tx_power,
                               reflection_eps_r,
                               reflection_sigma_e,
                               reflection_mu_r,
                               reflection_gain,
                               reflection_thickness,
                               wedge_eps_r0,
                               wedge_sigma_e0,
                               wedge_mu_r0,
                               wedge_gain0,
                               wedge_thickness0,
                               wedge_eps_r1,
                               wedge_sigma_e1,
                               wedge_mu_r1,
                               wedge_gain1,
                               wedge_thickness1})
        TORCH_CHECK(tensor.size(0) == count, "coupled field scalar rows must match source");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");

    auto complex_options = source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, source.options());
    auto direction = at::empty_like(source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(source.get_device()).stream();
        coupled_rd_field_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count,
            source.data_ptr<float>(),
            target.data_ptr<float>(),
            reflection_position.data_ptr<float>(),
            reflection_normal.data_ptr<float>(),
            edge_position.data_ptr<float>(),
            edge_direction.data_ptr<float>(),
            edge_n0.data_ptr<float>(),
            edge_n1.data_ptr<float>(),
            exterior_angle.data_ptr<float>(),
            tx_power.data_ptr<float>(),
            tx_polarization.data_ptr<float>(),
            rx_polarization.data_ptr<float>(),
            reflection_eps_r.data_ptr<float>(),
            reflection_sigma_e.data_ptr<float>(),
            reflection_mu_r.data_ptr<float>(),
            reflection_gain.data_ptr<float>(),
            reflection_thickness.data_ptr<float>(),
            wedge_eps_r0.data_ptr<float>(),
            wedge_sigma_e0.data_ptr<float>(),
            wedge_mu_r0.data_ptr<float>(),
            wedge_gain0.data_ptr<float>(),
            wedge_thickness0.data_ptr<float>(),
            wedge_eps_r1.data_ptr<float>(),
            wedge_sigma_e1.data_ptr<float>(),
            wedge_mu_r1.data_ptr<float>(),
            wedge_gain1.data_ptr<float>(),
            wedge_thickness1.data_ptr<float>(),
            static_cast<float>(frequency_hz),
            reverse,
            field_vector.data_ptr<c10::complex<float>>(),
            coefficient.data_ptr<c10::complex<float>>(),
            path_field.data_ptr<c10::complex<float>>(),
            path_gain.data_ptr<float>(),
            direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    pybind11::dict out;
    out["field_vector"] = field_vector;
    out["coefficient"] = coefficient;
    out["path_field"] = path_field;
    out["path_gain"] = path_gain;
    out["direction"] = direction;
    return out;
}
