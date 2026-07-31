// Copyright Xingyu Chen.
// Implements fields native integration.

#include <torch/extension.h>
#include <rayd/integration.h>

#include "registry.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

#define CHANNEL_TRANSMISSION_SEQUENCE_ARGUMENTS        \
    std::move(path_valid),                         \
        std::move(source),                         \
        std::move(target),                         \
        std::move(interaction_positions),          \
        std::move(interaction_normals),            \
        std::move(interaction_material_id),        \
        std::move(interaction_valid),              \
        std::move(tx_power),                       \
        std::move(tx_polarization),                \
        std::move(rx_polarization),                \
        std::move(layer_offset),                   \
        std::move(layer_count),                    \
        std::move(layer_thickness_m),              \
        std::move(layer_eps_r),                    \
        std::move(layer_sigma_e),                  \
        std::move(layer_mu_r),                     \
        frequency_hz

#define CHANNEL_DIFFRACTION_WEDGE_COMMON_ARGUMENTS \
    std::move(valid),                          \
        std::move(source),                     \
        std::move(target),                     \
        std::move(edge_position),              \
        std::move(edge_direction),             \
        std::move(edge_t_min),                 \
        std::move(edge_t_max),                 \
        std::move(edge_n0),                    \
        std::move(edge_n1),                    \
        std::move(exterior_angle),             \
        std::move(face0_valid),                \
        std::move(face0_eps_r),                \
        std::move(face0_sigma_e),              \
        std::move(face0_mu_r),                 \
        std::move(face0_gain),                 \
        std::move(face1_valid),                \
        std::move(face1_eps_r),                \
        std::move(face1_sigma_e),              \
        std::move(face1_mu_r),                 \
        std::move(face1_gain),                 \
        std::move(tx_power),                   \
        frequency_hz,                          \
        std::move(vertex_v0),                  \
        std::move(vertex_v1),                  \
        std::move(vertex_opp0),                \
        std::move(vertex_opp1),                \
        std::move(edge_boundary)

rayd::torch::TransmissionSequenceRequest transmission_sequence_request(
    torch::Tensor path_valid,
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor interaction_material_id,
    torch::Tensor interaction_valid,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz) {
    return {
        std::move(path_valid),
        std::move(source),
        std::move(target),
        std::move(interaction_positions),
        std::move(interaction_normals),
        std::move(interaction_material_id),
        std::move(interaction_valid),
        std::move(tx_power),
        std::move(tx_polarization),
        std::move(rx_polarization),
        std::move(layer_offset),
        std::move(layer_count),
        std::move(layer_thickness_m),
        std::move(layer_eps_r),
        std::move(layer_sigma_e),
        std::move(layer_mu_r),
        frequency_hz};
}

std::optional<at::Tensor> optional_tensor(pybind11::handle value);

rayd::torch::DiffractionWedgeRequest diffraction_wedge_request(
    at::Tensor valid,
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor face0_valid,
    at::Tensor face0_eps_r,
    at::Tensor face0_sigma_e,
    at::Tensor face0_mu_r,
    at::Tensor face0_gain,
    at::Tensor face1_valid,
    at::Tensor face1_eps_r,
    at::Tensor face1_sigma_e,
    at::Tensor face1_mu_r,
    at::Tensor face1_gain,
    at::Tensor tx_power,
    double frequency_hz,
    pybind11::object vertex_v0,
    pybind11::object vertex_v1,
    pybind11::object vertex_opp0,
    pybind11::object vertex_opp1,
    pybind11::object edge_boundary,
    double isb_boundary_taper_width) {
    rayd::torch::DiffractionWedgeRequest request;
    request.valid = std::move(valid);
    request.source = std::move(source);
    request.target = std::move(target);
    request.edge_position = std::move(edge_position);
    request.edge_direction = std::move(edge_direction);
    request.edge_t_min = std::move(edge_t_min);
    request.edge_t_max = std::move(edge_t_max);
    request.edge_n0 = std::move(edge_n0);
    request.edge_n1 = std::move(edge_n1);
    request.exterior_angle = std::move(exterior_angle);
    request.face0_valid = std::move(face0_valid);
    request.face0_eps_r = std::move(face0_eps_r);
    request.face0_sigma_e = std::move(face0_sigma_e);
    request.face0_mu_r = std::move(face0_mu_r);
    request.face0_gain = std::move(face0_gain);
    request.face1_valid = std::move(face1_valid);
    request.face1_eps_r = std::move(face1_eps_r);
    request.face1_sigma_e = std::move(face1_sigma_e);
    request.face1_mu_r = std::move(face1_mu_r);
    request.face1_gain = std::move(face1_gain);
    request.tx_power = std::move(tx_power);
    request.frequency_hz = frequency_hz;
    request.vertex_v0 = optional_tensor(vertex_v0);
    request.vertex_v1 = optional_tensor(vertex_v1);
    request.vertex_opp0 = optional_tensor(vertex_opp0);
    request.vertex_opp1 = optional_tensor(vertex_opp1);
    request.edge_boundary = optional_tensor(edge_boundary);
    request.isb_boundary_taper_width = isb_boundary_taper_width;
    return request;
}

std::optional<at::Tensor> optional_tensor(pybind11::handle value) {
    if (value.is_none())
        return std::nullopt;
    return pybind11::cast<at::Tensor>(value);
}

pybind11::object optional_tensor_object(
    const std::optional<at::Tensor> &value) {
    if (!value.has_value())
        return pybind11::none();
    return pybind11::cast(*value);
}

pybind11::dict transmission_sequence_result_dict(
    const rayd::torch::TransmissionSequenceResult &result) {
    pybind11::dict out;
    out["field_vector"] = result.field_vector;
    out["coefficient"] = result.coefficient;
    out["path_field"] = result.path_field;
    out["path_gain"] = result.path_gain;
    out["path_length_m"] = result.path_length_m;
    out["delay_s"] = result.delay_s;
    out["direction"] = result.direction;
    return out;
}

pybind11::dict transmission_sequence_jvp_result_dict(
    const rayd::torch::TransmissionSequenceJvpResult &result) {
    pybind11::dict out;
    out["field_vector"] = result.field_vector;
    out["coefficient"] = result.coefficient;
    out["path_field"] = result.path_field;
    out["path_gain"] = result.path_gain;
    out["path_length_m"] = result.path_length_m;
    out["delay_s"] = result.delay_s;
    return out;
}

pybind11::dict diffraction_wedge_result_dict(
    const rayd::torch::DiffractionWedgeResult &result) {
    pybind11::dict out;
    out["field_vector"] = result.field_vector;
    out["direction"] = result.direction;
    return out;
}

pybind11::dict diffraction_wedge_jvp_result_dict(
    const rayd::torch::DiffractionWedgeJvpResult &result) {
    pybind11::dict out;
    out["tangent_field_vector"] = result.tangent_field_vector;
    out["tangent_direction"] = result.tangent_direction;
    return out;
}

}  // namespace

std::vector<at::Tensor> channel_coupled_rd_prepare_cuda(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max);
pybind11::dict channel_coupled_rd_prepare_backward(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    pybind11::object grad_edge_point,
    pybind11::object grad_reflection_point,
    bool need_grad_source,
    bool need_grad_receiver);
pybind11::dict channel_coupled_rd_prepare_jvp(
    at::Tensor source,
    at::Tensor receiver,
    at::Tensor plane_point,
    at::Tensor plane_normal,
    at::Tensor edge_pos,
    at::Tensor edge_dir,
    at::Tensor edge_t_min,
    pybind11::object tangent_source,
    pybind11::object tangent_receiver);
pybind11::dict channel_field_diffraction_wedge(
    at::Tensor valid,
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor face0_valid,
    at::Tensor face0_eps_r,
    at::Tensor face0_sigma_e,
    at::Tensor face0_mu_r,
    at::Tensor face0_gain,
    at::Tensor face1_valid,
    at::Tensor face1_eps_r,
    at::Tensor face1_sigma_e,
    at::Tensor face1_mu_r,
    at::Tensor face1_gain,
    at::Tensor tx_power,
    double frequency_hz,
    pybind11::object vertex_v0,
    pybind11::object vertex_v1,
    pybind11::object vertex_opp0,
    pybind11::object vertex_opp1,
    pybind11::object edge_boundary,
    double isb_boundary_taper_width) {
    return diffraction_wedge_result_dict(
        rayd::torch::field_diffraction_wedge(diffraction_wedge_request(
            CHANNEL_DIFFRACTION_WEDGE_COMMON_ARGUMENTS,
            isb_boundary_taper_width)));
}
pybind11::dict channel_field_diffraction_wedge_backward(
    at::Tensor valid,
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor face0_valid,
    at::Tensor face0_eps_r,
    at::Tensor face0_sigma_e,
    at::Tensor face0_mu_r,
    at::Tensor face0_gain,
    at::Tensor face1_valid,
    at::Tensor face1_eps_r,
    at::Tensor face1_sigma_e,
    at::Tensor face1_mu_r,
    at::Tensor face1_gain,
    at::Tensor tx_power,
    double frequency_hz,
    pybind11::object vertex_v0,
    pybind11::object vertex_v1,
    pybind11::object vertex_opp0,
    pybind11::object vertex_opp1,
    pybind11::object edge_boundary,
    pybind11::object grad_field_vector,
    pybind11::object grad_direction,
    bool need_grad_material,
    bool need_grad_frequency,
    bool need_grad_geometry,
    bool need_grad_vertices,
    double isb_boundary_taper_width) {
    rayd::torch::DiffractionWedgeBackwardRequest request;
    request.primal = diffraction_wedge_request(
        CHANNEL_DIFFRACTION_WEDGE_COMMON_ARGUMENTS,
        isb_boundary_taper_width);
    request.grad_field_vector = optional_tensor(grad_field_vector);
    request.grad_direction = optional_tensor(grad_direction);
    request.need_grad_material = need_grad_material;
    request.need_grad_frequency = need_grad_frequency;
    request.need_grad_geometry = need_grad_geometry;
    request.need_grad_vertices = need_grad_vertices;

    const auto result = rayd::torch::field_diffraction_wedge_backward(request);
    pybind11::dict out;
    out["grad_source"] = optional_tensor_object(result.grad_source);
    out["grad_target"] = optional_tensor_object(result.grad_target);
    out["grad_face0_eps_r"] =
        optional_tensor_object(result.grad_face0_eps_r);
    out["grad_face0_sigma_e"] =
        optional_tensor_object(result.grad_face0_sigma_e);
    out["grad_face0_gain"] = optional_tensor_object(result.grad_face0_gain);
    out["grad_face1_eps_r"] =
        optional_tensor_object(result.grad_face1_eps_r);
    out["grad_face1_sigma_e"] =
        optional_tensor_object(result.grad_face1_sigma_e);
    out["grad_face1_gain"] = optional_tensor_object(result.grad_face1_gain);
    out["grad_frequency"] = optional_tensor_object(result.grad_frequency);
    out["grad_vertex_v0"] = optional_tensor_object(result.grad_vertex_v0);
    out["grad_vertex_v1"] = optional_tensor_object(result.grad_vertex_v1);
    out["grad_vertex_opp0"] =
        optional_tensor_object(result.grad_vertex_opp0);
    out["grad_vertex_opp1"] =
        optional_tensor_object(result.grad_vertex_opp1);
    return out;
}
pybind11::dict channel_field_diffraction_wedge_jvp(
    at::Tensor valid,
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge_position,
    at::Tensor edge_direction,
    at::Tensor edge_t_min,
    at::Tensor edge_t_max,
    at::Tensor edge_n0,
    at::Tensor edge_n1,
    at::Tensor exterior_angle,
    at::Tensor face0_valid,
    at::Tensor face0_eps_r,
    at::Tensor face0_sigma_e,
    at::Tensor face0_mu_r,
    at::Tensor face0_gain,
    at::Tensor face1_valid,
    at::Tensor face1_eps_r,
    at::Tensor face1_sigma_e,
    at::Tensor face1_mu_r,
    at::Tensor face1_gain,
    at::Tensor tx_power,
    double frequency_hz,
    pybind11::object vertex_v0,
    pybind11::object vertex_v1,
    pybind11::object vertex_opp0,
    pybind11::object vertex_opp1,
    pybind11::object edge_boundary,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_face0_eps_r,
    pybind11::object tangent_face0_sigma_e,
    pybind11::object tangent_face0_gain,
    pybind11::object tangent_face1_eps_r,
    pybind11::object tangent_face1_sigma_e,
    pybind11::object tangent_face1_gain,
    double tangent_frequency,
    pybind11::object tangent_vertex_v0,
    pybind11::object tangent_vertex_v1,
    pybind11::object tangent_vertex_opp0,
    pybind11::object tangent_vertex_opp1,
    double isb_boundary_taper_width) {
    rayd::torch::DiffractionWedgeJvpRequest request;
    request.primal = diffraction_wedge_request(
        CHANNEL_DIFFRACTION_WEDGE_COMMON_ARGUMENTS,
        isb_boundary_taper_width);
    request.tangent_source = optional_tensor(tangent_source);
    request.tangent_target = optional_tensor(tangent_target);
    request.tangent_face0_eps_r = optional_tensor(tangent_face0_eps_r);
    request.tangent_face0_sigma_e = optional_tensor(tangent_face0_sigma_e);
    request.tangent_face0_gain = optional_tensor(tangent_face0_gain);
    request.tangent_face1_eps_r = optional_tensor(tangent_face1_eps_r);
    request.tangent_face1_sigma_e = optional_tensor(tangent_face1_sigma_e);
    request.tangent_face1_gain = optional_tensor(tangent_face1_gain);
    request.tangent_frequency = tangent_frequency;
    request.tangent_vertex_v0 = optional_tensor(tangent_vertex_v0);
    request.tangent_vertex_v1 = optional_tensor(tangent_vertex_v1);
    request.tangent_vertex_opp0 = optional_tensor(tangent_vertex_opp0);
    request.tangent_vertex_opp1 = optional_tensor(tangent_vertex_opp1);
    return diffraction_wedge_jvp_result_dict(
        rayd::torch::field_diffraction_wedge_jvp(request));
}
pybind11::dict channel_field_coupled_rd_backward(
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
    at::Tensor edge_line_min,
    at::Tensor edge_line_max,
    double frequency_hz,
    bool reverse,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    bool need_grad_eps_r,
    bool need_grad_sigma_e,
    bool need_grad_gain,
    bool need_grad_thickness,
    bool need_grad_frequency,
    bool need_grad_geometry);
pybind11::dict channel_field_coupled_rd_jvp(
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
    at::Tensor edge_line_min,
    at::Tensor edge_line_max,
    double frequency_hz,
    bool reverse,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_reflection_position,
    pybind11::object tangent_edge_position,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency);
pybind11::dict channel_field_coupled_dd_backward(
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge1_position,
    at::Tensor edge1_direction,
    at::Tensor edge1_n0,
    at::Tensor edge1_n1,
    at::Tensor edge1_exterior,
    at::Tensor edge2_position,
    at::Tensor edge2_direction,
    at::Tensor edge2_n0,
    at::Tensor edge2_n1,
    at::Tensor edge2_exterior,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor wedge1_eps_r0,
    at::Tensor wedge1_sigma_e0,
    at::Tensor wedge1_mu_r0,
    at::Tensor wedge1_gain0,
    at::Tensor wedge1_thickness0,
    at::Tensor wedge1_eps_r1,
    at::Tensor wedge1_sigma_e1,
    at::Tensor wedge1_mu_r1,
    at::Tensor wedge1_gain1,
    at::Tensor wedge1_thickness1,
    at::Tensor wedge2_eps_r0,
    at::Tensor wedge2_sigma_e0,
    at::Tensor wedge2_mu_r0,
    at::Tensor wedge2_gain0,
    at::Tensor wedge2_thickness0,
    at::Tensor wedge2_eps_r1,
    at::Tensor wedge2_sigma_e1,
    at::Tensor wedge2_mu_r1,
    at::Tensor wedge2_gain1,
    at::Tensor wedge2_thickness1,
    at::Tensor edge1_line_min,
    at::Tensor edge1_line_max,
    at::Tensor edge2_line_min,
    at::Tensor edge2_line_max,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    bool need_grad_eps_r,
    bool need_grad_sigma_e,
    bool need_grad_gain,
    bool need_grad_thickness,
    bool need_grad_frequency,
    bool need_grad_geometry);
pybind11::dict channel_field_coupled_dd_jvp(
    at::Tensor source,
    at::Tensor target,
    at::Tensor edge1_position,
    at::Tensor edge1_direction,
    at::Tensor edge1_n0,
    at::Tensor edge1_n1,
    at::Tensor edge1_exterior,
    at::Tensor edge2_position,
    at::Tensor edge2_direction,
    at::Tensor edge2_n0,
    at::Tensor edge2_n1,
    at::Tensor edge2_exterior,
    at::Tensor tx_power,
    at::Tensor tx_polarization,
    at::Tensor rx_polarization,
    at::Tensor wedge1_eps_r0,
    at::Tensor wedge1_sigma_e0,
    at::Tensor wedge1_mu_r0,
    at::Tensor wedge1_gain0,
    at::Tensor wedge1_thickness0,
    at::Tensor wedge1_eps_r1,
    at::Tensor wedge1_sigma_e1,
    at::Tensor wedge1_mu_r1,
    at::Tensor wedge1_gain1,
    at::Tensor wedge1_thickness1,
    at::Tensor wedge2_eps_r0,
    at::Tensor wedge2_sigma_e0,
    at::Tensor wedge2_mu_r0,
    at::Tensor wedge2_gain0,
    at::Tensor wedge2_thickness0,
    at::Tensor wedge2_eps_r1,
    at::Tensor wedge2_sigma_e1,
    at::Tensor wedge2_mu_r1,
    at::Tensor wedge2_gain1,
    at::Tensor wedge2_thickness1,
    at::Tensor edge1_line_min,
    at::Tensor edge1_line_max,
    at::Tensor edge2_line_min,
    at::Tensor edge2_line_max,
    double frequency_hz,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency);
pybind11::dict channel_field_project_complex3_backward(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_gain,
    bool need_grad_field_vector,
    bool need_grad_direction);
pybind11::dict channel_field_project_complex3_jvp(
    at::Tensor field_vector,
    at::Tensor direction,
    at::Tensor rx_polarization,
    pybind11::object tangent_field_vector,
    pybind11::object tangent_direction);
pybind11::dict channel_field_free_space(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    double frequency_hz);
pybind11::dict channel_field_project_complex3(
    torch::Tensor field_vector,
    torch::Tensor direction,
    torch::Tensor rx_polarization);
pybind11::dict channel_field_reflection_sequence(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor eps_r,
    torch::Tensor sigma_e,
    torch::Tensor mu_r,
    torch::Tensor gain,
    torch::Tensor thickness,
    double frequency_hz);
pybind11::dict channel_field_transmission_sequence(
    torch::Tensor path_valid,
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor interaction_material_id,
    torch::Tensor interaction_valid,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz) {
    return transmission_sequence_result_dict(
        rayd::torch::field_transmission_sequence(
            transmission_sequence_request(CHANNEL_TRANSMISSION_SEQUENCE_ARGUMENTS)));
}
pybind11::dict channel_field_free_space_fwd64(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    double frequency_hz);
pybind11::dict channel_field_free_space_backward(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    pybind11::object grad_direction,
    bool need_grad_frequency,
    bool need_grad_geometry);
pybind11::dict channel_field_free_space_jvp(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    double frequency_hz,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target);
pybind11::dict channel_field_reflection_sequence_backward(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor eps_r,
    torch::Tensor sigma_e,
    torch::Tensor mu_r,
    torch::Tensor gain,
    torch::Tensor thickness,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    pybind11::object grad_direction,
    bool need_grad_eps_r,
    bool need_grad_sigma_e,
    bool need_grad_gain,
    bool need_grad_thickness,
    bool need_grad_frequency,
    bool need_grad_geometry);
pybind11::dict channel_field_reflection_sequence_jvp(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor eps_r,
    torch::Tensor sigma_e,
    torch::Tensor mu_r,
    torch::Tensor gain,
    torch::Tensor thickness,
    double frequency_hz,
    pybind11::object tangent_eps_r,
    pybind11::object tangent_sigma_e,
    pybind11::object tangent_gain,
    pybind11::object tangent_thickness,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_interaction_positions,
    pybind11::object tangent_interaction_normals);
pybind11::dict channel_field_transmission_sequence_backward(
    torch::Tensor path_valid,
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor interaction_material_id,
    torch::Tensor interaction_valid,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    pybind11::object grad_path_length,
    pybind11::object grad_delay,
    bool need_grad_layer_thickness,
    bool need_grad_layer_eps_r,
    bool need_grad_layer_sigma_e,
    bool need_grad_frequency,
    bool need_grad_geometry) {
    rayd::torch::TransmissionSequenceBackwardRequest request;
    request.primal = transmission_sequence_request(
        CHANNEL_TRANSMISSION_SEQUENCE_ARGUMENTS);
    request.grad_field_vector = optional_tensor(grad_field_vector);
    request.grad_coefficient = optional_tensor(grad_coefficient);
    request.grad_path_field = optional_tensor(grad_path_field);
    request.grad_path_gain = optional_tensor(grad_path_gain);
    request.grad_path_length_m = optional_tensor(grad_path_length);
    request.grad_delay_s = optional_tensor(grad_delay);
    request.need_grad_layer_thickness_m = need_grad_layer_thickness;
    request.need_grad_layer_eps_r = need_grad_layer_eps_r;
    request.need_grad_layer_sigma_e = need_grad_layer_sigma_e;
    request.need_grad_frequency = need_grad_frequency;
    request.need_grad_geometry = need_grad_geometry;

    const auto result =
        rayd::torch::field_transmission_sequence_backward(request);
    TORCH_CHECK(
        !result.grad_interaction_positions.has_value(),
        "RayD transmission backward must not materialize interaction-position gradients");
    pybind11::dict out;
    out["grad_layer_thickness_m"] =
        optional_tensor_object(result.grad_layer_thickness_m);
    out["grad_layer_eps_r"] = optional_tensor_object(result.grad_layer_eps_r);
    out["grad_layer_sigma_e"] =
        optional_tensor_object(result.grad_layer_sigma_e);
    out["grad_frequency"] = optional_tensor_object(result.grad_frequency);
    out["grad_source"] = optional_tensor_object(result.grad_source);
    out["grad_target"] = optional_tensor_object(result.grad_target);
    out["grad_interaction_positions"] = pybind11::none();
    out["grad_interaction_normals"] =
        optional_tensor_object(result.grad_interaction_normals);
    return out;
}
pybind11::dict channel_field_transmission_sequence_jvp(
    torch::Tensor path_valid,
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor interaction_positions,
    torch::Tensor interaction_normals,
    torch::Tensor interaction_material_id,
    torch::Tensor interaction_valid,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor layer_offset,
    torch::Tensor layer_count,
    torch::Tensor layer_thickness_m,
    torch::Tensor layer_eps_r,
    torch::Tensor layer_sigma_e,
    torch::Tensor layer_mu_r,
    double frequency_hz,
    pybind11::object tangent_layer_thickness_m,
    pybind11::object tangent_layer_eps_r,
    pybind11::object tangent_layer_sigma_e,
    double tangent_frequency,
    pybind11::object tangent_source,
    pybind11::object tangent_target,
    pybind11::object tangent_interaction_positions,
    pybind11::object tangent_interaction_normals) {
    rayd::torch::TransmissionSequenceJvpRequest request;
    request.primal = transmission_sequence_request(
        CHANNEL_TRANSMISSION_SEQUENCE_ARGUMENTS);
    request.tangent_layer_thickness_m =
        optional_tensor(tangent_layer_thickness_m);
    request.tangent_layer_eps_r = optional_tensor(tangent_layer_eps_r);
    request.tangent_layer_sigma_e = optional_tensor(tangent_layer_sigma_e);
    request.tangent_frequency = tangent_frequency;
    request.tangent_source = optional_tensor(tangent_source);
    request.tangent_target = optional_tensor(tangent_target);
    request.tangent_interaction_positions =
        optional_tensor(tangent_interaction_positions);
    request.tangent_interaction_normals =
        optional_tensor(tangent_interaction_normals);
    return transmission_sequence_jvp_result_dict(
        rayd::torch::field_transmission_sequence_jvp(request));
}
pybind11::dict channel_field_coupled_rd(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor reflection_position,
    torch::Tensor reflection_normal,
    torch::Tensor edge_position,
    torch::Tensor edge_direction,
    torch::Tensor edge_n0,
    torch::Tensor edge_n1,
    torch::Tensor exterior_angle,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor reflection_eps_r,
    torch::Tensor reflection_sigma_e,
    torch::Tensor reflection_mu_r,
    torch::Tensor reflection_gain,
    torch::Tensor reflection_thickness,
    torch::Tensor wedge_eps_r0,
    torch::Tensor wedge_sigma_e0,
    torch::Tensor wedge_mu_r0,
    torch::Tensor wedge_gain0,
    torch::Tensor wedge_thickness0,
    torch::Tensor wedge_eps_r1,
    torch::Tensor wedge_sigma_e1,
    torch::Tensor wedge_mu_r1,
    torch::Tensor wedge_gain1,
    torch::Tensor wedge_thickness1,
    torch::Tensor edge_line_min,
    torch::Tensor edge_line_max,
    double frequency_hz,
    bool reverse);
pybind11::dict channel_field_coupled_dd(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor edge1_position,
    torch::Tensor edge1_direction,
    torch::Tensor edge1_n0,
    torch::Tensor edge1_n1,
    torch::Tensor edge1_exterior,
    torch::Tensor edge2_position,
    torch::Tensor edge2_direction,
    torch::Tensor edge2_n0,
    torch::Tensor edge2_n1,
    torch::Tensor edge2_exterior,
    torch::Tensor tx_power,
    torch::Tensor tx_polarization,
    torch::Tensor rx_polarization,
    torch::Tensor wedge1_eps_r0,
    torch::Tensor wedge1_sigma_e0,
    torch::Tensor wedge1_mu_r0,
    torch::Tensor wedge1_gain0,
    torch::Tensor wedge1_thickness0,
    torch::Tensor wedge1_eps_r1,
    torch::Tensor wedge1_sigma_e1,
    torch::Tensor wedge1_mu_r1,
    torch::Tensor wedge1_gain1,
    torch::Tensor wedge1_thickness1,
    torch::Tensor wedge2_eps_r0,
    torch::Tensor wedge2_sigma_e0,
    torch::Tensor wedge2_mu_r0,
    torch::Tensor wedge2_gain0,
    torch::Tensor wedge2_thickness0,
    torch::Tensor wedge2_eps_r1,
    torch::Tensor wedge2_sigma_e1,
    torch::Tensor wedge2_mu_r1,
    torch::Tensor wedge2_gain1,
    torch::Tensor wedge2_thickness1,
    torch::Tensor edge1_line_min,
    torch::Tensor edge1_line_max,
    torch::Tensor edge2_line_min,
    torch::Tensor edge2_line_max,
    double frequency_hz);

#undef CHANNEL_DIFFRACTION_WEDGE_COMMON_ARGUMENTS
#undef CHANNEL_TRANSMISSION_SEQUENCE_ARGUMENTS
pybind11::dict channel_field_source_amplitude_scale(
    torch::Tensor field_vector,
    torch::Tensor tx_power);
pybind11::dict channel_field_source_amplitude_scale_backward(
    torch::Tensor tx_power,
    torch::Tensor grad_path_field_vector);
pybind11::dict channel_field_source_amplitude_scale_jvp(
    torch::Tensor tx_power,
    torch::Tensor tangent_field_vector);
pybind11::dict channel_field_rough_reflection_scale(
    torch::Tensor field_vector,
    torch::Tensor coefficient,
    torch::Tensor path_field,
    torch::Tensor path_gain,
    torch::Tensor positions,
    torch::Tensor normals,
    torch::Tensor source,
    torch::Tensor sigma_b,
    torch::Tensor rough_b,
    torch::Tensor replaced,
    double frequency_hz);
pybind11::dict channel_field_rough_reflection_scale_backward(
    torch::Tensor field_vector,
    torch::Tensor coefficient,
    torch::Tensor path_field,
    torch::Tensor path_gain,
    torch::Tensor positions,
    torch::Tensor normals,
    torch::Tensor source,
    torch::Tensor sigma_b,
    torch::Tensor rough_b,
    torch::Tensor replaced,
    double frequency_hz,
    pybind11::object grad_field_vector,
    pybind11::object grad_coefficient,
    pybind11::object grad_path_field,
    pybind11::object grad_path_gain,
    bool need_field,
    bool need_geometry,
    bool need_frequency);
pybind11::dict channel_field_rough_reflection_scale_jvp(
    torch::Tensor field_vector,
    torch::Tensor coefficient,
    torch::Tensor path_field,
    torch::Tensor path_gain,
    torch::Tensor positions,
    torch::Tensor normals,
    torch::Tensor source,
    torch::Tensor sigma_b,
    torch::Tensor rough_b,
    torch::Tensor replaced,
    double frequency_hz,
    pybind11::object tangent_field_vector,
    pybind11::object tangent_coefficient,
    pybind11::object tangent_path_field,
    pybind11::object tangent_path_gain,
    pybind11::object tangent_positions,
    pybind11::object tangent_normals,
    pybind11::object tangent_source,
    double tangent_frequency);

void register_fields(pybind11::module_ &module) {
    module.def(
        "field_free_space",
        &channel_field_free_space,
        "Evaluate the canonical complex3 free-space field and receiver projection.");
    module.def(
        "field_project_complex3",
        &channel_field_project_complex3,
        "Project a world-Cartesian complex3 field onto a receiver polarization.");
    module.def(
        "field_reflection_sequence",
        &channel_field_reflection_sequence,
        "Transport a canonical complex3 field through a finite-slab reflection sequence.");
    module.def(
        "field_transmission_sequence",
        &channel_field_transmission_sequence,
        "Transport a canonical complex3 field through a thin_sheet transmission sequence.");
    module.def(
        "field_coupled_rd",
        &channel_field_coupled_rd,
        "Transport a canonical complex3 field through coupled R-D or D-R events.");
    module.def(
        "field_coupled_dd",
        &channel_field_coupled_dd,
        "Transport a canonical complex3 field through coupled double-diffraction "
        "(TX->e1->e2->RX) events (ADR-013).");
    module.def(
        "field_free_space_fwd64",
        &channel_field_free_space_fwd64,
        "Float64 free-space field forward for the strict gradcheck AD path.");
    module.def(
        "field_free_space_backward",
        &channel_field_free_space_backward,
        "Fixed-topology VJP of the free-space field (frequency, endpoints, and "
        "the arrival-direction seam).");
    module.def(
        "field_free_space_jvp",
        &channel_field_free_space_jvp,
        "Fixed-topology JVP of the free-space field (frequency, endpoints, and "
        "the arrival-direction seam).");
    module.def(
        "field_reflection_sequence_backward",
        &channel_field_reflection_sequence_backward,
        "Fixed-topology VJP of the reflection sequence (materials, frequency, "
        "geometry, and the arrival-direction seam).");
    module.def(
        "field_reflection_sequence_jvp",
        &channel_field_reflection_sequence_jvp,
        "Fixed-topology JVP of the reflection sequence (materials, frequency, "
        "geometry, and the arrival-direction seam).");
    module.def(
        "field_transmission_sequence_backward",
        &channel_field_transmission_sequence_backward,
        "Fixed-topology VJP of the transmission sequence (CSR layers, frequency, geometry).");
    module.def(
        "field_transmission_sequence_jvp",
        &channel_field_transmission_sequence_jvp,
        "Fixed-topology JVP of the transmission sequence (CSR layers, frequency, geometry).");
    module.def(
        "field_diffraction_wedge",
        &channel_field_diffraction_wedge,
        "Re-evaluate RayD's order-1 UTD wedge field from the frozen topology (plan 07 AD-4).");
    module.def(
        "field_diffraction_wedge_backward",
        &channel_field_diffraction_wedge_backward,
        "Fixed-topology VJP of the wedge field (face materials, frequency, endpoints).");
    module.def(
        "field_diffraction_wedge_jvp",
        &channel_field_diffraction_wedge_jvp,
        "Fixed-topology JVP of the wedge field (face materials, frequency, endpoints).");
    module.def(
        "field_coupled_rd_backward",
        &channel_field_coupled_rd_backward,
        "Fixed-topology VJP of the coupled R-D field (12 material scalars, frequency, geometry).");
    module.def(
        "field_coupled_rd_jvp",
        &channel_field_coupled_rd_jvp,
        "Fixed-topology JVP of the coupled R-D field (12 material scalars, frequency, geometry).");
    module.def(
        "field_coupled_dd_backward",
        &channel_field_coupled_dd_backward,
        "Fixed-topology VJP of the coupled double-diffraction field "
        "(16 wedge material scalars, frequency, tx/rx geometry) (ADR-013).");
    module.def(
        "field_coupled_dd_jvp",
        &channel_field_coupled_dd_jvp,
        "Fixed-topology JVP of the coupled double-diffraction field "
        "(16 wedge material scalars, frequency, tx/rx geometry) (ADR-013).");
    module.def(
        "field_project_complex3_backward",
        &channel_field_project_complex3_backward,
        "VJP of the receiver projection (field vector and arrival direction).");
    module.def(
        "field_project_complex3_jvp",
        &channel_field_project_complex3_jvp,
        "JVP of the receiver projection (field vector and arrival direction).");
    module.def(
        "field_source_amplitude_scale",
        &channel_field_source_amplitude_scale,
        "Apply the source amplitude sqrt(tx_power) onto a transported complex3 "
        "field vector (ADR-039).");
    module.def(
        "field_source_amplitude_scale_backward",
        &channel_field_source_amplitude_scale_backward,
        "VJP of the source-amplitude scale (field vector; tx_power is frozen).");
    module.def(
        "field_source_amplitude_scale_jvp",
        &channel_field_source_amplitude_scale_jvp,
        "JVP of the source-amplitude scale (field vector; tx_power is frozen).");
    module.def(
        "field_rough_reflection_scale",
        &channel_field_rough_reflection_scale,
        "Apply the rough-surface coherent attenuation C_r onto the reflection field outputs (ADR-010 op 3).");
    module.def(
        "field_rough_reflection_scale_backward",
        &channel_field_rough_reflection_scale_backward,
        "Fixed-topology VJP of the rough-reflection scale (frequency and hit geometry).");
    module.def(
        "field_rough_reflection_scale_jvp",
        &channel_field_rough_reflection_scale_jvp,
        "Fixed-topology JVP of the rough-reflection scale (frequency and hit geometry).");
    module.def(
        "coupled_rd_prepare",
        &channel_coupled_rd_prepare_cuda,
        "Fixed-winner coupled stationary geometry re-solve (image source, edge point, wall crossing).");
    module.def(
        "coupled_rd_prepare_backward",
        &channel_coupled_rd_prepare_backward,
        "VJP of the coupled stationary geometry re-solve (endpoints).");
    module.def(
        "coupled_rd_prepare_jvp",
        &channel_coupled_rd_prepare_jvp,
        "JVP of the coupled stationary geometry re-solve (endpoints).");
}
